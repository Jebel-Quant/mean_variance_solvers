r"""Every identity in the quadprog note, checked against the running solver.

The note proves a short list of identities and then reasons entirely from them:
the two invariants that define the carried matrix, the closed forms for the two
search directions, the exact objective increment, and the recovery formulas for a
warm start. A proof establishes those in exact arithmetic. It does not establish
that the shipped code satisfies them, and that is a separate question with a
non-obvious answer, because the implementation reaches them through a packed
triangular layout addressed by hand and through BLAS calls that write in place
into their arguments' buffers. A sign error in the packed indexing, or a `dger`
handed a non-contiguous view, would leave the mathematics intact and the code
wrong.

So this script drives the solver's own internals -- not a reimplementation of
them -- into states with a nonempty working set, and evaluates both sides of every
identity the paper states. It reports the worst residual per identity over all
problems, scaled so the numbers are comparable:

    rel(P, Q) = ||P - Q||_inf / max(1, ||Q||_inf)

Reaching into the private modules is deliberate and is the point: the claims are
about `J`, packed `R`, `d`, `z` and `r`, which are internal by design. A test
written against the public surface could not see them.

Usage:
    uv run python -m quadprog_note.experiment_qp_identities   # from experiment/

Outputs:
    tables/quadprog_identities.tex        tabular rows, one per identity
    tables/quadprog_identities_defs.tex   headline numbers as \newcommand macros

NumPy + SciPy + cvx-quadprog only; no figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.util.runner import SMOKE, output_dirs

from cvx.quadprog import Sweep, solve_qp
from cvx.quadprog._qr import qr_delete, qr_insert
from cvx.quadprog._setup import _factorize
from cvx.quadprog._steps import _step_directions

HERE = Path(__file__).resolve().parents[1]
GRAPHS, TABLES = output_dirs(HERE)

# Sizes to sweep. Kept modest on purpose: these are exactness checks, not timings,
# and a residual that survives n = 120 with a working set of 60 is not going to be
# rescued by n = 1200. Breadth over depth -- many shapes, each cheap.
SIZES = [(12, 8), (40, 25)] if SMOKE else [(12, 8), (30, 20), (60, 40), (120, 80)]
SEEDS = 2 if SMOKE else 8

# Fraction of the available columns to drive into the working set before checking.
FILL = 0.5


def rel(p, q) -> float:
    """Return a scaled sup-norm residual between two arrays of equal shape."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    denom = max(1.0, float(np.max(np.abs(q))) if q.size else 1.0)
    if p.size == 0:
        return 0.0
    return float(np.max(np.abs(p - q))) / denom


def unpack(packed, k) -> np.ndarray:
    """Return the dense k-by-k upper triangle from the solver's packed columns.

    Column j occupies j+1 entries at offset j(j+1)/2, which is the layout the
    note records and the implementation addresses directly.
    """
    out = np.zeros((k, k))
    for j in range(k):
        start = j * (j + 1) // 2
        out[: j + 1, j] = packed[start : start + j + 1]
    return out


def problem(rng, n, m):
    """Return a strictly convex QP with a well-conditioned Hessian."""
    b_mat = rng.standard_normal((n, n))
    g = b_mat @ b_mat.T + n * np.eye(n)
    a = rng.standard_normal(n)
    c = rng.standard_normal((n, m))
    b = rng.standard_normal(m)
    return g, a, c, b


class Worst:
    """Running worst-case residual per named identity, with a problem count."""

    def __init__(self) -> None:
        self._worst: dict[str, float] = {}
        self._count: dict[str, int] = {}

    def add(self, key, value) -> None:
        value = float(value)
        self._worst[key] = max(self._worst.get(key, 0.0), value)
        self._count[key] = self._count.get(key, 0) + 1

    def worst(self, key) -> float:
        return self._worst[key]

    def count(self, key) -> int:
        return self._count[key]

    def overall(self) -> float:
        return max(self._worst.values())

    def keys(self):
        return self._worst.keys()


def check_state(w, g, a, c, active, j_mat, packed, ginv):
    """Check every identity that a factorisation with a nonempty working set satisfies."""
    n = g.shape[0]
    k = len(active)
    a_mat = c[:, active]
    r_mat = unpack(packed, k)
    j1, j2 = j_mat[:, :k], j_mat[:, k:]

    # --- Section 3: the two invariants and what the blocks mean -----------------
    w.add("inv1", rel(j_mat.T @ g @ j_mat, np.eye(n)))
    w.add("inv2", rel(j_mat.T @ a_mat, np.vstack([r_mat, np.zeros((n - k, k))])))
    w.add("blocks_i", rel(r_mat.T @ r_mat, a_mat.T @ ginv @ a_mat))
    w.add("blocks_ii_a", rel(j2.T @ a_mat, np.zeros((n - k, k))))
    w.add("blocks_ii_b", rel(j2.T @ g @ j2, np.eye(n - k)))
    w.add("blocks_iii", rel(j1, ginv @ a_mat @ np.linalg.inv(r_mat)))

    dual_hessian = a_mat.T @ ginv @ a_mat
    proj = ginv @ a_mat @ np.linalg.inv(dual_hessian) @ a_mat.T @ ginv
    w.add("reduced", rel(j2 @ j2.T, ginv - proj))

    # --- Section 4: the directions, taken from the solver's own routine ---------
    for col in range(c.shape[1]):
        if col in active:
            continue
        nstar = c[:, col]
        dv, zv, rv, ztn = _step_directions(j_mat, packed, k, False, 0.0, 0, nstar)
        if ztn is None:  # primal cannot move; the closed forms below still hold
            ztn = float(zv @ nstar)

        w.add("rclosed", rel(rv, np.linalg.solve(dual_hessian, a_mat.T @ ginv @ nstar)))
        w.add("zclosed", rel(zv, (ginv - proj) @ nstar))
        w.add("dir_i", rel(a_mat.T @ zv, np.zeros(k)))
        w.add("dir_ii", max(rel(ztn, dv[k:] @ dv[k:]), rel(ztn, zv @ g @ zv)))
        w.add("dir_iii", rel(g @ zv, nstar - a_mat @ rv))

        # --- the affine path and the objective increment ------------------------
        # Build a state that satisfies stationarity with the entering constraint
        # carrying multiplier u_star, which is what Proposition "path" assumes.
        u_vec = np.ones(k)
        for u_star in (0.0, 1.7):
            x = ginv @ (a + a_mat @ u_vec + u_star * nstar)
            w.add("path", rel(g @ x - a, a_mat @ u_vec + u_star * nstar))
            for t in (0.41, -0.63):
                lhs = obj(g, a, x + t * zv) - obj(g, a, x)
                rhs = t * (t / 2.0 + u_star) * ztn
                w.add("increment", rel(lhs, rhs))
        break  # one entering constraint per state is enough; the sweep is over states

    return k


def obj(g, a, x) -> float:
    """Return the objective value the note writes as f(x)."""
    return 0.5 * float(x @ g @ x) - float(a @ x)


def check_updates(w, g, c, active, j_mat, packed, entering):
    """Check that insertion and deletion preserve both invariants, and Prop. "rank"."""
    n = g.shape[0]
    k = len(active)

    # --- insertion -------------------------------------------------------------
    j_ins, p_ins = j_mat.copy(), packed.copy()
    dv = j_ins.T @ c[:, entering]
    d_tail_norm = float(np.linalg.norm(dv[k:]))
    qr_insert(k + 1, dv, j_ins, p_ins)
    a_ins = np.column_stack([c[:, active], c[:, entering]])
    r_ins = unpack(p_ins, k + 1)

    w.add("ins_inv1", rel(j_ins.T @ g @ j_ins, np.eye(n)))
    w.add("ins_inv2", rel(j_ins.T @ a_ins, np.vstack([r_ins, np.zeros((n - k - 1, k + 1))])))
    # Proposition "rank": |alpha| = ||d_2||, so alpha != 0 whenever z != 0.
    w.add("rank", rel(abs(r_ins[k, k]), d_tail_norm))

    # --- deletion, at every position ------------------------------------------
    for pos in range(1, k + 1):
        j_del, p_del = j_mat.copy(), packed.copy()
        qr_delete(k, pos, j_del, p_del)
        a_del = np.delete(c[:, active], pos - 1, axis=1)
        r_del = unpack(p_del, k - 1)
        w.add("del_inv1", rel(j_del.T @ g @ j_del, np.eye(n)))
        w.add("del_inv2", rel(j_del.T @ a_del, np.vstack([r_del, np.zeros((n - k + 1, k - 1))])))


def check_kkt(w, g, a, c, b, meq, sol):
    """Check the returned point against the conditions Prop. "sufficient" calls proof."""
    slack = c.T @ sol.x - b
    w.add("kkt_stat", rel(g @ sol.x - a, c @ sol.lagrangian))
    if meq:
        w.add("kkt_eq", rel(slack[:meq], np.zeros(meq)))
    # One-sided conditions: report the depth of any violation, zero when satisfied.
    scale = max(1.0, float(np.max(np.abs(b))))
    w.add("kkt_ineq", max(0.0, -float(np.min(slack[meq:], initial=0.0))) / scale)
    w.add("kkt_sign", max(0.0, -float(np.min(sol.lagrangian[meq:], initial=0.0))) / scale)
    w.add("kkt_comp", float(np.max(np.abs(sol.lagrangian[meq:] * slack[meq:]), initial=0.0)) / scale)


def check_recovery(w, g, c, b, a_vec):
    """Check Prop. "recover": the warm-start formulas reproduce the subproblem solution."""
    sweep = Sweep(g, c, b, meq=0)
    sol = sweep.solve(a_vec)
    active = sol.iact - 1
    if active.size == 0:
        return
    ginv = np.linalg.inv(g)
    a_mat = c[:, active]
    lam = sol.lagrangian[active]
    xu = ginv @ a_vec
    w.add("recover_u", rel(lam, np.linalg.solve(a_mat.T @ ginv @ a_mat, b[active] - a_mat.T @ xu)))
    w.add("recover_x", rel(sol.x, xu + ginv @ a_mat @ lam))


def check_scale_invariance(w, g, a, c, b, meq, rng):
    """Check Prop. "scale": rescaling a constraint changes neither answer nor trajectory."""
    base = solve_qp(g, a, c, b, meq)
    col = int(rng.integers(meq, c.shape[1]))
    gamma = float(10.0 ** rng.uniform(-3, 3))
    c2, b2 = c.copy(), b.copy()
    c2[:, col] *= gamma
    b2[col] *= gamma
    other = solve_qp(g, a, c2, b2, meq)
    w.add("scale_x", rel(other.x, base.x))
    # The trajectory, not just the answer: same number of additions and removals.
    w.add("scale_iters", rel(other.iterations, base.iterations))


def main() -> None:
    w = Worst()
    n_problems = 0
    n_states = 0

    for n, m in SIZES:
        for seed in range(SEEDS):
            rng = np.random.default_rng((n * 1000) + seed)
            g, a, c, b = problem(rng, n, m)
            ginv = np.linalg.inv(g)
            n_problems += 1

            # Drive the factorisation into a state with a nonempty working set by
            # inserting columns, exactly as the solver's own full steps do.
            j_mat, _xu = _factorize(g, a, False)
            r_cap = min(n, m)
            packed = np.zeros(r_cap * (r_cap + 1) // 2)
            order = list(rng.permutation(m))
            active: list[int] = []
            target = max(2, int(FILL * min(n, m)))

            for col in order:
                if len(active) >= target:
                    break
                dv = j_mat.T @ c[:, col]
                if np.linalg.norm(dv[len(active) :]) < 1e-10:
                    continue  # dependent on the current set; the solver would not add it
                qr_insert(len(active) + 1, dv, j_mat, packed)
                active.append(col)

                if len(active) >= 2:
                    check_state(w, g, a, c, active, j_mat, packed, ginv)
                    spare = [i for i in range(m) if i not in active]
                    if spare:
                        check_updates(w, g, c, active, j_mat, packed, spare[0])
                    n_states += 1

            for meq in (0, min(2, m)):
                check_kkt(w, g, a, c, b, meq, solve_qp(g, a, c, b, meq))
                check_scale_invariance(w, g, a, c, b, meq, rng)

            # Warm start needs a family the cached set stays optimal for; bounds
            # straddling the unconstrained minimum give one.
            xu = ginv @ a
            c_box = np.eye(n)
            b_box = xu - 0.5
            b_box[: max(1, n // 4)] = xu[: max(1, n // 4)] + 0.25
            check_recovery(w, g, c_box, b_box, a)

    emit(w, n_problems, n_states)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# (macro-safe key, LaTeX label, human description). Order is the paper's.
ROWS = [
    ("inv1", r"\eqref{eq:inv1}", r"$J^\top G J = I$"),
    ("inv2", r"\eqref{eq:inv2}", r"$J^\top A = [R;\,0]$"),
    ("blocks_i", r"Prop.~\ref{prop:blocks}(i)", r"$R^\top R = A^\top G^{-1} A$"),
    ("blocks_ii_a", r"Prop.~\ref{prop:blocks}(ii)", r"$J_2^\top A = 0$"),
    ("blocks_ii_b", r"Prop.~\ref{prop:blocks}(ii)", r"$J_2^\top G J_2 = I$"),
    ("blocks_iii", r"Prop.~\ref{prop:blocks}(iii)", r"$J_1 = G^{-1} A R^{-1}$"),
    ("reduced", r"\eqref{eq:reduced}", r"$J_2 J_2^\top$ is the reduced inverse Hessian"),
    ("rclosed", r"\eqref{eq:rclosed}", r"$r = (A^\top G^{-1}A)^{-1} A^\top G^{-1} n_*$"),
    ("zclosed", r"\eqref{eq:zclosed}", r"$z$ is the $G$-projection of $G^{-1}n_*$"),
    ("dir_i", r"Lem.~\ref{lem:directions}(i)", r"$A^\top z = 0$"),
    ("dir_ii", r"Lem.~\ref{lem:directions}(ii)", r"$n_*^\top z = \norm{d_2}^2 = z^\top G z$"),
    ("dir_iii", r"Lem.~\ref{lem:directions}(iii)", r"$Gz = n_* - Ar$"),
    ("path", r"Prop.~\ref{prop:path}", r"stationarity along the path"),
    ("increment", r"Prop.~\ref{prop:increment}", r"objective increment, both signs of $t$"),
    ("ins_inv1", r"\S\ref{ssec:insert}", r"insertion preserves \eqref{eq:inv1}"),
    ("ins_inv2", r"\S\ref{ssec:insert}", r"insertion preserves \eqref{eq:inv2}"),
    ("rank", r"Prop.~\ref{prop:rank}", r"$|\alpha| = \norm{d_2}$"),
    ("del_inv1", r"\S\ref{ssec:delete}", r"deletion preserves \eqref{eq:inv1}"),
    ("del_inv2", r"\S\ref{ssec:delete}", r"deletion preserves \eqref{eq:inv2}"),
    ("kkt_stat", r"\eqref{eq:stat}", r"$Gx - a = C\lambda$ at the returned point"),
    ("kkt_eq", r"\eqref{eq:pfeas}", r"$c_i^\top x = b_i$ on equalities"),
    ("kkt_ineq", r"\eqref{eq:pfeas}", r"$C^\top x \ge b$"),
    ("kkt_sign", r"\eqref{eq:dfeas}", r"$\lambda_i \ge 0$ on inequalities"),
    ("kkt_comp", r"\eqref{eq:comp}", r"complementarity"),
    ("scale_x", r"Prop.~\ref{prop:scale}", r"rescaling a constraint: same minimiser"),
    ("scale_iters", r"Prop.~\ref{prop:scale}", r"rescaling a constraint: same trajectory"),
    ("recover_u", r"Prop.~\ref{prop:recover}", r"warm-start multipliers"),
    ("recover_x", r"Prop.~\ref{prop:recover}", r"warm-start iterate"),
]


def sci(x) -> str:
    """Format a residual as LaTeX math, with exact zero kept as a bare 0."""
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(x)))
    mantissa = x / 10.0**exponent
    if round(mantissa, 1) >= 10.0:  # 9.99e-16 must print as 1.0e-15, not 10.0e-16
        mantissa, exponent = mantissa / 10.0, exponent + 1
    return f"{mantissa:.1f}\\times 10^{{{exponent}}}"


def emit(w, n_problems, n_states) -> None:
    lines = []
    for key, label, desc in ROWS:
        if key not in w.keys():
            continue
        lines.append(f"{label} & {desc} & ${sci(w.worst(key))}$ & {w.count(key)} \\\\\n")

    path = TABLES / "quadprog_identities.tex"
    path.write_text(
        "% Generated by quadprog_note/experiment_qp_identities.py -- do not edit by hand.\n"
        f"\\def\\quadprogIdentityRows{{%\n{''.join(lines)}}}\n"
    )
    print(f"Saved {path}")

    worst_overall = w.overall()
    defs = TABLES / "quadprog_identities_defs.tex"
    defs.write_text(
        "% Generated by quadprog_note/experiment_qp_identities.py -- do not edit by hand.\n"
        f"\\newcommand{{\\qpIdentProblems}}{{{n_problems}}}\n"
        f"\\newcommand{{\\qpIdentStates}}{{{n_states}}}\n"
        f"\\newcommand{{\\qpIdentChecks}}{{{sum(w.count(k) for k in w.keys())}}}\n"
        f"\\newcommand{{\\qpIdentCount}}{{{len([k for k, _, _ in ROWS if k in w.keys()])}}}\n"
        f"\\newcommand{{\\qpIdentWorst}}{{{sci(worst_overall)}}}\n"
        f"\\newcommand{{\\qpIdentSizes}}{{{', '.join(str(n) for n, _ in SIZES)}}}\n"
    )
    print(f"Saved {defs}")

    print(f"\n{'identity':<14} {'worst rel. residual':>22} {'checks':>8}")
    print("-" * 46)
    for key, _label, _desc in ROWS:
        if key in w.keys():
            print(f"{key:<14} {w.worst(key):>22.3e} {w.count(key):>8}")
    print("-" * 46)
    print(f"{'overall':<14} {worst_overall:>22.3e}")
    print(f"\n{n_problems} problems, {n_states} factorisation states, "
          f"{sum(w.count(k) for k in w.keys())} checks.")


if __name__ == "__main__":
    main()
