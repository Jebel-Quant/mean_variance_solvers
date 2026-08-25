r"""The dual active-set method against solvers that are not active-set methods.

The rest of this paper's experiments check the solver against its own claims. That
establishes correctness and says nothing about whether the method is worth using,
which is a comparative question. This script answers it against three
alternatives, on the constraint shapes Section 1 names as the method's territory:
dense problems of small to moderate size.

  * the reference C implementation of the same algorithm, when a wheel for it is
    installed -- the honest baseline, since it is the same method and differs only
    in language and in the choices the companion implementation examines;
  * OSQP, an operator-splitting method;
  * Clarabel, an interior-point method.

The comparison a speed table alone would get wrong is accuracy. An active-set
method terminates at an exactly feasible point of the active-set lattice and
returns a KKT point to machine precision; a splitting method stops when a
tolerance is met, and an interior-point method approaches the boundary
asymptotically. Reporting time without residual would flatter whichever solver
was configured loosest. So every solver is asked for a comparable accuracy and
the residual it achieves is reported beside its time, in the same scaled sup-norm
the certificate of Section 7 uses. A reader can then see what each solver's time
bought.

Usage:
    uv run python -m quadprog_note.experiment_qp_compare   # from experiment/

Outputs:
    graphs/quadprog_compare.pdf        time vs n, one panel per family
    tables/quadprog_compare.tex        tabular rows: time and residual
    tables/quadprog_compare_defs.tex   headline numbers as \newcommand macros

NumPy + SciPy + Matplotlib + cvx-quadprog + osqp + clarabel; the C reference is
optional and its row is omitted with a printed note when it is absent.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from common.util.runner import SMOKE, output_dirs

from cvx.quadprog import solve_qp

HERE = Path(__file__).resolve().parents[1]
GRAPHS, TABLES = output_dirs(HERE)

mpl.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150})

SIZES = [50, 100] if SMOKE else [50, 100, 200, 400]
REPEATS = 1 if SMOKE else 3
FAMILIES = ["box", "budget + bounds", "dense $C$"]

# Accuracy asked of the iterative solvers. Set tight on purpose: the active-set
# method returns an exact KKT point, so comparing against a solver at its default
# 1e-3 would be comparing different problems. Prose reports what this costs.
TOL = 1e-9
MAX_ITER = 200_000

try:  # optional: needs a wheel or a C toolchain, which is what this package avoids
    import quadprog as quadprog_c
except ImportError:  # pragma: no cover - depends on the host
    quadprog_c = None


def hessian(rng, n):
    b_mat = rng.standard_normal((n, n))
    return b_mat @ b_mat.T + n * np.eye(n)


def family_problem(name, rng, n):
    """Return (G, a, C, b, meq) in the column-wise form ``C^T x >= b``."""
    g = hessian(rng, n)
    a = rng.standard_normal(n)
    xu = np.linalg.solve(g, a)

    if name == "box":
        c = np.hstack([np.eye(n), -np.eye(n)])
        b = np.concatenate([xu - 0.5, -(xu + 0.5)])
        return g, a, c, b, 0
    if name == "budget + bounds":
        c = np.hstack([np.ones((n, 1)), np.eye(n)])
        b = np.concatenate([[1.0], np.zeros(n)])
        return g, a, c, b, 1
    if name == "dense $C$":
        m = max(2, n // 2)
        c = rng.standard_normal((n, m))
        b = c.T @ xu + rng.standard_normal(m) * 0.5
        return g, a, c, b, 0
    raise ValueError(name)


def kkt_residual(g, a, c, b, meq, x):
    """Return the scaled sup-norm KKT residual, with multipliers recovered by least squares.

    A solver that does not report multipliers, or reports them in another
    convention, must not be penalised for that, so the multipliers are recovered
    here from the returned ``x``: take the constraints it leaves near-active and
    solve the stationarity system for their multipliers in the least-squares
    sense. That is the most favourable reading of any returned point, which is
    the reading a comparison should use.
    """
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    slack = c.T @ x - b
    near = np.abs(slack) <= 1e-6 * scale
    near[:meq] = True

    grad = g @ x - a
    if near.any():
        lam_active, *_ = np.linalg.lstsq(c[:, near], grad, rcond=None)
        lam = np.zeros(c.shape[1])
        lam[near] = lam_active
    else:
        lam = np.zeros(c.shape[1])

    parts = [
        float(np.max(np.abs(grad - c @ lam))),
        float(np.max(np.abs(slack[:meq]), initial=0.0)),
        max(0.0, -float(np.min(slack[meq:], initial=0.0))),
        max(0.0, -float(np.min(lam[meq:], initial=0.0))),
        float(np.max(np.abs(lam[meq:] * slack[meq:]), initial=0.0)),
    ]
    return max(parts) / scale


def as_osqp(g, a, c, b, meq):
    """Return OSQP's (P, q, A, l, u) for ``min 1/2 x'Gx - a'x  s.t.  C^T x >= b``."""
    lower = b.astype(float).copy()
    upper = np.full(len(b), np.inf)
    upper[:meq] = b[:meq]  # equalities: l = u = b
    return sp.csc_matrix(g), -a, sp.csc_matrix(c.T), lower, upper


def run_osqp(g, a, c, b, meq):
    import osqp
    p_mat, q, a_mat, lower, upper = as_osqp(g, a, c, b, meq)
    prob = osqp.OSQP()
    prob.setup(P=p_mat, q=q, A=a_mat, l=lower, u=upper, verbose=False,
               eps_abs=TOL, eps_rel=TOL, max_iter=MAX_ITER, polishing=True)
    return np.asarray(prob.solve().x, dtype=float)


def run_clarabel(g, a, c, b, meq):
    """Clarabel wants ``Ax + s = rhs`` with ``s`` in a cone; flip the sign of C^T."""
    import clarabel
    m = len(b)
    a_mat = sp.csc_matrix(np.vstack([-c.T[:meq], -c.T[meq:]]))
    rhs = np.concatenate([-b[:meq], -b[meq:]])
    cones = []
    if meq:
        cones.append(clarabel.ZeroConeT(meq))
    if m - meq:
        cones.append(clarabel.NonnegativeConeT(m - meq))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = TOL
    settings.tol_gap_rel = TOL
    settings.tol_feas = TOL
    solver = clarabel.DefaultSolver(sp.csc_matrix(g), -a, a_mat, rhs, cones, settings)
    return np.asarray(solver.solve().x, dtype=float)


def solvers():
    """Return the (label, callable) pairs to time, in table order."""
    out = [
        ("cvx-quadprog", lambda g, a, c, b, meq: solve_qp(g, a, c, b, meq).x),
        ("cvx-quadprog, \\texttt{fast}",
         lambda g, a, c, b, meq: solve_qp(g, a, c, b, meq, fast=True).x),
    ]
    if quadprog_c is not None:
        # The reference destroys G and a in place, so it is handed copies.
        out.append(("\\texttt{quadprog} (C)",
                    lambda g, a, c, b, meq: quadprog_c.solve_qp(
                        g.copy(), a.copy(), c, b, meq)[0]))
    out.append(("OSQP", run_osqp))
    out.append(("Clarabel", run_clarabel))
    return out


def timed(fn, g, a, c, b, meq):
    """Return (best wall time, x) over REPEATS calls, or (nan, None) on failure."""
    best, x = float("inf"), None
    for _ in range(REPEATS):
        start = time.perf_counter()
        try:
            out = fn(g, a, c, b, meq)
        except Exception as exc:  # a baseline declining is data, not a crash
            print(f"      failed: {type(exc).__name__}: {exc}")
            return float("nan"), None
        elapsed = time.perf_counter() - start
        if elapsed < best:
            best, x = elapsed, out
    return best, x


def run():
    results = {}
    for family in FAMILIES:
        for n in SIZES:
            rng = np.random.default_rng(1_000 * n + FAMILIES.index(family))
            g, a, c, b, meq = family_problem(family, rng, n)
            print(f"{family}, n = {n}")
            for label, fn in solvers():
                seconds, x = timed(fn, g, a, c, b, meq)
                residual = (kkt_residual(g, a, c, b, meq, x)
                            if x is not None else float("nan"))
                results[(family, n, label)] = {"time": seconds, "resid": residual}
                print(f"  {label:<30} {seconds * 1e3:>9.2f} ms   resid {residual:.2e}")
    return results


def figure(results) -> None:
    labels = [label for label, _ in solvers()]
    colours = dict(zip(labels, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
    markers = dict(zip(labels, ["o", "s", "^", "v", "D"]))

    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(2.5 * len(FAMILIES) + 0.6, 2.9),
                             sharey=True)
    for ax, family in zip(np.atleast_1d(axes), FAMILIES):
        for label in labels:
            ys = [results[(family, n, label)]["time"] * 1e3 for n in SIZES]
            ax.plot(SIZES, ys, color=colours[label], marker=markers[label],
                    markersize=4, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$n$")
        ax.set_title(family)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    np.atleast_1d(axes)[0].set_ylabel("per solve (ms)")
    np.atleast_1d(axes)[-1].legend(framealpha=0.9, fontsize=7)

    fig.tight_layout(pad=0.8)
    fig.savefig(GRAPHS / "quadprog_compare.pdf", bbox_inches="tight")
    fig.savefig(GRAPHS / "quadprog_compare.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved {GRAPHS / 'quadprog_compare.pdf'}")


def sci(x) -> str:
    if not np.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(x)))
    mantissa = x / 10.0**exponent
    # Guard against the rounding used for display, not the true mantissa: this
    # prints to zero decimals, so 9.5e-16 would otherwise read "10 x 10^-16".
    if round(mantissa) >= 10:
        mantissa, exponent = mantissa / 10.0, exponent + 1
    return f"{mantissa:.0f}\\times 10^{{{exponent}}}"


def emit(results) -> None:
    labels = [label for label, _ in solvers()]
    biggest = SIZES[-1]

    rows = []
    for family in FAMILIES:
        rows.append(f"\\multicolumn{{{1 + 2 * len(SIZES)}}}{{l}}{{\\emph{{{family}}}}} \\\\\n")
        for label in labels:
            cells = []
            for n in SIZES:
                cell = results[(family, n, label)]
                ms = cell["time"] * 1e3
                cells.append(f"{ms:.2f}" if np.isfinite(ms) else "--")
                cells.append(f"${sci(cell['resid'])}$")
            rows.append(f"{label} & " + " & ".join(cells) + " \\\\\n")
        rows.append("\\addlinespace\n")

    path = TABLES / "quadprog_compare.tex"
    path.write_text(
        "% Generated by quadprog_note/experiment_qp_compare.py -- do not edit by hand.\n"
        f"\\def\\quadprogCompareRows{{%\n{''.join(rows)}}}\n"
    )
    print(f"Saved {path}")

    def speedup(family, other):
        mine = results[(family, biggest, "cvx-quadprog")]["time"]
        theirs = results[(family, biggest, other)]["time"]
        return theirs / mine if np.isfinite(theirs) and np.isfinite(mine) else float("nan")

    worst_exact = max(results[(f, n, "cvx-quadprog")]["resid"]
                      for f in FAMILIES for n in SIZES)
    worst_osqp = max(results[(f, n, "OSQP")]["resid"]
                     for f in FAMILIES for n in SIZES)
    worst_clarabel = max(results[(f, n, "Clarabel")]["resid"]
                         for f in FAMILIES for n in SIZES)

    path = TABLES / "quadprog_compare_defs.tex"
    with open(path, "w") as fh:
        fh.write("% Generated by quadprog_note/experiment_qp_compare.py -- do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\qpCmpSizes}}{{{', '.join(str(n) for n in SIZES)}}}\n")
        fh.write(f"\\newcommand{{\\qpCmpNhi}}{{{biggest}}}\n")
        fh.write(f"\\newcommand{{\\qpCmpTol}}{{10^{{{int(np.log10(TOL))}}}}}\n")
        fh.write(f"\\newcommand{{\\qpCmpResidExact}}{{{sci(worst_exact)}}}\n")
        fh.write(f"\\newcommand{{\\qpCmpResidOsqp}}{{{sci(worst_osqp)}}}\n")
        fh.write(f"\\newcommand{{\\qpCmpResidClarabel}}{{{sci(worst_clarabel)}}}\n")
        fh.write(f"\\newcommand{{\\qpCmpVsOsqp}}{{{speedup('box', 'OSQP'):.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpCmpVsClarabel}}{{{speedup('box', 'Clarabel'):.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpCmpVsOsqpDense}}{{{speedup('dense $C$', 'OSQP'):.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpCmpVsClarabelDense}}{{{speedup('dense $C$', 'Clarabel'):.1f}}}\n")
        if quadprog_c is not None:
            ratio = speedup("box", "\\texttt{quadprog} (C)")
            fh.write(f"\\newcommand{{\\qpCmpVsRef}}{{{ratio:.1f}}}\n")
            fh.write("\\newcommand{\\qpCmpHasRef}{yes}\n")
        else:
            fh.write("\\newcommand{\\qpCmpHasRef}{no}\n")
    print(f"Saved {path}")

    if quadprog_c is None:
        print("\nNOTE: the C reference implementation is not installed, so its row is "
              "omitted. Install `quadprog` to include it.")


def main() -> None:
    results = run()
    figure(results)
    emit(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
