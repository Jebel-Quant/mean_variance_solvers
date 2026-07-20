"""Warm-started matrix-free efficient frontier for "Non-Negative Conjugate Gradients".

The parametric sweep a portfolio manager actually runs is the long-only
efficient frontier itself. This script traces it with the matrix-free CG inner
solve on a factor-structured covariance -- the application-grounded counterpart
to the synthetic warm-start property of Proposition 5.4 (``prop:warm``) and the
matrix-free operator abstraction of Section 3.

For a covariance in factor form Sigma = D + L L^T (diagonal specific risk plus a
rank-K factor part, the Factor/Woodbury row of Table 1) and a mean vector mu, the
frontier is swept in the risk-aversion gamma:

    min_{w >= 0, 1^T w = 1}  1/2 w^T Sigma w - gamma mu^T w,

the equality-augmented bound-constrained quadratic of Section 3 with A = Sigma,
b = gamma mu, B = 1^T, c = 1, so moving along the frontier only rescales the
linear term. Sigma is reached only through v -> D v + L (L^T v); it is never
formed, so the dense O(n^2) storage and O(n^3) factorisation a direct solve needs
are avoided. As gamma grows the optimal holding concentrates from the
min-variance support toward a single name, and adjacent frontier points share
most of their support -- so warm-starting each solve from the previous gamma's
(free set, weights) pair collapses the active-set outer loop, and steps whose
support is unchanged are solved in a single warm outer step (``prop:warm``).

The factor model is fitted to real S&P 500 returns; the scaling table extends it
to a calibrated universe large enough that a dense Sigma cannot be formed, the
regime the matrix-free inner solve is built for. MPRGP does not enter this
comparison: the budget 1^T w = 1 is an equality constraint outside its bound-only
scope.

Usage:
    uv run python -m nncg_note.experiment_nncg_frontier   # from experiment/

Outputs:
    graphs/nncg_frontier.pdf        the frontier and the cold-vs-warm outer count
    tables/nncg_frontier.tex        matrix-free warm-start scaling vs dense storage
    tables/nncg_frontier_defs.tex   headline numbers as \\newcommand macros

Reads the committed data/sp500_pct_returns.parquet; depends on pandas in
addition to NumPy/Matplotlib.
"""


from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cvx.linalg import DenseOperator, SymmetricOperator

from common.util.runner import SMOKE, output_dirs

from nncg import CG, ActiveSetConfig, ActiveSetSolver, Exact, KrylovConfig

HERE = Path(__file__).resolve().parents[1]  # experiment/ root (data, graphs, tables)
GRAPHS, TABLES = output_dirs(HERE)

mpl.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150})

COLOR_COLD = "#1f77b4"
COLOR_WARM = "#ff7f0e"

NFACTORS = 10                                         # rank of the factor covariance


def sci(x):
    """Format a positive number as LaTeX ``m\\times10^{e}`` (one mantissa digit)."""
    e = int(np.floor(np.log10(x)))
    return f"{x / 10.0 ** e:.1f}\\times10^{{{e}}}"


class FactorOperator(SymmetricOperator):
    """Matrix-free factor covariance ``Sigma = diag(d) + L L^T`` (rank ``K``).

    Applies ``Sigma v = d * v + L (L^T v)`` in ``O(nK)``; the ``n x n`` matrix is
    never formed. The free block ``Sigma_FF = D_F + L_F L_F^T`` is itself a factor
    operator, so ``restricted`` returns a genuinely pre-sliced view (rows of ``L``
    and entries of ``d``) rather than re-embedding each CG iteration. The
    direct-solve hooks are unused (inner="cg").
    """

    def __init__(self, d, L):
        self._d, self._L, self._dim = d, L, len(d)

    @property
    def n(self):
        return self._dim

    def matvec(self, v):
        v = np.asarray(v, dtype=float)
        return self._d * v + self._L @ (self._L.T @ v)

    def block_matvec(self, rows, cols, v):
        full = np.zeros(self._dim)
        full[cols] = v
        return self.matvec(full)[rows]

    def restricted(self, idx):
        idx = np.asarray(idx)
        return FactorOperator(self._d[idx], self._L[idx])

    def rcond_free(self, idx):
        raise NotImplementedError("FactorOperator supports matrix-free CG only")

    def solve_free(self, idx, rhs):
        raise NotImplementedError("FactorOperator supports matrix-free CG only")


def fit_factor_model(R, k):
    """Fit a rank-``k`` factor covariance ``diag(d) + L L^T`` to returns ``R``.

    Statistical factors: the top ``k`` eigenvectors of the sample covariance carry
    the systematic risk (``L = V_k Lambda_k^{1/2}``); the residual diagonal ``d``
    is the specific variance. ``d > 0`` makes ``Sigma`` SPD, so the P-matrix
    premise of Theorem 4.4 holds without a further ridge.
    """
    S = np.cov(R, rowvar=False)
    w, V = np.linalg.eigh(S)
    L = V[:, -k:] * np.sqrt(np.clip(w[-k:], 0.0, None))
    d = np.clip(np.diag(S) - np.sum(L * L, axis=1), 1e-10, None)
    return d, L, np.trace(S)


def new_solver():
    return ActiveSetSolver(CG(KrylovConfig(tol=1e-10, maxit=6000)),
                           ActiveSetConfig(tol=1e-9, max_outer=300))


def sweep(op, mu, gammas):
    """Trace the frontier cold and warm; return per-gamma records.

    Each solve runs cold (from zero) and warm-started from the previous gamma's
    ``(free, x)`` pair. Both return the same optimum; only the effort differs.
    """
    solver = new_solver()
    B = np.ones((1, op.n))
    c = np.array([1.0])
    rows = []
    prev_free = prev_x = None
    for g in gammas:
        b = g * mu
        t0 = time.perf_counter()
        rc = solver.solve_eq(op, b, B, c)
        t_cold = time.perf_counter() - t0
        if prev_free is None:
            rw, t_warm = rc, t_cold
        else:
            t0 = time.perf_counter()
            rw = solver.solve_eq(op, b, B, c, warm=(prev_free, prev_x))
            t_warm = time.perf_counter() - t0
        rows.append({
            "gamma": g, "w": rc.x, "free": rc.free,
            "held": int(np.count_nonzero(rc.free)),
            "supp": frozenset(np.flatnonzero(rc.free).tolist()),
            "cold_outer": rc.outer, "warm_outer": rw.outer,
            "cold_inner": rc.inner or 0, "warm_inner": rw.inner or 0,
            "cold_t": t_cold, "warm_t": t_warm,
            "diff": float(np.max(np.abs(rw.x - rc.x))),
        })
        prev_free, prev_x = rw.free, rw.x
    return rows


def aggregate(rows):
    """Warm-start statistics over the steps where warm-starting applies (k>=1)."""
    tail = rows[1:]
    stable = [r for i, r in enumerate(tail) if rows[i]["supp"] == r["supp"]]
    return {
        "nstep": len(tail),
        "cold_outer": float(np.mean([r["cold_outer"] for r in tail])),
        "warm_outer": float(np.mean([r["warm_outer"] for r in tail])),
        "stable": len(stable),
        "one_step": sum(r["warm_outer"] <= 1 for r in stable),
        "cold_t": sum(r["cold_t"] for r in tail),
        "warm_t": sum(r["warm_t"] for r in tail),
        "cold_inner": sum(r["cold_inner"] for r in tail),
        "warm_inner": sum(r["warm_inner"] for r in tail),
        "max_diff": max(r["diff"] for r in tail),
    }


def synth_universe(d0, L0, mu0, n_big, seed=0):
    """A calibrated large universe: bootstrap real assets and jitter them."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d0), size=n_big)
    L = L0[idx] * (1.0 + 0.05 * rng.standard_normal((n_big, L0.shape[1])))
    d = d0[idx] * np.exp(0.1 * rng.standard_normal(n_big))
    mu = mu0[idx] + 0.05 * float(np.std(mu0)) * rng.standard_normal(n_big)
    return d, L, mu


# ---------------------------------------------------------------------------
# Real S&P 500 frontier: matrix-free CG, cold vs warm
# ---------------------------------------------------------------------------

df = pd.read_parquet(HERE / "data" / "sp500_pct_returns.parquet").dropna(axis=1, how="any")
if SMOKE:
    df = df.iloc[:, :40]
    GAMMAS = np.concatenate([[0.0], np.geomspace(1e-3, 1e2, 7)])
    SCALE_NS = [80, 200]
else:
    GAMMAS = np.concatenate([[0.0], np.geomspace(1e-3, 1e2, 59)])
    SCALE_NS = [2000, 8000, 20000]

R = df.to_numpy(float)
T, n = R.shape
d0, L0, tr_S = fit_factor_model(R, min(NFACTORS, n - 1))
mu0 = R.mean(axis=0)
op = FactorOperator(d0, L0)
sys_frac = float(np.sum(L0 * L0) / tr_S)

# condition number of the (small) real operator, for reporting only
Sig_full = L0 @ L0.T + np.diag(d0)
ev = np.linalg.eigvalsh(Sig_full)
kappa = float(ev[-1] / ev[0])

print("=" * 78)
print(f"Warm-started matrix-free efficient frontier  (S&P 500, n={n}, K={L0.shape[1]} factors)")
print(f"sample {df.index[0].date()}..{df.index[-1].date()}, "
      f"systematic var frac={sys_frac:.2f}, kappa(Sigma)={kappa:.2e}")
print("=" * 78)

rows = sweep(op, mu0, GAMMAS)
agg = aggregate(rows)

# correctness: matrix-free CG matches a dense direct solve on the same Sigma
_ones, _one = np.ones((1, n)), np.array([1.0])
r_dense = ActiveSetSolver(Exact(), ActiveSetConfig(tol=1e-9, max_outer=300)).solve_eq(
    DenseOperator(Sig_full), 0.5 * mu0, _ones, _one)
r_mf = new_solver().solve_eq(op, 0.5 * mu0, _ones, _one)
mf_vs_dense = float(np.max(np.abs(r_mf.x - r_dense.x)))

# The claims Section 7 makes about this run, verified as CI checks.
assert mf_vs_dense < 1e-6, "matrix-free CG disagrees with the dense direct solve"
assert agg["one_step"] == agg["stable"], "a support-stable step needed >1 warm outer step"
assert agg["max_diff"] < 1e-6, "warm and cold starts returned different optima"

held = [r["held"] for r in rows]
print(f"held assets along the frontier: {min(held)}..{max(held)} of {n}")
print(f"cold outer/step {agg['cold_outer']:.1f}  warm outer/step {agg['warm_outer']:.1f}  "
      f"warm speedup {agg['cold_t'] / agg['warm_t']:.1f}x (inner {agg['cold_inner']}/{agg['warm_inner']})")
print(f"support-stable {agg['stable']}/{agg['nstep']}, of those single warm outer step: {agg['one_step']}")
print(f"matrix-free CG vs dense direct optimum: max|dx| = {mf_vs_dense:.1e}")


# ---------------------------------------------------------------------------
# Scaling: matrix-free warm-start where a dense Sigma cannot be formed
# ---------------------------------------------------------------------------

print("\nScaling to a calibrated universe (matrix-free CG only):")
print(f"{'n':>8}{'dense(GB)':>11}{'cold(s)':>9}{'warm(s)':>9}{'speedup':>9}")
scale_rows = []
for n_big in SCALE_NS:
    d, L, mu = synth_universe(d0, L0, mu0, n_big)
    a = aggregate(sweep(FactorOperator(d, L), mu, GAMMAS))
    gb = n_big * n_big * 8 / 1e9
    scale_rows.append({"n": n_big, "gb": gb, "cold_t": a["cold_t"], "warm_t": a["warm_t"],
                       "speedup": a["cold_t"] / a["warm_t"]})
    print(f"{n_big:>8}{gb:>11.2f}{a['cold_t']:>9.2f}{a['warm_t']:>9.2f}{a['cold_t'] / a['warm_t']:>8.1f}x")


# ---------------------------------------------------------------------------
# Figure: (a) the frontier, coloured by holdings; (b) cold vs warm outer count
# ---------------------------------------------------------------------------

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 2.9))

risk = np.array([float(np.sqrt(max(r["w"] @ op.matvec(r["w"]), 0.0))) for r in rows])
ret = np.array([float(mu0 @ r["w"]) for r in rows])
sc = axA.scatter(risk, ret, c=held, cmap="viridis", s=16, zorder=3)
axA.plot(risk, ret, color="0.6", linewidth=0.7, zorder=2)
cb = fig.colorbar(sc, ax=axA, pad=0.02)
cb.set_label("assets held", fontsize=7)
cb.ax.tick_params(labelsize=7)
axA.set_xlabel(r"Portfolio risk $\sqrt{w^\top\Sigma w}$")
axA.set_ylabel(r"Portfolio return $\mu^\top w$")
axA.set_title("Long-only frontier")
axA.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

k = np.arange(1, len(rows))
axB.plot(k, [r["cold_outer"] for r in rows[1:]], marker="o", markersize=3,
         color=COLOR_COLD, label="cold start")
axB.plot(k, [r["warm_outer"] for r in rows[1:]], marker="s", markersize=3,
         color=COLOR_WARM, label="warm start")
axB.set_xlabel(r"Frontier step (increasing $\gamma$)")
axB.set_ylabel("Active-set outer steps")
axB.set_title("Warm start collapses the outer loop")
axB.legend(framealpha=0.9)
axB.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

fig.tight_layout(pad=0.8)
fig.savefig(GRAPHS / "nncg_frontier.pdf", bbox_inches="tight")
fig.savefig(GRAPHS / "nncg_frontier.png", bbox_inches="tight", dpi=150)
print(f"\nSaved {GRAPHS / 'nncg_frontier.pdf'}")


# ---------------------------------------------------------------------------
# Scaling table and headline macros
# ---------------------------------------------------------------------------

with open(TABLES / "nncg_frontier.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg_frontier.py -- do not edit by hand.\n")
    fh.write("\\begin{tabular}{rrrrr}\n\\toprule\n")
    fh.write("$n$ & dense $\\Sigma$ (GB) & cold (s) & warm (s) & speedup \\\\\n\\midrule\n")
    for r in scale_rows:
        fh.write(f"{r['n']} & {r['gb']:.2f} & {r['cold_t']:.2f} & {r['warm_t']:.2f} "
                 f"& {r['speedup']:.0f}$\\times$ \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(f"Saved {TABLES / 'nncg_frontier.tex'}")

big = scale_rows[-1]
with open(TABLES / "nncg_frontier_defs.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg_frontier.py -- do not edit by hand.\n")
    fh.write(f"\\newcommand{{\\nncgFrontN}}{{{n}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontK}}{{{L0.shape[1]}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontStart}}{{{df.index[0].strftime('%B %Y')}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontEnd}}{{{df.index[-1].strftime('%B %Y')}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontSysFrac}}{{{100 * sys_frac:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontKappa}}{{${sci(kappa)}$}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontGammas}}{{{len(GAMMAS)}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontHeldMin}}{{{min(held)}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontHeldMax}}{{{max(held)}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmColdOuter}}{{{agg['cold_outer']:.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmOuter}}{{{agg['warm_outer']:.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmStable}}{{{agg['stable']}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmStableTot}}{{{agg['nstep']}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmOneStep}}{{{agg['one_step']}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmX}}{{{agg['cold_t'] / agg['warm_t']:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmInnerX}}{{{agg['cold_inner'] / max(agg['warm_inner'], 1):.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmErr}}{{{max(agg['max_diff'], mf_vs_dense):.0e}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontMaxN}}{{{big['n']}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontMaxGB}}{{{big['gb']:.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgFrontMaxX}}{{{big['speedup']:.0f}}}\n")
print(f"Saved {TABLES / 'nncg_frontier_defs.tex'}")
print("\nDone.")
