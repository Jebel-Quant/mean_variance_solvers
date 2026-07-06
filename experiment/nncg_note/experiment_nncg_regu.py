"""Official ill-posed test problem for "Non-Negative Conjugate Gradients".

Runs the paper's active-set solver (Algorithm 1, from nncg.py) on the `shaw`
problem from P. C. Hansen's Regularization Tools -- a standard severely
ill-posed test problem whose exact solution is a sum of two Gaussians and is
therefore strictly positive, a genuine non-negative least-squares instance.

The kernel matrix M is symmetric and numerically rank-deficient, so the Gram
operator A = M^T M is numerically singular. We add a small amount of noise to
the data (as any real measurement carries): the unconstrained least-squares
solution then oscillates into negative territory, so non-negativity becomes an
active regulariser -- the canonical use of NNLS on ill-posed problems -- and
the noisy right-hand side is inconsistent with the singular Gram.

This instantiates on an *official* problem the rank-deficiency claim of
Sections 5-6: at alpha = 0 the active-set loop cannot certify a solution (the
free block is singular, CG stalls on the inconsistent system), while any
alpha > 0 in the ridge/Tikhonov split A_alpha = (1-alpha)A + alpha*tau*I
restores the P-matrix property and the loop terminates with a KKT certificate,
agreeing with Lawson-Hanson on the regularised operator.

Usage:
    uv run python -m nncg_note.experiment_nncg_regu   # from the experiment/ directory

Outputs:
    tables/nncg_regu.tex        the ridge-split sweep (booktabs)
    tables/nncg_regu_defs.tex   headline numbers as \\newcommand macros
"""


from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from cvx.linalg import DenseOperator
from scipy.optimize import nnls as scipy_nnls
from common.util.runner import output_dirs

from nncg import CG, ActiveSetConfig, ActiveSetSolver, KrylovConfig, kkt_violation
from nncg_note.problems import phillips, shaw

HERE = Path(__file__).resolve().parents[1]  # experiment/ root
_, TABLES = output_dirs(HERE)


def ridge(gram, alpha):
    """Convex ridge split A_alpha = (1-alpha) A + alpha*tau*I, tau = tr(A)/n."""
    n = gram.shape[0]
    tau = np.trace(gram) / n
    return (1.0 - alpha) * gram + alpha * tau * np.eye(n)


def cond(A):
    """Spectral condition number via singular values (robust to singularity)."""
    sv = np.linalg.svd(A, compute_uv=False)
    smin = sv[-1]
    return np.inf if smin <= 0 else float(sv[0] / smin)


def lawson_hanson(A, b):
    """scipy NNLS on the SPD operator A via its Cholesky factor (A = L L^T)."""
    L = np.linalg.cholesky(A)                         # SPD => alpha > 0 only
    d = solve_triangular(L, b, lower=True)
    x, _ = scipy_nnls(L.T, d)
    return x


N = 128
ALPHAS = [0.0, 1e-6, 1e-4, 1e-2]
ALPHA_STAR = 1e-4                                     # headline regularisation
NOISE_REL = 1e-3                                      # relative data noise
CG_MAXIT = 2000                                       # matches the rank-def panel
MAX_OUTER = 30

M, x_true, d = shaw(N)
rng = np.random.default_rng(0)
g = rng.standard_normal(N)
d_noisy = d + NOISE_REL * np.linalg.norm(d) * g / np.linalg.norm(g)
gram = M.T @ M
rhs = M.T @ d_noisy
kappa0 = cond(gram)

print("=" * 74)
print(f"Hansen `shaw` ill-posed NNLS, n = {N}")
print(f"kappa(A) = kappa(M^T M) = {kappa0:.2e}  (numerically rank-deficient)")
print("=" * 74)
print(f"noise = {NOISE_REL:.0e} (relative), n = {N}\n")
print(f"{'alpha':>8}  {'kappa(A_a)':>11}  {'cert':>5}  {'outer':>5}  "
      f"{'inner':>7}  {'active':>6}  {'KKT':>9}  {'|x-x*|':>9}  {'|x-xLH|':>9}")

rows = []
for alpha in ALPHAS:
    A = ridge(gram, alpha) if alpha > 0 else gram
    ka = cond(A)
    op = DenseOperator(A)
    res = ActiveSetSolver(
        CG(KrylovConfig(tol=1e-10, maxit=CG_MAXIT)), ActiveSetConfig(max_outer=MAX_OUTER)
    ).solve(op, rhs)
    x = res.x
    kkt = kkt_violation(op, rhs, x)
    certified = bool(res.converged and kkt < 1e-6)
    active = int(np.sum(x <= 1e-8))                   # components pinned at zero
    err_star = float(np.max(np.abs(x - x_true)))
    if alpha > 0:
        x_lh = lawson_hanson(A, rhs)
        err_lh = float(np.max(np.abs(x - x_lh)))
    else:
        err_lh = np.nan
    rows.append((alpha, ka, certified, res.outer, res.inner,
                 active, kkt, err_star, err_lh))
    lh = "--" if np.isnan(err_lh) else f"{err_lh:.1e}"
    print(f"{alpha:>8.0e}  {ka:>11.2e}  {('yes' if certified else 'NO'):>5}  "
          f"{res.outer:>5}  {res.inner:>7}  {active:>6}  {kkt:>9.1e}  "
          f"{err_star:>9.1e}  {lh:>9}")

star = next(r for r in rows if r[0] == ALPHA_STAR)


# ---------------------------------------------------------------------------
# A second official problem: `phillips`, with a known non-negative solution
# (zero outside |t| < 3) -- recovery, and a large structured active set.
# ---------------------------------------------------------------------------

PHIL_N = 512
Mp, xp_true, dp = phillips(PHIL_N)
gp = np.random.default_rng(0).standard_normal(PHIL_N)
dp_noisy = dp + NOISE_REL * np.linalg.norm(dp) * gp / np.linalg.norm(gp)
Ap = ridge(Mp.T @ Mp, ALPHA_STAR)
bp = Mp.T @ dp_noisy
resp = ActiveSetSolver(
    CG(KrylovConfig(tol=1e-10, maxit=CG_MAXIT)), ActiveSetConfig(max_outer=MAX_OUTER)
).solve(DenseOperator(Ap), bp)
xp = resp.x
xp_lh = lawson_hanson(Ap, bp)
phil_active = int(np.sum(xp <= 1e-8))
phil_truezeros = int(np.sum(xp_true == 0.0))
phil_recov = float(np.max(np.abs(xp - xp_true)))
phil_err_lh = float(np.max(np.abs(xp - xp_lh)))
print(f"\nphillips n = {PHIL_N}: outer {resp.outer}, active {phil_active} "
      f"(true zeros {phil_truezeros}), recovery |x-x*| = {phil_recov:.1e}, "
      f"agree with LH = {phil_err_lh:.1e}")


# ---------------------------------------------------------------------------
# LaTeX table + macros
# ---------------------------------------------------------------------------

def sci(v):
    if v == 0.0:
        return "$0$"
    if not np.isfinite(v):
        return r"$\infty$"
    mant, expo = f"{v:.0e}".split("e")
    return f"${mant}\\cdot10^{{{int(expo)}}}$"


def expo_only(v):
    """Order of magnitude as 10^{k} for prose (e.g. 1e20 -> 10^{20})."""
    if not np.isfinite(v):
        return r"\infty"
    return f"10^{{{int(np.floor(np.log10(v)))}}}"


with open(TABLES / "nncg_regu.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg_regu.py -- do not edit by hand.\n")
    fh.write("\\begin{tabular}{lrcrrrr}\n\\toprule\n")
    fh.write("$\\alpha$ & $\\kappa(A_\\alpha)$ & certified & outer & CG inner "
             "& active & KKT resid. \\\\\n\\midrule\n")
    for alpha, ka, conv, outer, inner, active, kkt, _, _ in rows:
        a = "$0$" if alpha == 0 else sci(alpha)
        c = "yes" if conv else "no"
        fh.write(f"{a} & {sci(ka)} & {c} & {outer} & {inner} & {active} & "
                 f"{sci(kkt)} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(f"\nSaved {TABLES / 'nncg_regu.tex'}")

with open(TABLES / "nncg_regu_defs.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg_regu.py -- do not edit by hand.\n")
    fh.write(f"\\newcommand{{\\nncgReguN}}{{{N}}}\n")
    fh.write(f"\\newcommand{{\\nncgReguNoise}}{{{expo_only(NOISE_REL)}}}\n")
    fh.write(f"\\newcommand{{\\nncgReguKappa}}{{{expo_only(kappa0)}}}\n")
    fh.write(f"\\newcommand{{\\nncgReguAlpha}}{{{expo_only(ALPHA_STAR)}}}\n")
    fh.write(f"\\newcommand{{\\nncgReguKappaReg}}{{{expo_only(star[1])}}}\n")
    fh.write(f"\\newcommand{{\\nncgReguOuter}}{{{star[3]}}}\n")
    fh.write(f"\\newcommand{{\\nncgReguActive}}{{{star[5]}}}\n")
    fh.write(f"\\newcommand{{\\nncgReguSupp}}{{{N - star[5]}}}\n")
    fh.write(f"\\newcommand{{\\nncgReguKKT}}{{{star[6]:.0e}}}\n")
    fh.write(f"\\newcommand{{\\nncgReguErrLH}}{{{star[8]:.0e}}}\n")
    # phillips: recovery of a known non-negative signal + structured active set
    fh.write(f"\\newcommand{{\\nncgPhilN}}{{{PHIL_N}}}\n")
    fh.write(f"\\newcommand{{\\nncgPhilActive}}{{{phil_active}}}\n")
    fh.write(f"\\newcommand{{\\nncgPhilTrueZeros}}{{{phil_truezeros}}}\n")
    fh.write(f"\\newcommand{{\\nncgPhilRecov}}{{{phil_recov:.0e}}}\n")
    fh.write(f"\\newcommand{{\\nncgPhilErrLH}}{{{phil_err_lh:.0e}}}\n")
    fh.write(f"\\newcommand{{\\nncgPhilOuter}}{{{resp.outer}}}\n")
print(f"Saved {TABLES / 'nncg_regu_defs.tex'}")
print("\nDone.")
