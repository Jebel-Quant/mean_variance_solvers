"""Experiments for the companion paper.

    "Eigenvalue Cleaning and Direct Solvers for Long-Only Portfolio Optimisation"

Usage:
    uv run experiment_rmt.py      # from the experiment/ directory

Inputs:
    data/sp500_pct_returns.parquet   — S&P 500 daily pct returns
    Fetch with:  uv run fetch_sp500.py

Outputs (stdout):
    A: Preprocessing benchmark — dense eigendecomp vs randomised SVD
    B: Solver comparison — CVXPY | Cholesky | Woodbury on the RMT estimator
    C: k-sensitivity — portfolio change at k±1
    D: Scaling with preprocessing — runtime vs n

Outputs (files):
    graphs/rmt_frontier.pdf           — efficient frontier coloured by active-asset count
    graphs/rmt_scaling_full.pdf       — scaling: Cholesky vs Woodbury + preprocessing
    tables/rmt_preprocessing.tex
    tables/rmt_solver_comparison.tex
    tables/rmt_k_sensitivity.tex

Hardware: Apple M4 Pro, 14-core CPU, 48 GB RAM.
Software: Python 3.12, NumPy 2.4, SciPy 1.17, scikit-learn 1.x.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "fast-minimum-variance",
#     "pyarrow",
# ]
#
# [tool.uv.sources]
# fast-minimum-variance = { git = "https://github.com/Jebel-Quant/fast_minimum_variance" }
# ///

from __future__ import annotations

import time as _time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from util.runner import run_timed

from simulate import simulate_equity_returns
from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem
from fast_minimum_variance.shrinkage.util import (
    rmt_target_and_alpha,
)

HERE = Path(__file__).parent
GRAPHS = HERE / "graphs"
TABLES = HERE / "tables"
GRAPHS.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    }
)

SP500_DATA = HERE / "data/sp500_pct_returns.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rmt_target_at_k(X, k):
    """Return (target, lr_factors, k) using exactly k signal factors."""
    T, n = X.shape
    cov = (X.T @ X) / T
    bar_lam = float(np.trace(cov) / n)
    eigs, vecs = np.linalg.eigh(cov)  # ascending
    eigs_k = eigs[-k:]
    vecs_k = vecs[:, -k:]
    delta_k = eigs_k - bar_lam
    target = bar_lam * np.eye(n) + vecs_k @ np.diag(delta_k) @ vecs_k.T
    lr_factors = (bar_lam, vecs_k, delta_k)
    return target, lr_factors, k


def rsvd_eigenpairs(X, k, p=10):
    """Compute top-k eigenpairs of X^T X / T via randomised SVD."""
    T = X.shape[0]
    _, s, Vt = randomized_svd(X, n_components=k + p, random_state=0)
    eigs_k = (s[:k] ** 2) / T
    vecs_k = Vt[:k].T  # (n, k)
    bar_lam = float(np.linalg.norm(X, "fro") ** 2) / (X.shape[1] * T)
    return vecs_k, eigs_k, bar_lam


# ===========================================================================
# Section A: Preprocessing benchmark
# ===========================================================================

print("=" * 70)
print("A: Preprocessing benchmark  (dense eigendecomp vs randomised SVD)")
print("=" * 70)

df_sp = pd.read_parquet(SP500_DATA)
R_sp = df_sp.to_numpy()
R_sp = R_sp - R_sp.mean(axis=0)
T_sp, N_sp = R_sp.shape

_, _, k_sp, _ = rmt_target_and_alpha(R_sp)
print(f"S&P 500: n={N_sp}, T={T_sp}, k={k_sp} signal factors, p=10 oversampling")

# Dense path
cov_sp = (R_sp.T @ R_sp) / T_sp
bar_lam_sp = float(np.trace(cov_sp) / N_sp)

(eigs_dense, vecs_dense), t_dense = run_timed(lambda: np.linalg.eigh((R_sp.T @ R_sp) / T_sp))
U_dense = vecs_dense[:, -k_sp:]

# Randomised SVD path
(U_rsvd, eigs_rsvd, _), t_rsvd = run_timed(lambda: rsvd_eigenpairs(R_sp, k_sp, p=10))

# Subspace distance
proj_dense = U_dense @ U_dense.T
proj_rsvd = U_rsvd @ U_rsvd.T
subspace_err = float(np.linalg.norm(proj_dense - proj_rsvd, "fro"))

print(f"\n  {'Method':<35} {'Time (s)':>10} {'Storage':>10} {'Subspace err':>14}")
print(f"  {'-' * 75}")
print(f"  {'Dense eigendecomp (eigh)':<35} {t_dense:>10.4f} {'n² floats':>10} {'---':>14}")
print(f"  {'Randomised SVD (p=10)':<35} {t_rsvd:>10.4f} {'nk floats':>10} {subspace_err:>14.2e}")
print(f"\n  Speedup: {t_dense / t_rsvd:.1f}x")
print(f"  Storage ratio: {N_sp**2 / (N_sp * k_sp):.0f}x  ({N_sp**2} vs {N_sp * k_sp} floats)")

prep_table_lines = (
    f"Dense eigendecomposition & {t_dense:.4f} & $n^2$ & --- \\\\\n"
    f"Randomised SVD ($p=10$)  & {t_rsvd:.4f} & $nk$ & ${subspace_err:.1e}$ \\\\\n"
    f"\\midrule\n"
    f"Speedup & ${t_dense / t_rsvd:.1f}\\times$ & ${N_sp**2 // (N_sp * k_sp)}\\times$ & \\\\\n"
)
(TABLES / "rmt_preprocessing.tex").write_text(f"\\def\\dataPreprocessing{{%\n{prep_table_lines}}}\n")
print("  → wrote tables/rmt_preprocessing.tex")


# ===========================================================================
# Section B: Solver comparison
#            CVXPY | KKT-Cholesky | Woodbury on the same RMT estimator
#            n=500, T=1250, synthetic, 50-point efficient frontier
# ===========================================================================

print()
print("=" * 70)
print("B: Solver comparison  (CVXPY | Cholesky | Woodbury, all on RMT, n=500, T=1250)")
print("=" * 70)

n_ef, T_ef = 500, 1250
R_ef = simulate_equity_returns(n_ef, T_ef, rng=42)
rng_ef = np.random.default_rng(42)
betas_ef = rng_ef.uniform(0.4, 0.8, n_ef)
mu_ef = betas_ef * (0.10 / 250)

tgt_rmt_ef, lr_rmt_ef, k_rmt_ef, alpha_rmt_ef = rmt_target_and_alpha(R_ef)
Sigma_rmt_ef = tgt_rmt_ef

rhos_ef = np.linspace(0, 2, 50)
N_PTS = len(rhos_ef)

print(f"  RMT: alpha={alpha_rmt_ef:.4f}, k={k_rmt_ef} signal factors")


def _cold_sweep(solve_fn_name, prob_fn, repeats=3):
    best_total = float("inf")
    best_times = None
    for _ in range(repeats):
        times = []
        for rho in rhos_ef:
            p = prob_fn(rho)
            t0 = _time.perf_counter()
            getattr(p, solve_fn_name)()
            times.append(_time.perf_counter() - t0)
        if sum(times) < best_total:
            best_total = sum(times)
            best_times = times
    return best_times


def _warm_sweep_kkt(prob_fn, repeats=3):
    best_total = float("inf")
    best_times = None
    for _ in range(repeats):
        times = []
        warm = None
        for rho in rhos_ef:
            p = prob_fn(rho)
            t0 = _time.perf_counter()
            _, _, warm = p.solve_kkt_warm(warm_start=warm)
            times.append(_time.perf_counter() - t0)
        if sum(times) < best_total:
            best_total = sum(times)
            best_times = times
    return best_times


print("  Running sweeps...")

# CVXPY reference (cold only — warm-starting not available)
t_cvxpy_cold = []
for rho in rhos_ef:
    p = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho, mu=mu_ef)
    t0 = _time.perf_counter()
    p.solve_cvxpy()
    t_cvxpy_cold.append(_time.perf_counter() - t0)
cvxpy_c = sum(t_cvxpy_cold)

# KKT-Cholesky: solve_kkt WITHOUT target_lr (assembles n_a x n_a, then Cholesky)
t_kkt_cold = _cold_sweep(
    "solve_kkt", lambda rho: MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho, mu=mu_ef)
)
t_kkt_warm = _warm_sweep_kkt(lambda rho: MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho, mu=mu_ef))

# Woodbury: solve_kkt WITH target_lr (never assembles n_a x n_a matrix)
t_wb_cold = _cold_sweep(
    "solve_kkt",
    lambda rho: MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, target_lr=lr_rmt_ef, rho=rho, mu=mu_ef),
)
t_wb_warm = _warm_sweep_kkt(
    lambda rho: MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, target_lr=lr_rmt_ef, rho=rho, mu=mu_ef)
)

print(f"\n  {'Solver':<42} {'Cold (s)':>9} {'ms/pt':>7} {'Warm (s)':>9} {'ms/pt':>7} {'WB speedup':>11}")
print(f"  {'-' * 95}")

kkt_c, kkt_w = sum(t_kkt_cold), sum(t_kkt_warm)
wb_c, wb_w = sum(t_wb_cold), sum(t_wb_warm)
print(f"  {'CVXPY (Clarabel)':<42} {cvxpy_c:>9.3f} {cvxpy_c / N_PTS * 1000:>7.1f} {'---':>9} {'---':>7} {'---':>11}")
print(
    f"  {'KKT-Cholesky':<42} {kkt_c:>9.3f} {kkt_c / N_PTS * 1000:>7.1f} "
    f"{kkt_w:>9.3f} {kkt_w / N_PTS * 1000:>7.1f} {'---':>11}"
)
print(
    f"  {'Woodbury':<42} {wb_c:>9.3f} {wb_c / N_PTS * 1000:>7.1f} "
    f"{wb_w:>9.3f} {wb_w / N_PTS * 1000:>7.1f} {kkt_w / wb_w:>10.1f}x"
)

print(f"\n  Woodbury vs KKT-Cholesky (cold): {kkt_c / wb_c:.1f}x")
print(f"  Woodbury vs KKT-Cholesky (warm): {kkt_w / wb_w:.1f}x")
print(f"  Woodbury vs CVXPY (cold):         {cvxpy_c / wb_c:.0f}x")

# Woodbury warm sweep: capture frontier points and active-set sizes
ef_vols_rmt, ef_rets_rmt, active_sizes = [], [], []
warm_rmt = None
for rho in rhos_ef:
    p = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, target_lr=lr_rmt_ef, rho=rho, mu=mu_ef)
    w, _, warm_rmt = p.solve_kkt_warm(warm_start=warm_rmt)
    ef_vols_rmt.append(float(np.sqrt(w @ Sigma_rmt_ef @ w)) * np.sqrt(250) * 100)
    ef_rets_rmt.append(float(w @ mu_ef) * 250 * 100)
    active_sizes.append(int((w > 1e-6).sum()))

# KKT-Cholesky warm sweep: frontier overlay (should match Woodbury numerically)
ef_vols_kkt, ef_rets_kkt = [], []
warm_kkt = None
for rho in rhos_ef:
    p = MinVarProblem(R_ef, alpha=alpha_rmt_ef, target=tgt_rmt_ef, rho=rho, mu=mu_ef)
    w, _, warm_kkt = p.solve_kkt_warm(warm_start=warm_kkt)
    ef_vols_kkt.append(float(np.sqrt(w @ Sigma_rmt_ef @ w)) * np.sqrt(250) * 100)
    ef_rets_kkt.append(float(w @ mu_ef) * 250 * 100)

print(
    f"\n  Active-set sizes (Woodbury warm): mean={np.mean(active_sizes):.1f}, "
    f"min={min(active_sizes)}, max={max(active_sizes)}"
)
max_diff = float(np.max(np.abs(np.array(ef_vols_rmt) - np.array(ef_vols_kkt))))
print(f"  Max vol difference (WB vs KKT-Chol): {max_diff:.2e}% (numerical precision check)")

# S&P 500 single min-var solve (cold and warm) —
# uses the real return matrix already loaded in Section A
tgt_sp_b, lr_sp_b, k_sp_b, alpha_sp_b = rmt_target_and_alpha(R_sp)


def _sp_cold(solve_fn_name, prob, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = _time.perf_counter()
        getattr(prob, solve_fn_name)()
        best = min(best, _time.perf_counter() - t0)
    return best


p_cvxpy_sp = MinVarProblem(R_sp, alpha=alpha_sp_b, target=tgt_sp_b)
p_kkt_sp = MinVarProblem(R_sp, alpha=alpha_sp_b, target=tgt_sp_b)
p_wb_sp = MinVarProblem(R_sp, alpha=alpha_sp_b, target=tgt_sp_b, target_lr=lr_sp_b)

cvxpy_sp_c = _sp_cold("solve_cvxpy", p_cvxpy_sp)
kkt_sp_c = _sp_cold("solve_kkt", p_kkt_sp)
wb_sp_c = _sp_cold("solve_kkt", p_wb_sp)

# Warm: one immediate re-solve after a cold run (active-set already converged)
_, _, warm_kkt_sp = p_kkt_sp.solve_kkt_warm(warm_start=None)
t0 = _time.perf_counter()
p_kkt_sp.solve_kkt_warm(warm_start=warm_kkt_sp)
kkt_sp_w = _time.perf_counter() - t0

_, _, warm_wb_sp = p_wb_sp.solve_kkt_warm(warm_start=None)
t0 = _time.perf_counter()
p_wb_sp.solve_kkt_warm(warm_start=warm_wb_sp)
wb_sp_w = _time.perf_counter() - t0

print(f"\n  S&P 500 single min-var solve (n={N_sp}, T={T_sp}, k={k_sp_b}):")
print(f"    CVXPY cold:          {cvxpy_sp_c * 1000:>7.1f} ms")
print(f"    KKT-Cholesky cold:   {kkt_sp_c * 1000:>7.1f} ms   warm: {kkt_sp_w * 1000:.1f} ms")
print(f"    Woodbury cold:       {wb_sp_c * 1000:>7.1f} ms   warm: {wb_sp_w * 1000:.1f} ms")
print(f"    Woodbury vs KKT-Chol (cold): {kkt_sp_c / wb_sp_c:.1f}x")


# Write combined two-panel solver comparison table
def _fmt(t) -> str:
    """Format seconds as e.g. 88.1 or 0.042."""
    if t >= 10:
        return f"{t:.1f}"
    if t >= 1:
        return f"{t:.3f}"
    if t >= 0.1:
        return f"{t:.4f}"
    return f"{t:.4f}"


def _ms(t) -> str:
    return f"{t * 1000:.1f}"


synth_rows = (
    f"{'CVXPY (Clarabel)':<38} & {_fmt(cvxpy_c):>7} & {_ms(cvxpy_c / N_PTS):>6}"
    f" & \\multicolumn{{2}}{{c}}{{--}} \\\\\n"
    f"{'KKT-Cholesky ($k=' + str(k_rmt_ef) + '$)':<38} & {_fmt(kkt_c):>7} & {_ms(kkt_c / N_PTS):>6}"
    f" & {_fmt(kkt_w):>7} & {_ms(kkt_w / N_PTS):>6} \\\\\n"
    f"{'Woodbury ($k=' + str(k_rmt_ef) + '$)':<38} & {_fmt(wb_c):>7} & {_ms(wb_c / N_PTS):>6}"
    f" & {_fmt(wb_w):>7} & {_ms(wb_w / N_PTS):>6} \\\\\n"
)
sp_rows = (
    f"{'CVXPY (Clarabel)':<38} & {_fmt(cvxpy_sp_c):>7} & {_ms(cvxpy_sp_c):>6}"
    f" & \\multicolumn{{2}}{{c}}{{--}} \\\\\n"
    f"{'KKT-Cholesky ($k=' + str(k_sp_b) + '$)':<38} & {_fmt(kkt_sp_c):>7} & {_ms(kkt_sp_c):>6}"
    f" & {_fmt(kkt_sp_w):>7} & {_ms(kkt_sp_w):>6} \\\\\n"
    f"{'Woodbury ($k=' + str(k_sp_b) + '$)':<38} & {_fmt(wb_sp_c):>7} & {_ms(wb_sp_c):>6}"
    f" & {_fmt(wb_sp_w):>7} & {_ms(wb_sp_w):>6} \\\\\n"
)

panel_a = (
    "\\multicolumn{5}{l}{\\textit{Synthetic, $n=500$, $T=1250$, $k=22$,"
    " 50-point sweep (total / ms per point)}} \\\\\n"
    "\\addlinespace[2pt]\n"
)
panel_b = (
    "\\multicolumn{5}{l}{\\textit{S\\&P~500, $n=494$, $T=1213$, $k=23$,"
    " single min-var solve (seconds / ms)}} \\\\\n"
    "\\addlinespace[2pt]\n"
)

(TABLES / "rmt_solver_comparison.tex").write_text(
    "\\def\\dataRmtSolverComp{%\n" + panel_a + synth_rows + "\\midrule\n" + panel_b + sp_rows + "}\n"
)
print("  → wrote tables/rmt_solver_comparison.tex")

# Efficient frontier figure: Woodbury (coloured by active assets) + KKT-Cholesky (dashed overlay)
fig_ef, ax_ef = plt.subplots(figsize=(4.8, 3.4))
sc = ax_ef.scatter(
    ef_vols_rmt, ef_rets_rmt, c=active_sizes, cmap="plasma_r", s=18, zorder=3, label=rf"Woodbury (RMT, $k={k_rmt_ef}$)"
)
ax_ef.plot(ef_vols_rmt, ef_rets_rmt, color="gray", linewidth=0.7, zorder=2)
ax_ef.plot(
    ef_vols_kkt,
    ef_rets_kkt,
    color="steelblue",
    linewidth=1.2,
    linestyle="--",
    zorder=4,
    label=rf"KKT-Cholesky (RMT, $k={k_rmt_ef}$)",
)
ax_ef.scatter(
    [ef_vols_rmt[0]], [ef_rets_rmt[0]], marker="*", s=110, color="#ff7f0e", zorder=5, label="Min-var (Woodbury)"
)
cbar_ef = fig_ef.colorbar(sc, ax=ax_ef, pad=0.02)
cbar_ef.set_label("Active assets")
ax_ef.set_xlabel("Annualised volatility (\\%)")
ax_ef.set_ylabel("Annualised expected return (\\%)")
ax_ef.set_title(rf"Efficient frontier ($n={n_ef}$, $T={T_ef}$, RMT $k={k_rmt_ef}$)")
ax_ef.legend(framealpha=0.9, loc="lower right", fontsize=7)
ax_ef.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
fig_ef.tight_layout(pad=1.0)
fig_ef.savefig(GRAPHS / "rmt_frontier.pdf", bbox_inches="tight")
fig_ef.savefig(GRAPHS / "rmt_frontier.png", bbox_inches="tight", dpi=150)
print("  → saved graphs/rmt_frontier.pdf")
plt.close(fig_ef)


# ===========================================================================
# Section C: k-sensitivity
# ===========================================================================

print()
print("=" * 70)
print("C: k-sensitivity on S&P 500  (portfolios at k-1, k, k+1)")
print("=" * 70)

k_ref = k_sp
k_vals = [k_ref - 1, k_ref, k_ref + 1]
w_at_k = {}

for k_try in k_vals:
    tgt_k, lr_k, _ = rmt_target_at_k(R_sp, k_try)
    prob_k = MinVarProblem(R_sp, alpha=1.0, target=tgt_k, target_lr=lr_k)
    w_k, _ = prob_k.solve_kkt()
    w_at_k[k_try] = w_k

w_ref = w_at_k[k_ref]
cov_sp_full = (R_sp.T @ R_sp) / T_sp

print(
    f"\n  k={k_ref} reference: {int((w_ref > 1e-6).sum())} active assets, "
    f"vol={float(np.sqrt(w_ref @ cov_sp_full @ w_ref)) * np.sqrt(250) * 100:.3f}% ann."
)
print(f"\n  {'k':>4} {'Active':>8} {'Ann.Vol(%)':>12} {'||w-w*||_inf':>14} {'||w-w*||_2':>12}")
print(f"  {'-' * 55}")
for k_try in k_vals:
    w = w_at_k[k_try]
    vol = float(np.sqrt(w @ cov_sp_full @ w)) * np.sqrt(250) * 100
    active = int((w > 1e-6).sum())
    diff_inf = float(np.abs(w - w_ref).max())
    diff_2 = float(np.linalg.norm(w - w_ref))
    marker = " ← reference" if k_try == k_ref else ""
    print(f"  {k_try:>4} {active:>8} {vol:>12.3f} {diff_inf:>14.4f} {diff_2:>12.4f}{marker}")

k_sens_lines = ""
for k_try in k_vals:
    w = w_at_k[k_try]
    vol = float(np.sqrt(w @ cov_sp_full @ w)) * np.sqrt(250) * 100
    active = int((w > 1e-6).sum())
    diff_inf = float(np.abs(w - w_ref).max())
    diff_2 = float(np.linalg.norm(w - w_ref))
    k_sens_lines += f"{k_try} & {active} & {vol:.2f} & {diff_inf:.4f} & {diff_2:.4f} \\\\\n"

(TABLES / "rmt_k_sensitivity.tex").write_text(f"\\def\\dataKsensitivity{{%\n{k_sens_lines}}}\n")
print("  → wrote tables/rmt_k_sensitivity.tex")


# ===========================================================================
# Section D: Scaling — Cholesky vs Woodbury + preprocessing
# ===========================================================================

print()
print("=" * 70)
print("D: Scaling  (Cholesky vs Woodbury + preprocessing, T=1250 fixed)")
print("=" * 70)

T_FIXED = 1250
ns = [300, 500, 750, 1000, 1500, 2000, 3000]
t_kkt_scale = []  # KKT-Cholesky (no target_lr)
t_wb_solve = []  # Woodbury solve only
t_wb_dense_prep = []  # dense eigendecomp preprocessing
t_wb_rsvd_prep = []  # randomised SVD preprocessing
t_wb_total_dense = []
t_wb_total_rsvd = []
k_detected = []

print(
    f"\n  {'n':>5} {'k':>4} {'Cholesky(s)':>13} {'WB-solve(s)':>12} "
    f"{'Dense-prep(s)':>14} {'rSVD-prep(s)':>13} {'WB+dense(s)':>12} {'WB+rSVD(s)':>11}"
)
print(f"  {'-' * 95}")

for n in ns:
    R_s = simulate_equity_returns(n, T_FIXED, rng=n)

    tgt_s_rmt, lr_s_rmt, k_s, _ = rmt_target_and_alpha(R_s)

    # KKT-Cholesky (no target_lr): assembles n_a x n_a matrix, then Cholesky
    prob_kkt = MinVarProblem(R_s, alpha=1.0, target=tgt_s_rmt)
    (_, _), t_kkt = run_timed(lambda p=prob_kkt: p.solve_kkt())

    # Dense preprocessing
    _, t_dense_s = run_timed(lambda R=R_s: np.linalg.eigh((R.T @ R) / T_FIXED))

    # Woodbury solve only (using dense eigenpairs, preprocessing already done)
    prob_wb = MinVarProblem(R_s, alpha=1.0, target=tgt_s_rmt, target_lr=lr_s_rmt)
    (_, _), t_wb = run_timed(lambda p=prob_wb: p.solve_kkt())

    # Randomised SVD preprocessing
    _, t_rsvd_s = run_timed(lambda R=R_s, k=k_s: rsvd_eigenpairs(R, k, p=10))

    t_kkt_scale.append(t_kkt)
    t_wb_solve.append(t_wb)
    t_wb_dense_prep.append(t_dense_s)
    t_wb_rsvd_prep.append(t_rsvd_s)
    t_wb_total_dense.append(t_dense_s + t_wb)
    t_wb_total_rsvd.append(t_rsvd_s + t_wb)
    k_detected.append(k_s)

    print(
        f"  {n:>5} {k_s:>4} {t_kkt:>13.4f} {t_wb:>12.4f} "
        f"{t_dense_s:>14.4f} {t_rsvd_s:>13.4f} "
        f"{t_dense_s + t_wb:>12.4f} {t_rsvd_s + t_wb:>11.4f}"
    )

# Scaling figure
fig_sc, ax_sc = plt.subplots(figsize=(4.8, 3.4))
n_arr = np.array(ns, dtype=float)

ax_sc.plot(ns, t_kkt_scale, marker="o", markersize=4, color="#ff7f0e", label=r"KKT-Cholesky (assemble + Chol)")
ax_sc.plot(ns, t_wb_solve, marker="s", markersize=4, color="#2ca02c", label=r"Woodbury solve only")
ax_sc.plot(ns, t_wb_total_rsvd, marker="^", markersize=4, color="#1f77b4", label=r"Woodbury + rSVD prep")
ax_sc.plot(
    ns, t_wb_total_dense, marker="v", markersize=4, color="#d62728", linestyle="--", label=r"Woodbury + dense prep"
)

idx_500 = ns.index(500) if 500 in ns else 0
t_ref = t_kkt_scale[idx_500]
ax_sc.plot(n_arr, t_ref * (n_arr / 500.0), color="gray", linestyle=":", linewidth=0.9, label=r"$O(n)$")
ax_sc.plot(
    n_arr, t_ref * (n_arr / 500.0) ** 2, color="gray", linestyle=(0, (3, 1, 1, 1)), linewidth=0.9, label=r"$O(n^2)$"
)

ax_sc.set_xscale("log")
ax_sc.set_yscale("log")
ax_sc.set_xlabel("Number of assets $n$")
ax_sc.set_ylabel("Wall-clock time (s)")
ax_sc.set_title(r"Scaling with $n$ ($T=1250$): Cholesky vs Woodbury")
ax_sc.legend(framealpha=0.9, fontsize=7, loc="upper left")
ax_sc.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
fig_sc.tight_layout(pad=1.0)
fig_sc.savefig(GRAPHS / "rmt_scaling_full.pdf", bbox_inches="tight")
fig_sc.savefig(GRAPHS / "rmt_scaling_full.png", bbox_inches="tight", dpi=150)
print("\n  → saved graphs/rmt_scaling_full.pdf")
plt.close(fig_sc)


# ===========================================================================
# Summary
# ===========================================================================

print()
print("=" * 70)
print("Summary of generated outputs")
print("=" * 70)
print("  Tables:")
print("    tables/rmt_preprocessing.tex      (Section A)")
print("    tables/rmt_solver_comparison.tex  (Section B)")
print("    tables/rmt_k_sensitivity.tex      (Section C)")
print("  Figures:")
print("    graphs/rmt_frontier.pdf           (Section B)")
print("    graphs/rmt_scaling_full.pdf       (Section D)")
