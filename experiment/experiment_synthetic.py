"""Scaling, iteration, and frontier experiments using simulated data.

    "Shrinkage as Preconditioning: Matrix-Free Methods for
     Long-Only Portfolio Optimization"

Usage:
    uv run experiment_synthetic.py   # from the experiment/ directory

Outputs (stdout):
    Scaling table — runtime vs n for KKT / CG / proximal (T=1250 fixed).
    Iterations table — CG iterations vs alpha (n=500, T=250, rank-deficient).
    Frontier table — warm- vs cold-start sweep timings (n=500, T=1250).

Outputs (files):
    graphs/minvar_scaling.pdf   — Figure 1: runtime vs n (log-log)
    graphs/minvar_iters.pdf     — Figure 2: CG iterations vs alpha
    graphs/minvar_frontier.pdf  — Figure 3: efficient frontier coloured by active assets

Hardware used in the paper: Apple M4 Pro, 14-core CPU, 48 GB RAM.
Software: Python 3.12, NumPy 2.4, SciPy 1.17, CVXPY 1.8.2, Clarabel 0.11.1.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scikit-learn",
#     "fast-minimum-variance",
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
from util.runner import run_timed, write_frontier_def

from simulate import simulate_equity_returns
from fast_minimum_variance.minvar_problem import _MinVarProblem as MinVarProblem
from fast_minimum_variance.shrinkage.util import (
    lw_alpha_and_target,
    lw_alpha_and_target_hard,
)

HERE = Path(__file__).parent
GRAPHS = HERE / "graphs"
TABLES = HERE / "tables"
GRAPHS.mkdir(exist_ok=True)

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

# ---------------------------------------------------------------------------
# Panel A: runtime vs n  (LW shrinkage, T=1250 fixed)
# ---------------------------------------------------------------------------

print("=" * 70)
print("Runtime vs n  (T=1250 fixed, LW shrinkage, long-only minimum variance)")
print("=" * 70)

T_FIXED = 1250
ns = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
times = {k: [] for k in ("kkt", "cg", "proximal", "fista")}

print(
    f"{'n':>6}  {'k_active':>8}  {'kkt(s)':>10}  {'kkt_out':>8}"
    f"  {'cg(s)':>10}  {'cg_out':>7}  {'cg_in':>7}"
    f"  {'prox(s)':>10}  {'prox_in':>8}"
    f"  {'fista(s)':>10}  {'fista_in':>9}"
)
print("-" * 110)

for n in ns:
    R = simulate_equity_returns(n, T_FIXED, rng=n)
    alpha, tgt = lw_alpha_and_target_hard(R, alpha=0.5)
    prob = MinVarProblem(R, alpha=alpha, target=tgt)

    (w_kkt, kkt_outer), t_kkt = run_timed(lambda p=prob: p.solve_kkt())
    k_active = int((w_kkt > 1e-6).sum())
    (_, cg_outer, cg_inner), t_cg = run_timed(lambda p=prob: p.solve_cg())
    (_, prox_inner), t_prox = run_timed(lambda p=prob: p.solve_proximal())
    (_, fista_inner), t_fista = run_timed(lambda p=prob: p.solve_fista())

    times["kkt"].append(t_kkt)
    times["cg"].append(t_cg)
    times["proximal"].append(t_prox)
    times["fista"].append(t_fista)
    print(
        f"{n:>6}  {k_active:>8}  {t_kkt:>10.4f}  {kkt_outer:>8}"
        f"  {t_cg:>10.4f}  {cg_outer:>7}  {cg_inner:>7}"
        f"  {t_prox:>10.4f}  {prox_inner:>8}"
        f"  {t_fista:>10.4f}  {fista_inner:>9}"
    )

# Empirical scaling exponent for CG (log-log least squares over n >= 300).
_ns = np.array(ns, dtype=float)
_fit_mask = _ns >= 300
_log_n = np.log(_ns[_fit_mask])
_log_t = np.log(np.array(times["cg"])[_fit_mask])
_slope, _intercept = np.polyfit(_log_n, _log_t, 1)
_pred = _slope * _log_n + _intercept
_ss_res = float(np.sum((_log_t - _pred) ** 2))
_ss_tot = float(np.sum((_log_t - _log_t.mean()) ** 2))
_r2 = 1.0 - _ss_res / _ss_tot if _ss_tot > 0 else float("nan")
print(
    f"\nCG empirical scaling: slope={_slope:.3f}, R^2={_r2:.4f} "
    f"over {_fit_mask.sum()} points (n in [300, {int(_ns.max())}], single seed rng=n)"
)

# ---------------------------------------------------------------------------
# Panel B: CG iterations vs alpha  (n=500, T=250, rank-deficient)
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("CG iterations vs alpha  (n=500, T=250, rank-deficient)")
print("=" * 70)

n_iter, T_iter = 500, 250
R_iter = simulate_equity_returns(n_iter, T_iter, rng=1)
_, tgt_iter = lw_alpha_and_target(R_iter)
alphas = np.linspace(0.01, 0.99, 40)
cg_iters = []

print(f"{'alpha':>8}  {'outer':>7}  {'inner':>8}")
for a in alphas:
    _, outer, inner = MinVarProblem(R_iter, alpha=a, target=tgt_iter).solve_cg()
    cg_iters.append(inner)
    print(f"{a:>8.3f}  {outer:>7}  {inner:>8}")

# ---------------------------------------------------------------------------
# Figures 1 and 2
# ---------------------------------------------------------------------------

COLORS = {"cg": "#ff7f0e", "proximal": "#9467bd", "kkt": "#1f77b4"}
LABELS = {
    "cg": "CG (LW, $\\alpha=0.5$)",
    "proximal": "Proximal gradient",
    "kkt": "KKT (Cholesky)",
}

# Figure 1: runtime vs n
fig1, ax1 = plt.subplots(figsize=(4.5, 3.2))
for key in ("kkt", "cg", "proximal"):
    ax1.plot(ns, times[key], marker="o", markersize=4, label=LABELS[key], color=COLORS[key])
n_arr = np.array(ns, dtype=float)
anchor_idx = ns.index(500)
t_anchor = times["cg"][anchor_idx]
ax1.plot(n_arr, t_anchor * (n_arr / 500.0), color="gray", linestyle="--", linewidth=0.9, label=r"$O(n)$")
ax1.plot(n_arr, t_anchor * (n_arr / 500.0) ** 2, color="gray", linestyle=":", linewidth=0.9, label=r"$O(n^2)$")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Number of assets $n$")
ax1.set_ylabel("Wall-clock time (s)")
ax1.set_title(r"Runtime vs. $n$  ($T=1250$ fixed)")
ax1.legend(framealpha=0.9)
ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
fig1.tight_layout(pad=1.0)
fig1.savefig(GRAPHS / "minvar_scaling.pdf", bbox_inches="tight")
fig1.savefig(GRAPHS / "minvar_scaling.png", bbox_inches="tight", dpi=150)
print(f"\nSaved {GRAPHS / 'minvar_scaling.pdf'}")

# Figure 2: CG iterations vs alpha
fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
ax2.plot(alphas, cg_iters, marker="o", markersize=4, color=COLORS["cg"], label=LABELS["cg"])
ax2.set_xlabel(r"Shrinkage intensity $\alpha$  ($\kappa$ decreases $\rightarrow$)")
ax2.set_ylabel("CG iterations to convergence")
ax2.set_title(r"CG iterations vs. $\alpha$  ($n=500,\,T=250$)")
ax2.legend(framealpha=0.9)
ax2.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
fig2.tight_layout(pad=1.0)
fig2.savefig(GRAPHS / "minvar_iters.pdf", bbox_inches="tight")
fig2.savefig(GRAPHS / "minvar_iters.png", bbox_inches="tight", dpi=150)
print(f"Saved {GRAPHS / 'minvar_iters.pdf'}")

# ---------------------------------------------------------------------------
# Section 9: Efficient frontier  (n=500, T=1250)
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("Efficient frontier  (n=500, T=1250, multi-solver)")
print("=" * 70)

n_ef, T_ef = 500, 1250
R_ef = simulate_equity_returns(n_ef, T_ef, rng=42)
rng_ef = np.random.default_rng(42)
betas_ef = rng_ef.uniform(0.4, 0.8, n_ef)
mu_ef = betas_ef * (0.10 / 250)

_, tgt_ef = lw_alpha_and_target(R_ef)
alpha_ef = 0.5
Sigma_ef = (1 - alpha_ef) * (R_ef.T @ R_ef) / T_ef + alpha_ef * tgt_ef
print(f"Frontier alpha (LW) = {alpha_ef}")

rhos_ef = np.linspace(0, 2, 50)


def _sweep_cold(solve_fn, repeats=3, ef_alpha=None, ef_target=None):
    """Return best-of-repeats list of per-point cold-start times."""
    _alpha = alpha_ef if ef_alpha is None else ef_alpha
    _target = tgt_ef if ef_target is None else ef_target
    runs = []
    for _ in range(repeats):
        sweep_times = []
        for rho in rhos_ef:
            prob = MinVarProblem(R_ef, alpha=_alpha, target=_target, rho=rho, mu=mu_ef)
            t0 = _time.perf_counter()
            solve_fn(prob)
            sweep_times.append(_time.perf_counter() - t0)
        runs.append(sweep_times)
    return runs[int(np.argmin([sum(r) for r in runs]))]


print("Running cold sweeps...")
ef_times_cvxpy = _sweep_cold(lambda p: p.solve_cvxpy(), repeats=1)
ef_times_osqp = _sweep_cold(lambda p: p.solve_osqp(), repeats=3)
ef_times_proximal = _sweep_cold(lambda p: p.solve_proximal(), repeats=3)
ef_times_cg_cold = _sweep_cold(lambda p: p.solve_cg(), repeats=3)
print("Running CG warm-start sweep...")
ef_warm_runs = []
ef_vols, ef_rets, ef_active = [], [], []
for rep in range(3):
    sweep_times = []
    warm = None
    vols_r, rets_r, act_r = [], [], []
    for rho in rhos_ef:
        prob = MinVarProblem(R_ef, alpha=alpha_ef, target=tgt_ef, rho=rho, mu=mu_ef)
        t0 = _time.perf_counter()
        w, _, _, warm = prob.solve_cg_warm(warm_start=warm)
        sweep_times.append(_time.perf_counter() - t0)
        vols_r.append(float(np.sqrt(w @ Sigma_ef @ w)) * np.sqrt(250) * 100)
        rets_r.append(float(w @ mu_ef) * 250 * 100)
        act_r.append(int((w > 1e-6).sum()))
    ef_warm_runs.append(sweep_times)
    if rep == 0:
        ef_vols, ef_rets, ef_active = vols_r, rets_r, act_r
ef_times_cg_warm = ef_warm_runs[int(np.argmin([sum(r) for r in ef_warm_runs]))]

N_PTS = len(rhos_ef)
ref = sum(ef_times_cvxpy)


def _row(label, cold_times, warm_times=None) -> None:
    """Print one row of the frontier timing table."""
    total = sum(cold_times)
    per_ms = total / N_PTS * 1000
    spd = ref / total
    if warm_times is not None:
        w_total = sum(warm_times)
        w_per = w_total / N_PTS * 1000
        w_spd = ref / w_total
        print(
            f"  {label:<28}  cold: {total:6.3f}s ({per_ms:5.1f}ms/pt, {spd:5.0f}x)  "
            f"warm: {w_total:6.3f}s ({w_per:5.1f}ms/pt, {w_spd:5.0f}x)"
        )
    else:
        print(f"  {label:<28}  cold: {total:6.3f}s ({per_ms:5.1f}ms/pt, {spd:5.0f}x)  warm: --")


print(f"\n{'Solver':<30}  {'Cold total':>12}  {'Warm total':>12}")
print("-" * 70)
_row("cvxpy (Clarabel)", ef_times_cvxpy)
_row("OSQP (direct API)", ef_times_osqp)
_row("Proximal gradient", ef_times_proximal)
_row("CG (alpha=0.5, LW)", ef_times_cg_cold, ef_times_cg_warm)

total_ef_cold = sum(ef_times_cg_cold)
total_ef_warm = sum(ef_times_cg_warm)
print(f"\nCG (LW)  warm vs cold speedup: {total_ef_cold / total_ef_warm:.1f}x")
print(f"CG (LW)  warm vs CVXPY speedup: {ref / total_ef_warm:.0f}x")

print(f"\n{'rho':>6}  {'vol%ann':>9}  {'ret%ann':>9}  {'active':>7}  {'cg_cold(ms)':>12}  {'cg_warm(ms)':>12}")
print("-" * 65)
for rho, vol, ret, act, tc, tw in zip(
    rhos_ef, ef_vols, ef_rets, ef_active, ef_times_cg_cold, ef_times_cg_warm, strict=False
):
    print(f"{rho:>6.2f}  {vol:>9.3f}  {ret:>9.3f}  {act:>7}  {tc * 1000:>12.1f}  {tw * 1000:>12.1f}")

# Figure 3: efficient frontier — LW coloured by active assets
fig3, ax3 = plt.subplots(figsize=(4.5, 3.2))
sc = ax3.scatter(ef_vols, ef_rets, c=ef_active, cmap="plasma_r", s=20, zorder=3, label=r"LW ($\alpha=0.5$)")
ax3.plot(ef_vols, ef_rets, color="gray", linewidth=0.8, zorder=2)
ax3.scatter([ef_vols[0]], [ef_rets[0]], marker="*", s=120, color="#ff7f0e", zorder=4, label="Min-var (LW)")
cbar = fig3.colorbar(sc, ax=ax3, pad=0.02)
cbar.set_label("Active assets")
ax3.set_xlabel("Annualised volatility (\\%)")
ax3.set_ylabel("Annualised expected return (\\%)")
ax3.set_title(f"Efficient frontier  ($n={n_ef}$, $T={T_ef}$)")
ax3.legend(framealpha=0.9, loc="lower right", fontsize=7)
ax3.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
fig3.tight_layout(pad=1.0)
fig3.savefig(GRAPHS / "minvar_frontier.pdf", bbox_inches="tight")
fig3.savefig(GRAPHS / "minvar_frontier.png", bbox_inches="tight", dpi=150)
print(f"Saved {GRAPHS / 'minvar_frontier.pdf'}")

write_frontier_def(
    TABLES / "frontier_def.tex",
    "dataFrontier",
    [
        {"label": "cvxpy (Clarabel)", "cold": sum(ef_times_cvxpy), "warm": None},
        {"label": "OSQP (direct API)", "cold": sum(ef_times_osqp), "warm": None},
        {"label": "Proximal gradient", "cold": sum(ef_times_proximal), "warm": None},
        {"label": r"CG (LW, $\alpha=0.5$)", "cold": sum(ef_times_cg_cold), "warm": sum(ef_times_cg_warm)},
    ],
    n_pts=N_PTS,
)
print("  → wrote experiment/tables/frontier_def.tex")
