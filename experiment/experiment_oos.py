r"""Out-of-sample shrinkage study for the paper (referee response M3/R1).

    "Matrix-Free Methods for Long-Only Portfolio Optimization"

Goal:
    Demonstrate empirically that the *conditioning-optimal* shrinkage regime
    (heavy alpha ~ 0.3-0.5, which compresses kappa and slashes CG iterations)
    is not statistically harmful out-of-sample -- i.e. the two objectives,
    statistical accuracy and numerical conditioning, are reconcilable on real
    data.  We run a rolling-window long-only minimum-variance backtest on the
    S&P 500 universe and report, as a function of the shrinkage intensity alpha:

      * realized out-of-sample annualised volatility (the statistical objective),
      * annualised return / Sharpe,
      * average one-way turnover per rebalance,
      * mean CG iterations to convergence (the conditioning objective).

    R1 robustness additions:
      * moving-block bootstrap confidence intervals on OOS volatility,
      * a paired bootstrap test of vol(alpha=0.5) - vol(alpha=0.017),
      * a second estimation window (L=252) to check the optimum is stable.

Usage:
    uv run experiment_oos.py          # from the experiment/ directory

Inputs:
    data/sp500_pct_returns.parquet    — S&P 500 daily pct returns

Outputs (stdout):
    Per-alpha table of OOS vol / Sharpe / turnover / CG iterations,
    bootstrap CIs, paired test, and L=252 robustness summary.

Outputs (files):
    tables/oos_defs.tex   — \\def\\dataOos macro for the paper table
    graphs/minvar_oos.pdf — OOS volatility and CG iterations vs alpha

Hardware used in the paper: Apple M4 Pro, 14-core CPU, 48 GB RAM.
Software: Python 3.12, NumPy 2.4, SciPy 1.17.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "scikit-learn",
#     "nncg==0.2.2",
#     "cvx-linalg",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minvar import MinVarProblem, lw_alpha_and_target

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOLD = 21  # rebalance / holding period (~1 trading month)
ANN = 252  # trading days per year
ALPHAS = [0.01, 0.017, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
BLOCK = 21  # bootstrap block length (one month, captures serial structure)
N_BOOT = 2000  # bootstrap resamples
BOOT_SEED = 0

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_parquet(HERE / "data/sp500_pct_returns.parquet")
R = df.to_numpy()  # raw daily fractional returns (T, n)
T, n = R.shape
print("=" * 78)
print("Out-of-sample rolling-window minimum-variance backtest  (S&P 500)")
print("=" * 78)
print(f"n={n} assets, T={T} days ({df.index[0].date()} -> {df.index[-1].date()})")


def _drift_adjust(w_prev, r_block):
    """Return the buy-and-hold drifted weights at the end of a holding block."""
    growth = np.prod(1.0 + r_block, axis=0)
    w_d = w_prev * growth
    s = w_d.sum()
    return w_d / s if s > 0 else w_prev


def backtest(est_window):
    """Run the rolling backtest for a given estimation window.

    Returns (daily, turnover, cg, oracle_alphas) where daily[a] is the
    concatenated OOS daily portfolio-return series for intensity a.
    """
    daily = {a: [] for a in ALPHAS}
    turnover = {a: [] for a in ALPHAS}
    cg = {a: [] for a in ALPHAS}
    prev_w = dict.fromkeys(ALPHAS)
    prev_block = dict.fromkeys(ALPHAS)
    oracle_alphas = []

    for t in range(est_window, T - 1, HOLD):
        R_in = R[t - est_window : t]
        X = R_in - R_in.mean(axis=0)
        bar_lam = float(np.linalg.norm(X, "fro") ** 2) / (n * est_window)
        target = bar_lam * np.eye(n)
        a_oracle, _ = lw_alpha_and_target(X)
        oracle_alphas.append(a_oracle)

        R_out = R[t : t + HOLD]
        if R_out.shape[0] == 0:
            continue

        for a in ALPHAS:
            w, _outer, inner = MinVarProblem(X, alpha=a, target=target).solve_cg()
            cg[a].append(inner)
            daily[a].extend((R_out @ w).tolist())
            if prev_w[a] is not None:
                w_drift = _drift_adjust(prev_w[a], prev_block[a])
                turnover[a].append(0.5 * float(np.abs(w - w_drift).sum()))
            prev_w[a] = w
            prev_block[a] = R_out

    return daily, turnover, cg, oracle_alphas


def block_indices(rng, length, block, n_blocks):
    """Return a circular moving-block bootstrap index array of given length."""
    starts = rng.integers(0, length, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + block) % length) for s in starts])
    return idx[:length]


# ---------------------------------------------------------------------------
# Primary window L = 504  (n/L ~ 0.98, high-dimensional regime)
# ---------------------------------------------------------------------------
EST = 504
print(f"\nPrimary estimation window L={EST} days (n/L={n / EST:.2f}), hold H={HOLD} days")
daily, turnover, cg, oracle_alphas = backtest(EST)
n_months = len(daily[ALPHAS[0]]) // HOLD
print(f"{n_months} out-of-sample months")
print(
    f"Mean per-window LW oracle alpha = {np.mean(oracle_alphas):.4f} "
    f"(range {np.min(oracle_alphas):.4f}-{np.max(oracle_alphas):.4f})\n"
)

rows = []
daily_arr = {a: np.asarray(daily[a]) for a in ALPHAS}
print(f"{'alpha':>6}  {'OOSvol%':>8}  {'ret%':>7}  {'Sharpe':>7}  {'turn%':>7}  {'CGiters':>8}")
print("-" * 60)
for a in ALPHAS:
    d = daily_arr[a]
    ann_vol = float(d.std(ddof=1) * np.sqrt(ANN) * 100)
    ann_ret = float(d.mean() * ANN * 100)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    turn = float(np.mean(turnover[a]) * 100)
    iters = float(np.mean(cg[a]))
    rows.append((a, ann_vol, ann_ret, sharpe, turn, iters))
    print(f"{a:>6.3f}  {ann_vol:>8.2f}  {ann_ret:>7.2f}  {sharpe:>7.3f}  {turn:>7.2f}  {iters:>8.1f}")

vol_opt = min(rows, key=lambda r: r[1])
print(f"\nOOS-volatility-minimising alpha = {vol_opt[0]:.3f}  (vol {vol_opt[1]:.2f}%)")

# ---------------------------------------------------------------------------
# R1: moving-block bootstrap CIs + paired test (alpha=0.5 vs 0.017)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(BOOT_SEED)
length = len(daily_arr[0.5])
n_blocks = int(np.ceil(length / BLOCK))
boot_vol = {a: np.empty(N_BOOT) for a in ALPHAS}
boot_diff = np.empty(N_BOOT)  # vol(0.5) - vol(0.017), paired
for b in range(N_BOOT):
    idx = block_indices(rng, length, BLOCK, n_blocks)
    for a in ALPHAS:
        boot_vol[a][b] = daily_arr[a][idx].std(ddof=1) * np.sqrt(ANN) * 100
    boot_diff[b] = boot_vol[0.5][b] - boot_vol[0.017][b]

print(f"\nMoving-block bootstrap 95% CI for OOS volatility (block={BLOCK}, B={N_BOOT}):")
for a in (0.017, 0.30, 0.50):
    lo, hi = np.percentile(boot_vol[a], [2.5, 97.5])
    print(f"  alpha={a:.3f}: {np.mean(boot_vol[a]):.2f}%  CI [{lo:.2f}, {hi:.2f}]")
d_lo, d_hi = np.percentile(boot_diff, [2.5, 97.5])
p_worse = float(np.mean(boot_diff > 0))  # fraction of resamples where 0.5 is worse
print(
    f"Paired vol(0.5) - vol(0.017): mean {boot_diff.mean():+.3f} pp, "
    f"95% CI [{d_lo:+.3f}, {d_hi:+.3f}], P(0.5 worse)={p_worse:.2f}"
)

# ---------------------------------------------------------------------------
# R1: robustness to a shorter estimation window L = 252
# ---------------------------------------------------------------------------
EST2 = 252
print(f"\nRobustness: estimation window L={EST2} days (n/L={n / EST2:.2f})")
daily2, turnover2, cg2, oracle2 = backtest(EST2)
rows2 = []
for a in ALPHAS:
    d = np.asarray(daily2[a])
    ann_vol = float(d.std(ddof=1) * np.sqrt(ANN) * 100)
    iters = float(np.mean(cg2[a]))
    rows2.append((a, ann_vol, iters))
vol_opt2 = min(rows2, key=lambda r: r[1])
v05_2 = {r[0]: r[1] for r in rows2}[0.50]
v017_2 = {r[0]: r[1] for r in rows2}[0.017]
print(f"  mean oracle alpha = {np.mean(oracle2):.4f}")
print(f"  OOS-vol-minimising alpha = {vol_opt2[0]:.3f} (vol {vol_opt2[1]:.2f}%)")
print(f"  vol(alpha=0.5)={v05_2:.2f}%  vol(alpha=0.017)={v017_2:.2f}%")

# ---------------------------------------------------------------------------
# Seed sensitivity (referee m3): CG deterministic (x0=0); proximal random.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Seed sensitivity  (last L=504 window, alpha=0.5)")
print("=" * 78)
t_last = list(range(EST, T - 1, HOLD))[-1]
X_last = R[t_last - EST : t_last]
X_last = X_last - X_last.mean(axis=0)
bar_lam = float(np.linalg.norm(X_last, "fro") ** 2) / (n * EST)
tgt = bar_lam * np.eye(n)
cg_counts = [MinVarProblem(X_last, alpha=0.5, target=tgt).solve_cg()[2] for _ in range(10)]
prox_counts = [MinVarProblem(X_last, alpha=0.5, target=tgt).solve_proximal()[1] for _ in range(10)]
print(f"CG inner iters over 10 runs:       {set(cg_counts)}  (deterministic, x0=0)")
print(
    f"Proximal iters over 10 runs: min={min(prox_counts)} max={max(prox_counts)} "
    f"mean={np.mean(prox_counts):.0f} std={np.std(prox_counts):.1f}  (random start)"
)

# ---------------------------------------------------------------------------
# Write LaTeX table macro (primary L=504)
# ---------------------------------------------------------------------------
lines = []
for a, ann_vol, _ann_ret, sharpe, turn, iters in rows:
    lines.append(f"{a:>5.3f} & {ann_vol:>6.2f} & {sharpe:>6.3f} & {turn:>6.2f} & {iters:>6.0f} \\\\\n")
(TABLES / "oos_defs.tex").write_text(f"\\def\\dataOos{{%\n{''.join(lines)}}}\n")
print(f"\n  -> wrote {TABLES / 'oos_defs.tex'}")

# ---------------------------------------------------------------------------
# Figure: OOS volatility (left, with bootstrap band) and CG iterations (right)
# ---------------------------------------------------------------------------
a_arr = np.array([r[0] for r in rows])
vol_arr = np.array([r[1] for r in rows])
it_arr = np.array([r[5] for r in rows])
vol_lo = np.array([np.percentile(boot_vol[a], 2.5) for a in ALPHAS])
vol_hi = np.array([np.percentile(boot_vol[a], 97.5) for a in ALPHAS])

fig, ax1 = plt.subplots(figsize=(4.5, 3.2))
c_vol, c_it = "#1f77b4", "#ff7f0e"
ax1.fill_between(a_arr, vol_lo, vol_hi, color=c_vol, alpha=0.18, linewidth=0)
ax1.plot(a_arr, vol_arr, marker="o", markersize=4, color=c_vol, label="OOS volatility")
ax1.set_xscale("log")
ax1.set_xlabel(r"Shrinkage intensity $\alpha$")
ax1.set_ylabel("Realized OOS volatility (\\% ann.)", color=c_vol)
ax1.tick_params(axis="y", labelcolor=c_vol)
ax1.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

ax2 = ax1.twinx()
ax2.plot(a_arr, it_arr, marker="s", markersize=4, color=c_it, linestyle="--", label="CG iterations")
ax2.set_ylabel("Mean CG iterations", color=c_it)
ax2.tick_params(axis="y", labelcolor=c_it)
ax2.set_yscale("log")

ax1.set_title(r"Out-of-sample risk vs.\ conditioning  (S\&P 500)")
fig.tight_layout(pad=1.0)
fig.savefig(GRAPHS / "minvar_oos.pdf", bbox_inches="tight")
fig.savefig(GRAPHS / "minvar_oos.png", bbox_inches="tight", dpi=150)
print(f"  -> wrote {GRAPHS / 'minvar_oos.pdf'}")
