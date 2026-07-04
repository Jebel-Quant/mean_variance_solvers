"""External-solver benchmark for "Non-Negative Conjugate Gradients".

Benchmarks the paper's active-set solver (Algorithm 1, from nncg.py) against
established baselines on the planted-optimum synthetic family of Section 7.2,
instantiating the rows of the complexity table (Table 1):

  * scipy.optimize.nnls        -- Lawson-Hanson active set (the NNLS reference)
  * Clarabel                   -- modern interior-point QP solver
  * Active set + direct solve  -- Algorithm 1, dense inner solve
  * Active set + CG            -- Algorithm 1, matrix-free CG inner solve

Every solver is checked against the known planted optimum x*, so accuracy is
compared on equal footing rather than by tuning tolerances.

Usage:
    uv run experiment_nncg_bench.py   # from non_negative_cg/experiment/

Outputs (files):
    graphs/nncg_bench.pdf     wall-clock time vs n (log-log, kappa=1e4)
    tables/nncg_bench.tex     time vs kappa at n=500 + accuracy (booktabs)
    tables/nncg_bench_defs.tex  headline numbers as \\newcommand macros

Unlike experiment_nncg.py (NumPy only), this script depends on SciPy and
Clarabel; the paper's own study remains reproducible without them.

Hardware/software recorded when the paper's numbers were generated:
Apple M4 Pro, Python 3.13, NumPy 2.x, SciPy 1.x, Clarabel 0.x (see uv lock).
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "clarabel",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///

from __future__ import annotations

import time
from pathlib import Path

import clarabel
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular
from scipy.optimize import nnls as scipy_nnls

from nncg import make_problem, shaw, solve_nnqp

HERE = Path(__file__).parent
GRAPHS = HERE.parent / "graphs"
TABLES = HERE.parent / "tables"
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
# Solver wrappers: each takes (A, b) and returns x
# ---------------------------------------------------------------------------

def run_lawson_hanson(A, b):
    """scipy.optimize.nnls on min ||Mx-d||^2 with M^T M = A, M^T d = b.

    M = L^T from the Cholesky factor A = L L^T, d = L^{-1} b; the
    factorisation is included in the timing, as any NNLS user would pay it.
    (np.linalg.cholesky, not scipy.linalg.cholesky: the latter's lower=True
    path returns a corrupted factor in SciPy 1.18.0.)
    """
    L = np.linalg.cholesky(A)
    d = solve_triangular(L, b, lower=True)
    x, _ = scipy_nnls(L.T, d)
    return x


def run_clarabel(A, b):
    """Clarabel interior point on min 1/2 x^T A x - b^T x s.t. x >= 0."""
    n = len(b)
    P = sp.triu(sp.csc_matrix(A), format="csc")
    Ac = -sp.identity(n, format="csc")
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(P, -b, Ac, np.zeros(n),
                                    [clarabel.NonnegativeConeT(n)], settings)
    sol = solver.solve()
    return np.asarray(sol.x)


def run_as_direct(A, b):
    """Algorithm 1 with a dense direct inner solve."""
    return solve_nnqp(A, b, inner="exact")["x"]


def run_as_cg(A, b):
    """Algorithm 1 with the matrix-free CG inner solve (the paper's method)."""
    return solve_nnqp(A, b)["x"]


METHODS = [
    ("Lawson--Hanson (SciPy)", run_lawson_hanson, "#1f77b4", "o"),
    ("Interior point (Clarabel)", run_clarabel, "#2ca02c", "^"),
    ("Active set $+$ direct", run_as_direct, "#d62728", "s"),
    ("Active set $+$ CG", run_as_cg, "#ff7f0e", "D"),
]

SKIP_AFTER_SECONDS = 60.0  # drop a method from larger sizes once it exceeds this


def bench(A, b, x_star, fn):
    """Run one solver once; return (seconds, max abs error vs planted optimum)."""
    t0 = time.perf_counter()
    x = fn(A, b)
    dt = time.perf_counter() - t0
    return dt, float(np.max(np.abs(x - x_star)))


# ---------------------------------------------------------------------------
# Sweep 1: wall-clock vs n at fixed kappa = 1e4
# ---------------------------------------------------------------------------

KAPPA_N = 1e4
NS = [100, 200, 500, 1000, 2000]

print("=" * 78)
print(f"Wall-clock vs n  (kappa={KAPPA_N:.0e}, support 50%, mean over seeds)")
print("=" * 78)
header = f"{'n':>6}" + "".join(f"  {name.split(' (')[0][:18]:>20}" for name, *_ in METHODS)
print(header)

times = {name: [] for name, *_ in METHODS}
errs = {name: 0.0 for name, *_ in METHODS}
skipped = {name: False for name, *_ in METHODS}
for n in NS:
    seeds = range(3) if n <= 1000 else range(1)
    row = f"{n:>6}"
    for name, fn, *_ in METHODS:
        if skipped[name]:
            times[name].append(np.nan)
            row += f"  {'skipped':>20}"
            continue
        ts = []
        for sd in seeds:
            A, b, x_star, _ = make_problem(n, KAPPA_N, seed=sd)
            dt, err = bench(A, b, x_star, fn)
            ts.append(dt)
            errs[name] = max(errs[name], err)
        t = float(np.mean(ts))
        times[name].append(t)
        if t > SKIP_AFTER_SECONDS:
            skipped[name] = True
        row += f"  {t:>19.3f}s"
    print(row)

print("\nmax |x - x*| per method: "
      + ", ".join(f"{name.split(' (')[0]}: {errs[name]:.1e}" for name, *_ in METHODS))


# ---------------------------------------------------------------------------
# Sweep 2: wall-clock vs kappa at fixed n = 500
# ---------------------------------------------------------------------------

N_K = 500
KAPPAS_B = [1e2, 1e4, 1e6]

print()
print("=" * 78)
print(f"Wall-clock vs kappa  (n={N_K}, support 50%, mean over 3 seeds)")
print("=" * 78)
print(f"{'kappa':>9}" + "".join(f"  {name.split(' (')[0][:18]:>20}" for name, *_ in METHODS))

ktimes = {name: [] for name, *_ in METHODS}
kerrs = {name: 0.0 for name, *_ in METHODS}
for kap in KAPPAS_B:
    row = f"{kap:>9.0e}"
    for name, fn, *_ in METHODS:
        ts = []
        for sd in range(3):
            A, b, x_star, _ = make_problem(N_K, kap, seed=sd)
            dt, err = bench(A, b, x_star, fn)
            ts.append(dt)
            kerrs[name] = max(kerrs[name], err)
        t = float(np.mean(ts))
        ktimes[name].append(t)
        row += f"  {t:>19.3f}s"
    print(row)


# ---------------------------------------------------------------------------
# Sweep 3: wall-clock on the official `shaw` ill-posed NNLS problem
#          (active set vs Lawson-Hanson on the regularised Gram operator)
# ---------------------------------------------------------------------------

SHAW_NS = [256, 512, 1024]
SHAW_ALPHA = 1e-4                                     # matches experiment_nncg_regu.py
SHAW_NOISE = 1e-3
SHAW_REP = 3                                          # min over repeats

print()
print("=" * 78)
print(f"Wall-clock on shaw (alpha={SHAW_ALPHA:.0e}, noise={SHAW_NOISE:.0e}, "
      f"min over {SHAW_REP} runs)")
print("=" * 78)
print(f"{'n':>6}  {'LH (s)':>10}  {'AS+dir (s)':>11}  {'AS+CG (s)':>11}  "
      f"{'LH/AS+CG':>9}")

shaw_lh_vs_ascg = {}
shaw_lh_vs_asd = {}
for n in SHAW_NS:
    M, _, d = shaw(n)
    rng = np.random.default_rng(0)
    g = rng.standard_normal(n)
    d_noisy = d + SHAW_NOISE * np.linalg.norm(d) * g / np.linalg.norm(g)
    gram = M.T @ M
    A = (1.0 - SHAW_ALPHA) * gram + SHAW_ALPHA * (np.trace(gram) / n) * np.eye(n)
    b = M.T @ d_noisy
    t = {}
    for name, fn in [("lh", run_lawson_hanson), ("asd", run_as_direct),
                     ("ascg", run_as_cg)]:
        best = np.inf
        for _ in range(SHAW_REP):
            t0 = time.perf_counter()
            fn(A, b)
            best = min(best, time.perf_counter() - t0)
        t[name] = best
    shaw_lh_vs_ascg[n] = t["lh"] / t["ascg"]
    shaw_lh_vs_asd[n] = t["lh"] / t["asd"]
    print(f"{n:>6}  {t['lh']:>10.3f}  {t['asd']:>11.3f}  {t['ascg']:>11.3f}  "
          f"{shaw_lh_vs_ascg[n]:>8.0f}x")


# ---------------------------------------------------------------------------
# Figure: time vs n
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(4.5, 3.2))
for name, _, color, marker in METHODS:
    ax.plot(NS, times[name], marker=marker, markersize=4, color=color, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Problem size $n$")
ax.set_ylabel("Wall-clock time (s)")
ax.set_title(rf"Solver comparison  ($\kappa = 10^4$)")
ax.legend(framealpha=0.9, fontsize=7)
ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
fig.tight_layout(pad=1.0)
fig.savefig(GRAPHS / "nncg_bench.pdf", bbox_inches="tight")
fig.savefig(GRAPHS / "nncg_bench.png", bbox_inches="tight", dpi=150)
print(f"\nSaved {GRAPHS / 'nncg_bench.pdf'}")


# ---------------------------------------------------------------------------
# Table: time vs kappa at n=500, plus accuracy
# ---------------------------------------------------------------------------

def sci(v):
    """LaTeX scientific notation, e.g. $2\\cdot10^{-9}$."""
    if v == 0.0:
        return "$0$"
    mant, expo = f"{v:.0e}".split("e")
    return f"${mant}\\cdot10^{{{int(expo)}}}$"


with open(TABLES / "nncg_bench.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg_bench.py -- do not edit by hand.\n")
    fh.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    fh.write(" & \\multicolumn{3}{c}{Time (s), $n = 500$} & \\\\\n")
    fh.write("\\cmidrule(lr){2-4}\n")
    fh.write("Method & $\\kappa = 10^2$ & $\\kappa = 10^4$ & $\\kappa = 10^6$ & "
             "$\\max\\lVert x - x^\\star\\rVert_\\infty$ \\\\\n\\midrule\n")
    for name, *_ in METHODS:
        cells = " & ".join(f"{t:.3f}" for t in ktimes[name])
        fh.write(f"{name} & {cells} & {sci(max(errs[name], kerrs[name]))} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(f"Saved {TABLES / 'nncg_bench.tex'}")

with open(TABLES / "nncg_bench_defs.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg_bench.py -- do not edit by hand.\n")
    largest_done = max(n for n, t in zip(NS, times["Active set $+$ CG"]) if np.isfinite(t))
    fh.write(f"\\newcommand{{\\nncgBenchNmax}}{{{largest_done}}}\n")
    fh.write(f"\\newcommand{{\\nncgBenchErr}}{{{max(errs.values()):.0e}}}\n")
    # kappa sensitivity at n=500: ratio time(1e6)/time(1e2) per method
    for key, name in [("LH", "Lawson--Hanson (SciPy)"),
                      ("IP", "Interior point (Clarabel)"),
                      ("ASD", "Active set $+$ direct"),
                      ("ASCG", "Active set $+$ CG")]:
        fh.write(f"\\newcommand{{\\nncgBench{key}Sens}}"
                 f"{{{ktimes[name][-1] / ktimes[name][0]:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgBench{key}Mid}}{{{ktimes[name][1]:.3f}}}\n")
    # speed ratios at the largest n completed by every method
    i_last = max(i for i in range(len(NS))
                 if all(np.isfinite(times[name][i]) for name, *_ in METHODS))
    fh.write(f"\\newcommand{{\\nncgBenchRatioN}}{{{NS[i_last]}}}\n")
    t_asd = times["Active set $+$ direct"][i_last]
    t_ascg = times["Active set $+$ CG"][i_last]
    fh.write(f"\\newcommand{{\\nncgBenchLHvsASD}}"
             f"{{{times['Lawson--Hanson (SciPy)'][i_last] / t_asd:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgBenchIPvsASD}}"
             f"{{{times['Interior point (Clarabel)'][i_last] / t_asd:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgBenchLHvsASCG}}"
             f"{{{times['Lawson--Hanson (SciPy)'][i_last] / t_ascg:.0f}}}\n")
    # shaw ill-posed problem: speedup over Lawson-Hanson at the largest size
    shaw_nmax = SHAW_NS[-1]
    fh.write(f"\\newcommand{{\\nncgBenchShawN}}{{{shaw_nmax}}}\n")
    fh.write(f"\\newcommand{{\\nncgBenchShawLHvsASCG}}"
             f"{{{shaw_lh_vs_ascg[shaw_nmax]:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgBenchShawLHvsASD}}"
             f"{{{shaw_lh_vs_asd[shaw_nmax]:.0f}}}\n")
print(f"Saved {TABLES / 'nncg_bench_defs.tex'}")
print("\nDone.")
