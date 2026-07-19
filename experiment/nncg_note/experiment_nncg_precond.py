"""Preconditioner comparison: CG, Jacobi, Nystrom, GlobalNystrom on repeated solves.

Companion figure for Section "Preconditioning" (s6_conditioning.tex), which
claims that GlobalNystrom "pays its (larger, whole-operator) sketch cost once
and amortises it across every outer step of a solve and across the
support-stable warm-started re-solves ... exactly where a resketched-per-block
Nystrom pays repeatedly for work GlobalNystrom does once." This script makes
that claim checkable: the same SPD operator A is solved against many
independent right-hand sides, sharing one solver instance (and hence, for
GlobalNystrom, one cached sketch -- see nncg.inner.GlobalNystrom, which
memoises its sketch keyed on operator identity) across all of them, and plots
cumulative wall time against the number of right-hand sides solved so far.

Modelled directly on github.com/Jebel-Quant/nncg's
book/marimo/notebooks/04_solver_comparison.py: the test problem
(make_problem_with_spectrum) plants a handful of dominant eigenvalues over a
geometric tail -- the shape that gives Nystrom-style sketching something cheap
to deflate, per that notebook's own framing (a smoothly decaying spectrum
gives a sketch nothing to buy). Unlike the notebook's interactive sliders,
this script sweeps the number of repeated solves as its varying parameter, to
produce the one static figure the paper needs.

GlobalNystrom is not yet in a released nncg version (see pyproject.toml,
pinned to the commit that added it) -- CG and Jacobi are included alongside
Nystrom/GlobalNystrom as a baseline showing that repetition alone buys
neither of them anything; only the sketch-once design benefits from reuse.

Usage:
    uv run python -m nncg_note.experiment_nncg_precond

Outputs:
    graphs/nncg_precond.pdf        cumulative wall time vs. number of solves
    tables/nncg_precond_defs.tex   headline numbers as \\newcommand macros
"""

from __future__ import annotations

import argparse
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cvx.linalg import DenseOperator
from common.util.runner import SMOKE, output_dirs
from pathlib import Path

from nncg import ActiveSetConfig, ActiveSetSolver, CG, GlobalNystrom, Jacobi, Nystrom, NystromConfig

HERE = Path(__file__).resolve().parents[1]  # experiment/ root (graphs, tables)
GRAPHS, TABLES = output_dirs(HERE)

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

N = 500
N_DOMINANT = 3
TOP_GAP = 1000.0
TAIL_KAPPA = 500.0
RANK = 4
N_SOLVES = 60
if SMOKE:  # shrink everything so the smoke run stays fast
    N = 60
    N_SOLVES = 8


def make_problem_with_spectrum(n, n_dominant, top_gap, tail_kappa, support_frac=0.5, seed=0):
    """Planted-optimum SPD problem with a spectral gap (Jebel-Quant/nncg notebook 04).

    `n_dominant` eigenvalues sit at `top_gap * tail_kappa` and above, over a
    geometric tail spanning `[1, tail_kappa]` -- a handful of directions a
    low-rank sketch can deflate cheaply, over a tail neither Nystrom nor
    GlobalNystrom target.
    """
    rng = np.random.default_rng(seed)
    top = tail_kappa * np.geomspace(top_gap, 2.0, n_dominant)
    tail = np.geomspace(1.0, tail_kappa, n - n_dominant)
    eig = np.concatenate([top, tail])
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = (q * eig) @ q.T
    a = 0.5 * (a + a.T)

    k = max(1, round(support_frac * n))
    perm = rng.permutation(n)
    supp = perm[:k]
    x_star = np.zeros(n)
    x_star[supp] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
    b = a @ x_star - s_star
    return a, b, x_star


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--n-dominant", type=int, default=N_DOMINANT)
    ap.add_argument("--top-gap", type=float, default=TOP_GAP)
    ap.add_argument("--tail-kappa", type=float, default=TAIL_KAPPA)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--n-solves", type=int, default=N_SOLVES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a, _, _ = make_problem_with_spectrum(
        args.n, args.n_dominant, args.top_gap, args.tail_kappa, seed=args.seed)
    op = DenseOperator(a)
    cfg = ActiveSetConfig(tol=1e-8)

    # Fresh random right-hand sides per solve (as in the notebook's reuse
    # experiment) -- these have no relation to the planted x_star above, which
    # is only the optimum for the one b used to construct the spectrum, so
    # correctness here is checked by cross-solver agreement instead: every
    # solver must reach the same free-set optimum for each rhs regardless of
    # preconditioner.
    rng = np.random.default_rng(args.seed + 1)
    rhs = [rng.standard_normal(args.n) for _ in range(args.n_solves)]

    solvers = {
        "CG (no preconditioner)": CG(),
        "Jacobi": Jacobi(),
        "Nystrom (resketched per solve)": Nystrom(nystrom=NystromConfig(rank=args.rank, seed=0)),
        "GlobalNystrom (sketched once)": GlobalNystrom(nystrom=NystromConfig(rank=args.rank, seed=0)),
    }

    cumulative = {}
    solutions = {}
    for name, inner in solvers.items():
        cum, t0 = [], time.perf_counter()
        xs = []
        for b in rhs:
            res = ActiveSetSolver(inner=inner, config=cfg).solve(op, b)
            xs.append(res.x)
            cum.append((time.perf_counter() - t0) * 1e3)
        cumulative[name] = cum
        solutions[name] = xs
        print(f"{name}: {cum[-1]:.2f} ms cumulative over {args.n_solves} solves")

    ref_name = "CG (no preconditioner)"
    max_disagreement = max(
        float(np.max(np.abs(np.array(solutions[name]) - np.array(solutions[ref_name]))))
        for name in solvers if name != ref_name
    )
    print(f"Max disagreement vs {ref_name} across all solvers/solves: {max_disagreement:.2e}")

    nystrom_key = "Nystrom (resketched per solve)"
    global_key = "GlobalNystrom (sketched once)"
    nystrom_cum = np.array(cumulative[nystrom_key])
    global_cum = np.array(cumulative[global_key])
    ahead = np.where(global_cum < nystrom_cum)[0]
    crossover = int(ahead[0]) + 1 if ahead.size else -1
    final_speedup = float(nystrom_cum[-1] / global_cum[-1])
    print(f"GlobalNystrom overtakes Nystrom at solve #{crossover}; "
          f"{final_speedup:.2f}x cumulative speedup after {args.n_solves} solves")

    # -----------------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    colors = {"CG (no preconditioner)": "#888888", "Jacobi": "#55aa88",
              "Nystrom (resketched per solve)": "#5588bb", "GlobalNystrom (sketched once)": "#bb5588"}
    for name, cum in cumulative.items():
        ax.plot(range(1, len(cum) + 1), cum, marker="o", markersize=3, label=name, color=colors[name])
    ax.set_xlabel("right-hand side #")
    ax.set_ylabel("cumulative wall time (ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "nncg_precond.pdf")
    print(f"\nSaved {GRAPHS / 'nncg_precond.pdf'}")

    # -----------------------------------------------------------------------
    # LaTeX macros
    # -----------------------------------------------------------------------
    with open(TABLES / "nncg_precond_defs.tex", "w") as fh:
        fh.write("% Generated by experiment_nncg_precond.py -- do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\nncgPrecondN}}{{{args.n}}}\n")
        fh.write(f"\\newcommand{{\\nncgPrecondRank}}{{{args.rank}}}\n")
        fh.write(f"\\newcommand{{\\nncgPrecondNSolves}}{{{args.n_solves}}}\n")
        fh.write(f"\\newcommand{{\\nncgPrecondCrossover}}{{{crossover}}}\n")
        fh.write(f"\\newcommand{{\\nncgPrecondFinalSpeedup}}{{{final_speedup:.2f}}}\n")
        fh.write(f"\\newcommand{{\\nncgPrecondNystromFinalMs}}{{{nystrom_cum[-1]:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgPrecondGlobalFinalMs}}{{{global_cum[-1]:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgPrecondMaxError}}{{{max_disagreement:.1e}}}\n")
    print(f"Saved {TABLES / 'nncg_precond_defs.tex'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
