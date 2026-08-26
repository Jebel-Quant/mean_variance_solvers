r"""What a warm-started solve costs, and where the cost comes from.

Section 8 of the quadprog note derives the recovery formulas for a family of
programs differing only in the linear term, and states their cost: ``O(n^2)``,
all of it in ``x_u = J J^T a``, with ``O(nk + k^2)`` for everything after it. The
``n^2`` does not shrink with the active set, because ``J`` is dense once the first
insertion has touched it.

That is a flop count, and a flop count is a claim about asymptotics rather than
about what a solve costs. This script measures the cost, and separates the two
regimes the flop count alone does not reveal.

The measurement that matters holds the active set *fixed and small* while ``n``
grows. Under ``O(nk)`` with ``k`` fixed, doubling ``n`` doubles the cost; under
``O(n^2)`` it quadruples it. The distinction is invisible on a box-constrained
family, where ``k`` grows with ``n`` and the two predictions agree, which is
exactly how an ``O(nk)`` reading survives casual checking.

The second regime is the one a caller notices. At small ``n`` a hit costs what its
dozen array operations cost to dispatch, not what its arithmetic costs, so the
measured cost is nearly flat in ``n`` -- and that flatness is an overhead floor,
not a property of the recovery. It ends where the ``n^2`` overtakes the dispatch.

Usage:
    uv run python -m quadprog_note.experiment_qp_sweep   # from experiment/

Outputs:
    graphs/quadprog_sweep.pdf        hit cost vs n, fixed k against growing k
    tables/quadprog_sweep_defs.tex   headline numbers as \newcommand macros

NumPy + SciPy + Matplotlib + cvx-quadprog.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common.util.runner import SMOKE, output_dirs

from cvx.quadprog import Sweep, solve_qp

HERE = Path(__file__).resolve().parents[1]
GRAPHS, TABLES = output_dirs(HERE)

mpl.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150})

SIZES = [100, 200] if SMOKE else [100, 200, 400, 800, 1600]
REPEATS = 2 if SMOKE else 25
FIXED_K = 5

# Relative perturbation applied to the linear term between solves. Small enough
# that the cached active set almost always stays optimal, which is the case the
# class exists for.
PERTURB = 1e-3


def hessian(rng, n):
    b_mat = rng.standard_normal((n, n))
    return b_mat @ b_mat.T + n * np.eye(n)


def fixed_k_problem(rng, n, k):
    """Bounds placed so that exactly `k` of them bind at the optimum."""
    g = hessian(rng, n)
    a = rng.standard_normal(n)
    xu = np.linalg.solve(g, a)
    b = xu - 1.0
    b[:k] = xu[:k] + 0.5
    return g, a, np.eye(n), b


def growing_k_problem(rng, n):
    """A box family whose active set grows with n, where O(nk) and O(n^2) agree."""
    g = hessian(rng, n)
    a = rng.standard_normal(n)
    xu = np.linalg.solve(g, a)
    return g, a, np.eye(n), xu + rng.standard_normal(n) * 0.1


def time_hits(g, a, c, b, repeats):
    """Return (k, seconds per hit, hit count) for a family of perturbed solves."""
    sweep = Sweep(g, c, b, meq=0)
    first = sweep.solve(a)
    k = len(first.iact)

    terms = [a * (1.0 + PERTURB * (i + 1) / repeats) for i in range(repeats)]
    hits_before = sweep.hits
    start = time.perf_counter()
    for term in terms:
        sweep.solve(term)
    elapsed = time.perf_counter() - start
    return k, elapsed / repeats, sweep.hits - hits_before


def time_cold(g, a, c, b, repeats):
    """Return seconds per solve for the same family solved from scratch."""
    terms = [a * (1.0 + PERTURB * (i + 1) / repeats) for i in range(repeats)]
    start = time.perf_counter()
    for term in terms:
        solve_qp(g, term, c, b, 0)
    return (time.perf_counter() - start) / repeats


def run():
    fixed, growing = [], []
    for n in SIZES:
        rng = np.random.default_rng(n)
        g, a, c, b = fixed_k_problem(rng, n, FIXED_K)
        k, per_hit, hits = time_hits(g, a, c, b, REPEATS)
        cold = time_cold(g, a, c, b, REPEATS)
        fixed.append({"n": n, "k": k, "hit": per_hit, "cold": cold, "hits": hits})
        print(f"fixed k    n={n:<5} k={k:<4} hit={per_hit * 1e6:>8.1f} us  "
              f"cold={cold * 1e6:>9.1f} us  hits={hits}/{REPEATS}")

        rng = np.random.default_rng(n + 7)
        g, a, c, b = growing_k_problem(rng, n)
        k, per_hit, hits = time_hits(g, a, c, b, REPEATS)
        cold = time_cold(g, a, c, b, REPEATS)
        growing.append({"n": n, "k": k, "hit": per_hit, "cold": cold, "hits": hits})
        print(f"growing k  n={n:<5} k={k:<4} hit={per_hit * 1e6:>8.1f} us  "
              f"cold={cold * 1e6:>9.1f} us  hits={hits}/{REPEATS}")
    return fixed, growing


def log_x_ticks(ax, sizes) -> None:
    """Label a log x-axis at the sizes actually measured, and nowhere else.

    A log axis spanning a narrow range -- 12 to 200 here -- puts minor ticks at
    2, 3, 4, 6 times each decade, and at figure width their labels overlap into an
    illegible smear. The sizes are a short discrete list, so label exactly those
    and silence the minor formatter.
    """
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2)


def figure(fixed, growing) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ns = np.array([row["n"] for row in fixed], dtype=float)
    hit_fixed = np.array([row["hit"] for row in fixed]) * 1e6
    hit_grow = np.array([row["hit"] for row in growing]) * 1e6
    cold_fixed = np.array([row["cold"] for row in fixed]) * 1e6

    ax_a.plot(ns, hit_fixed, color="#1f77b4", marker="o", markersize=4,
              label=f"hit, $k = {FIXED_K}$ fixed")
    ax_a.plot(ns, hit_grow, color="#d62728", marker="s", markersize=4, linestyle="--",
              label="hit, $k$ grows with $n$")
    ax_a.plot(ns, cold_fixed, color="#7f7f7f", marker="^", markersize=4, linestyle=":",
              label="cold solve")
    # Reference slopes, anchored at the largest size where arithmetic dominates.
    ax_a.plot(ns, hit_fixed[-1] * (ns / ns[-1]) ** 2, color="#1f77b4",
              linewidth=0.8, alpha=0.5, linestyle="-.", label="$O(n^2)$")
    ax_a.plot(ns, hit_fixed[-1] * (ns / ns[-1]), color="#2ca02c",
              linewidth=0.8, alpha=0.5, linestyle="-.", label="$O(n)$")
    ax_a.set_xscale("log")
    log_x_ticks(ax_a, SIZES)
    ax_a.set_yscale("log")
    ax_a.set_xlabel("$n$")
    ax_a.set_ylabel("per solve ($\\mu$s)")
    ax_a.set_title("Cost of a reused solve")
    ax_a.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax_a.legend(framealpha=0.9, fontsize=7)

    # Local slope, which is what separates the two regimes.
    slopes = np.diff(np.log(hit_fixed)) / np.diff(np.log(ns))
    mids = np.sqrt(ns[:-1] * ns[1:])
    ax_b.plot(mids, slopes, color="#1f77b4", marker="o", markersize=4)
    ax_b.axhline(2.0, color="#1f77b4", linewidth=0.8, linestyle="-.", alpha=0.6)
    ax_b.axhline(1.0, color="#2ca02c", linewidth=0.8, linestyle="-.", alpha=0.6)
    ax_b.text(mids[0], 2.06, "$O(n^2)$", fontsize=7, color="#1f77b4")
    ax_b.text(mids[0], 1.06, "$O(n)$", fontsize=7, color="#2ca02c")
    ax_b.set_xscale("log")
    log_x_ticks(ax_b, SIZES)
    ax_b.set_xlabel("$n$")
    ax_b.set_ylabel("local exponent $d\\log t / d\\log n$")
    ax_b.set_ylim(-0.2, 2.8)
    ax_b.set_title("Dispatch-bound below, arithmetic-bound above")
    ax_b.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout(pad=0.8)
    fig.savefig(GRAPHS / "quadprog_sweep.pdf", bbox_inches="tight")
    fig.savefig(GRAPHS / "quadprog_sweep.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved {GRAPHS / 'quadprog_sweep.pdf'}")


def emit(fixed, growing) -> None:
    ns = np.array([row["n"] for row in fixed], dtype=float)
    hit = np.array([row["hit"] for row in fixed])
    slopes = np.diff(np.log(hit)) / np.diff(np.log(ns))

    path = TABLES / "quadprog_sweep_defs.tex"
    with open(path, "w") as fh:
        fh.write("% Generated by quadprog_note/experiment_qp_sweep.py -- do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\qpSweepFixedK}}{{{FIXED_K}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepRepeats}}{{{REPEATS}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepPerturb}}{{{PERTURB:g}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepNlo}}{{{int(ns[0])}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepNhi}}{{{int(ns[-1])}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepHitLo}}{{{hit[0] * 1e6:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepHitHi}}{{{hit[-1] * 1e6:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepSlopeLo}}{{{slopes[0]:.2f}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepSlopeHi}}{{{slopes[-1]:.2f}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepSpeedLo}}{{{fixed[0]['cold'] / fixed[0]['hit']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepSpeedHi}}{{{fixed[-1]['cold'] / fixed[-1]['hit']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpSweepKhi}}{{{growing[-1]['k']}}}\n")
        hit_rate = 100.0 * sum(r["hits"] for r in fixed) / (len(fixed) * REPEATS)
        fh.write(f"\\newcommand{{\\qpSweepHitRate}}{{{hit_rate:.0f}}}\n")
    print(f"Saved {path}")


def main() -> None:
    fixed, growing = run()
    figure(fixed, growing)
    emit(fixed, growing)
    print("\nDone.")


if __name__ == "__main__":
    main()
