r"""What the memory layout costs, measured rather than reasoned about.

The derivation is indifferent to how the triangular factor is stored and to
whether a constraint normal is dense or a single scaled unit vector. The
implementation is not, and the two effects measured here are the reason.

**Packed against dense triangular storage.** The solver reads the leading
k-by-k block of R once per iteration, to solve for the dual direction. Held as
packed columns that block is the leading k(k+1)/2 entries -- contiguous at every
k, so a packed triangular solve reads it in place. Held in a dense (r, r) array
the same block is a *strided* view whose column stride is r rather than k, and a
LAPACK wrapper handed one must copy it first. The flop count is identical either
way; the wall-clock is not, and the gap is entirely memory traffic and library
boundary. This is the concrete form of a general point: in an array language the
layout decides which kernels are reachable, not merely how the bytes sit.

**Bound constraints as single nonzeros.** A bound is a column of C holding one
nonzero, and a box-constrained problem consists of nothing else. Detecting that
turns three per-iteration quantities from products into indexing. The detection is
per column rather than per matrix, which is what makes it useful on the mixed
shape a portfolio problem has -- one dense budget row among 2n bounds -- where an
all-or-nothing test would see the dense column and send everything down the
general path.

Both are measured against the shipped implementation. The triangular solve is
timed directly at the three call shapes; the structure detection is timed
end-to-end, by comparing a problem whose columns are single nonzeros against the
same problem perturbed so that they are not, which is the only A/B available given
that the package detects the shape for itself.

Usage:
    uv run python -m quadprog_note.experiment_qp_layout   # from experiment/

Outputs:
    graphs/quadprog_layout.pdf        triangular solve cost at the three shapes
    tables/quadprog_layout_defs.tex   headline numbers as \newcommand macros

NumPy + SciPy + Matplotlib + cvx-quadprog.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg.blas import dtpsv
from scipy.linalg.lapack import dtrtrs

from common.util.runner import SMOKE, output_dirs

from cvx.quadprog import solve_qp

HERE = Path(__file__).resolve().parents[1]
GRAPHS, TABLES = output_dirs(HERE)

mpl.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150})

# Working-set sizes for the triangular solve, with the enclosing dense array twice
# as wide -- the ratio a solver at half its constraint budget actually sees.
KS = [100, 400] if SMOKE else [50, 100, 200, 400, 800]
STRIDE_FACTOR = 2
SOLVE_REPEATS = 20 if SMOKE else 400

# End-to-end sizes for the structure comparison.
NS = [50, 100] if SMOKE else [50, 100, 200, 400]
SOLVE_N_REPEATS = 1 if SMOKE else 5


def best_of(fn, repeats):
    """Return the best wall time over `repeats` calls; best, because noise is one-sided."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def triangular_shapes(k, rng):
    """Time the same triangular solve at the three call shapes the solver could use."""
    r = k * STRIDE_FACTOR

    # Dense (r, r) upper triangular, of which the leading (k, k) block is live.
    dense = np.triu(rng.standard_normal((r, r)))
    dense[np.diag_indices(r)] += r  # keep it comfortably nonsingular
    rhs = rng.standard_normal(k)

    # The same leading block, packed by columns: entry (i, j) at j(j+1)/2 + i.
    packed = np.zeros(k * (k + 1) // 2)
    for j in range(k):
        packed[j * (j + 1) // 2 : j * (j + 1) // 2 + j + 1] = dense[: j + 1, j]

    strided = dense[:k, :k]                       # a view: column stride is r, not k
    contiguous = np.ascontiguousarray(strided)    # what the wrapper is forced to make

    times = {
        "strided view": best_of(
            lambda: dtrtrs(strided, rhs, lower=0), SOLVE_REPEATS),
        "contiguous copy": best_of(
            lambda: dtrtrs(contiguous, rhs, lower=0), SOLVE_REPEATS),
        "packed": best_of(
            lambda: dtpsv(k, packed, rhs.copy(), lower=0), SOLVE_REPEATS),
    }

    # The three must agree, or the comparison is between different computations.
    ref = dtrtrs(contiguous, rhs, lower=0)[0]
    got = dtpsv(k, packed, rhs.copy(), lower=0)
    assert np.allclose(ref, got, atol=1e-8), "packed and dense solves disagree"
    return times


def structured_problems(rng, n):
    """Return two problems that differ only in whether C's columns are single nonzeros.

    The first is a box, every column a scaled unit vector. The second perturbs each
    column with one extra small entry, so no column is a single nonzero and the
    detection cannot fire -- while the constraint set stays close enough that the
    solver does comparable work.
    """
    b_mat = rng.standard_normal((n, n))
    g = b_mat @ b_mat.T + n * np.eye(n)
    a = rng.standard_normal(n)
    xu = np.linalg.solve(g, a)

    unit = np.hstack([np.eye(n), -np.eye(n)])
    b = np.concatenate([xu - 0.5, -(xu + 0.5)])

    dense_cols = unit.copy()
    rows = (np.arange(2 * n) + 1) % n           # one extra entry per column
    dense_cols[rows, np.arange(2 * n)] += 1e-6  # small: same geometry, no detection
    return g, a, unit, dense_cols, b


def run():
    solve_rows = []
    for k in KS:
        rng = np.random.default_rng(k)
        times = triangular_shapes(k, rng)
        solve_rows.append({"k": k, **times})
        print(f"k = {k:<5} strided {times['strided view'] * 1e6:>8.1f} us   "
              f"copy {times['contiguous copy'] * 1e6:>8.1f} us   "
              f"packed {times['packed'] * 1e6:>8.1f} us   "
              f"ratio {times['strided view'] / times['packed']:>5.1f}x")

    struct_rows = []
    for n in NS:
        rng = np.random.default_rng(n + 11)
        g, a, unit, dense_cols, b = structured_problems(rng, n)
        t_unit = best_of(lambda: solve_qp(g, a, unit, b, 0), SOLVE_N_REPEATS)
        t_dense = best_of(lambda: solve_qp(g, a, dense_cols, b, 0), SOLVE_N_REPEATS)
        # Confirm the two really are the same problem to the eye of the solver.
        gap = float(np.max(np.abs(solve_qp(g, a, unit, b, 0).x
                                  - solve_qp(g, a, dense_cols, b, 0).x)))
        struct_rows.append({"n": n, "unit": t_unit, "dense": t_dense, "gap": gap})
        print(f"n = {n:<5} unit columns {t_unit * 1e3:>8.2f} ms   "
              f"general {t_dense * 1e3:>8.2f} ms   "
              f"ratio {t_dense / t_unit:>5.2f}x   |dx| {gap:.2e}")

    return solve_rows, struct_rows


def figure(solve_rows) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ks = [row["k"] for row in solve_rows]
    for label, colour, marker in (("strided view", "#d62728", "o"),
                                  ("contiguous copy", "#ff7f0e", "s"),
                                  ("packed", "#1f77b4", "^")):
        ax.plot(ks, [row[label] * 1e6 for row in solve_rows],
                color=colour, marker=marker, markersize=4, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("working-set size $k$")
    ax.set_ylabel("per solve ($\\mu$s)")
    ax.set_title("One triangular solve, three call shapes")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(framealpha=0.9)
    fig.tight_layout(pad=0.6)
    fig.savefig(GRAPHS / "quadprog_layout.pdf", bbox_inches="tight")
    fig.savefig(GRAPHS / "quadprog_layout.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved {GRAPHS / 'quadprog_layout.pdf'}")


def emit(solve_rows, struct_rows) -> None:
    biggest = solve_rows[-1]
    ratio_packed = biggest["strided view"] / biggest["packed"]
    ratio_copy = biggest["strided view"] / biggest["contiguous copy"]
    struct_ratio = max(row["dense"] / row["unit"] for row in struct_rows)
    struct_at = max(struct_rows, key=lambda row: row["dense"] / row["unit"])

    path = TABLES / "quadprog_layout_defs.tex"
    with open(path, "w") as fh:
        fh.write("% Generated by quadprog_note/experiment_qp_layout.py -- do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\qpLayKhi}}{{{biggest['k']}}}\n")
        fh.write(f"\\newcommand{{\\qpLayStrided}}{{{biggest['strided view'] * 1e6:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpLayCopy}}{{{biggest['contiguous copy'] * 1e6:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpLayPacked}}{{{biggest['packed'] * 1e6:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpLayRatio}}{{{ratio_packed:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpLayRatioCopy}}{{{ratio_copy:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpLayStride}}{{{STRIDE_FACTOR}}}\n")
        fh.write(f"\\newcommand{{\\qpLayKs}}{{{', '.join(str(k) for k in KS)}}}\n")
        fh.write(f"\\newcommand{{\\qpStructRatio}}{{{struct_ratio:.2f}}}\n")
        fh.write(f"\\newcommand{{\\qpStructN}}{{{struct_at['n']}}}\n")
        fh.write(f"\\newcommand{{\\qpStructNs}}{{{', '.join(str(n) for n in NS)}}}\n")
    print(f"Saved {path}")


def main() -> None:
    solve_rows, struct_rows = run()
    figure(solve_rows)
    emit(solve_rows, struct_rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
