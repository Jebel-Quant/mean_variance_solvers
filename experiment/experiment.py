"""Real-data benchmark for the paper.

    "Shrinkage as Preconditioning: Matrix-Free Methods for
     Long-Only Portfolio Optimization"

Usage:
    uv run experiment.py          # from the experiment/ directory

Inputs:
    data/sp500_pct_returns.parquet   — S&P 500 daily pct returns
    data/ftse100_pct_returns.parquet — FTSE 100 daily pct returns
    Fetch with:  uv run fetch_sp500.py / uv run fetch_ftse100.py

Output (stdout):
    For each dataset: solver panels (no shrinkage, LW alpha=0.5, LW oracle)
    across the matrix-free and general-purpose solvers; timings,
    iterations, and speedup vs CVXPY.

Hardware used in the paper: Apple M4 Pro, 14-core CPU, 48 GB RAM.
Software: Python 3.12, NumPy 2.4, SciPy 1.17, CVXPY 1.8.2, Clarabel 0.11.1.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "scikit-learn",
#     "pyarrow",
#     "numpy",
#     "nncg==0.2.2",
#     "cvx-linalg",
#     "clarabel",
#     "osqp",
#     "scipy",
#     "cvxpy",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from util.runner import SMOKE, _fmt_time, output_dirs, print_table, run_timed, write_table_defs

from minvar import (
    MinVarProblem,
    lw_alpha_and_target,
    oas_alpha_and_target,
)

HERE = Path(__file__).parent
_GRAPHS_BASE, TABLES = output_dirs(HERE)

# Solver rows written to the paper tables (first-order methods are treated in the
# nncg companion paper and excluded here).
_TABLE_METHODS_ALL = [
    "cvxpy (Clarabel)",
    "cvxpy (OSQP)",
    "Clarabel (direct API)",
    "OSQP (direct API)",
    "KKT (Cholesky)",
    "CG (SPD)",
]
_TABLE_METHODS_KEY = [
    "cvxpy (Clarabel)",
    "KKT (Cholesky)",
    "CG (SPD)",
]
_FOOTNOTE = set()

DATASETS = {
    "sp500": HERE / "data/sp500_pct_returns.parquet",
    "ftse": HERE / "data/ftse100_pct_returns.parquet",
}

SOLVERS_ALL = [
    ("cvxpy (Clarabel)", lambda p: p.solve_cvxpy(), False),
    ("cvxpy (OSQP)", lambda p: p.solve_cvxpy(backend="osqp"), False),
    ("Clarabel (direct API)", lambda p: p.solve_clarabel(), False),
    ("OSQP (direct API)", lambda p: p.solve_osqp(), False),
    ("KKT (Cholesky)", lambda p: p.solve_kkt(), True),
    ("CG (SPD)", lambda p: p.solve_cg(), False),
]
SOLVERS_KEY = [
    ("cvxpy (Clarabel)", lambda p: p.solve_cvxpy(), False),
    ("KKT (Cholesky)", lambda p: p.solve_kkt(), True),
    ("CG (SPD)", lambda p: p.solve_cg(), False),
]


def _make_entry(prob, fn, is_kkt=False):
    """Return {"time_s", "outer", "inner"} for one solver on one problem."""
    raw, t = run_timed(lambda: fn(prob))
    if len(raw) == 3:  # solve_cg -> (w, outer, inner)
        _, outer, inner = raw
    elif is_kkt:  # solve_kkt -> (w, outer_steps)
        _, outer = raw
        inner = None
    else:  # cvxpy / clarabel / osqp -> (w, iters)
        _, inner = raw
        outer = None
    return {"time_s": t, "outer": outer, "inner": inner}


for dataset_name, data_file in DATASETS.items():
    GRAPHS = _GRAPHS_BASE / dataset_name
    GRAPHS.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"{dataset_name.upper()} benchmark  (long-only minimum variance)")
    print("=" * 70)

    df = pd.read_parquet(data_file)
    R = df.to_numpy()
    R = R - R.mean(axis=0)
    if SMOKE:  # exercise every panel on a small universe rather than the full one
        R = R[:, :60]
    _T, N = R.shape

    alpha_lw, target = lw_alpha_and_target(R)
    alpha_oas, _ = oas_alpha_and_target(R)
    alpha_hard = 0.5

    print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"n={N}, T={_T}, n/T={N / _T:.3f}")
    print(f"LW  oracle alpha = {alpha_lw:.4f}")
    print(f"OAS oracle alpha = {alpha_oas:.4f}")
    print(f"Demonstrational alpha = {alpha_hard}")

    results_no_shrink = {}
    results_lw_oracle = {}
    results_oas_oracle = {}
    results_lw = {}

    prob_no_shrink = MinVarProblem(R)
    prob_lw_ora = MinVarProblem(R, alpha=alpha_lw, target=target)
    prob_oas_ora = MinVarProblem(R, alpha=alpha_oas, target=target)
    prob_lw = MinVarProblem(R, alpha=alpha_hard, target=target)

    for sname, fn, is_kkt in SOLVERS_ALL:
        # KKT (dense Cholesky) requires an SPD system; without shrinkage the
        # covariance can be singular (it is for FTSE), so it is reported only
        # for the alpha>0 panels where Theorem 4.1 guarantees SPD.
        if sname != "KKT (Cholesky)":
            results_no_shrink[sname] = _make_entry(prob_no_shrink, fn, is_kkt)
        results_lw[sname] = _make_entry(prob_lw, fn, is_kkt)

    for sname, fn, is_kkt in SOLVERS_KEY:
        results_lw_oracle[sname] = _make_entry(prob_lw_ora, fn, is_kkt)
        results_oas_oracle[sname] = _make_entry(prob_oas_ora, fn, is_kkt)

    print_table("Without shrinkage", results_no_shrink, ref_key="cvxpy (Clarabel)")
    print_table(f"Oracle LW (alpha={alpha_lw:.4f})", results_lw_oracle, ref_key="cvxpy (Clarabel)")
    print_table(f"Oracle OAS (alpha={alpha_oas:.4f})", results_oas_oracle, ref_key="cvxpy (Clarabel)")
    print_table(f"Demonstrational LW (alpha={alpha_hard})", results_lw, ref_key="cvxpy (Clarabel)")

    ref_key = "cvxpy (Clarabel)"
    if dataset_name == "sp500":
        write_table_defs(
            TABLES / "sp500_defs.tex",
            [
                {
                    "macro_name": "dataSpNoshrink",
                    "results": results_no_shrink,
                    "ref_key": ref_key,
                    "footnote_methods": _FOOTNOTE,
                    "method_order": _TABLE_METHODS_ALL,
                },
                {
                    "macro_name": "dataSpLwHalf",
                    "results": results_lw,
                    "ref_key": ref_key,
                    "footnote_methods": _FOOTNOTE,
                    "method_order": _TABLE_METHODS_ALL,
                },
                {
                    "macro_name": "dataSpLwOracle",
                    "results": results_lw_oracle,
                    "ref_key": ref_key,
                    "footnote_methods": _FOOTNOTE,
                    "method_order": _TABLE_METHODS_KEY,
                },
            ],
        )
        print("  → wrote experiment/tables/sp500_defs.tex")

        # Balance-system panel: the production package now accepts (B, c), so the
        # p in {1, 4, 8} sleeve systems of Section (balance) run through the same
        # matrix-free CG / dense-KKT solvers as the budget benchmark above, with
        # timings directly comparable to the budget rows.  Sleeve construction
        # matches experiment_balance.py (seed 0, proportional shares).
        print("\nBalance systems (sleeves, LW alpha=0.5)")
        print(f"{'solver':<16} {'p':>3} {'time(s)':>9} {'iters':>7} {'speedup':>9}")
        print("-" * 50)
        sleeve_rng = np.random.default_rng(0)

        def _sleeve(p, n=N, rng=sleeve_rng):
            """Partition the universe into p sleeves each holding its budget share."""
            if p == 1:
                return np.ones((1, n)), np.array([1.0])
            groups = np.array_split(rng.permutation(n), p)
            b_eq = np.zeros((p, n))
            c_eq = np.zeros(p)
            for g, idx in enumerate(groups):
                b_eq[g, idx] = 1.0
                c_eq[g] = len(idx) / n
            return b_eq, c_eq

        sleeve_lines = []
        for solver_label, solver_fn, is_kkt in (
            ("CG (SPD)", lambda pr: pr.solve_cg(), False),
            ("KKT (Cholesky)", lambda pr: pr.solve_kkt(), True),
        ):
            for p in (1, 4, 8):
                b_eq, c_eq = _sleeve(p)
                prob = MinVarProblem(R, alpha=alpha_hard, target=target, B=b_eq, c=c_eq)
                (_ref, _), t_ref = run_timed(lambda pr=prob: pr.solve_cvxpy(project=False))
                entry = _make_entry(prob, solver_fn, is_kkt)
                iters = entry["inner"] if entry["inner"] is not None else entry["outer"]
                speedup = t_ref / entry["time_s"]
                tag = "(budget)" if p == 1 else "(sleeves)"
                label = f"{solver_label}, $p = {p}$ {tag}"
                sleeve_lines.append(
                    f"{label:<32} & {_fmt_time(entry['time_s']):>8} & {iters:>6} & {speedup:>6.1f}x \\\\\n"
                )
                print(f"{solver_label:<16} {p:>3} {entry['time_s']:>9.4f} {iters:>7} {speedup:>8.1f}x")

        (TABLES / "sp500_sleeves_def.tex").write_text(f"\\def\\dataSpSleeves{{%\n{''.join(sleeve_lines)}}}\n")
        print("  → wrote experiment/tables/sp500_sleeves_def.tex")
    else:
        write_table_defs(
            TABLES / "ftse_defs.tex",
            [
                {
                    "macro_name": "dataFtseNoshrink",
                    "results": results_no_shrink,
                    "ref_key": ref_key,
                    "footnote_methods": _FOOTNOTE,
                    "method_order": _TABLE_METHODS_ALL,
                },
                {
                    "macro_name": "dataFtseLwHalf",
                    "results": results_lw,
                    "ref_key": ref_key,
                    "footnote_methods": _FOOTNOTE,
                    "method_order": _TABLE_METHODS_ALL,
                },
            ],
        )
        print("  → wrote experiment/tables/ftse_defs.tex")
