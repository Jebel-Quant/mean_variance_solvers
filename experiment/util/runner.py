"""Timing and result-printing helpers for solver benchmarks."""

from __future__ import annotations

import time
from pathlib import Path


def run_timed(fn, repeats=3):
    """Return (result, best_wall_time_s) over `repeats` calls."""
    best = float("inf")
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return result, best


def print_table(label, results, ref_key="cvxpy") -> None:
    """Print a benchmark table with speedup relative to ref_key.

    Each entry in results maps to a dict with keys:
      time_s   float
      outer    int | None   (active-set outer steps; None if no outer loop)
      inner    int | None   (inner solver iterations; None for direct solvers)
    """
    ref = results[ref_key]["time_s"]
    print(f"\n{label}")
    print(f"{'Method':<30} {'Time (s)':>10} {'Outer':>7} {'Inner':>8} {'Speedup':>10}")
    print("-" * 70)
    for key, v in results.items():
        outer_str = str(v["outer"]) if v.get("outer") is not None else "--"
        inner_str = str(v["inner"]) if v.get("inner") is not None else "--"
        print(f"{key:<30} {v['time_s']:>10.4f} {outer_str:>7} {inner_str:>8} {ref / v['time_s']:>9.1f}x")


def _fmt_time(t) -> str:
    """Format a wall-clock time in seconds for a LaTeX table."""
    if t >= 10:
        return f"{t:.1f}"
    if t >= 0.1:
        return f"{t:.3f}"
    return f"{t:.4f}"


def _format_benchmark_rows(results, ref_key, footnote_methods=None, method_order=None):
    r"""Return formatted tabular row strings (no surrounding \\def)."""
    ref = results[ref_key]["time_s"]
    order = method_order if method_order is not None else list(results)
    lines = []
    for method in order:
        if method not in results:
            continue
        v = results[method]
        label = method
        if footnote_methods and method in footnote_methods:
            label = f"{method}$^\\dagger$"
        iters = v.get("inner") if v.get("inner") is not None else v.get("outer")
        iters_str = str(iters) if iters is not None else "--"
        speedup = ref / v["time_s"]
        lines.append(f"{label:<35} & {_fmt_time(v['time_s']):>8} & {iters_str:>6} & {speedup:>6.1f}x \\\\\n")
    return "".join(lines)


def _format_frontier_rows(rows, n_pts):
    r"""Return formatted frontier sweep row strings (no surrounding \\def)."""
    lines = []
    for row in rows:
        label = row["label"]
        cold_ms = row["cold"] / n_pts * 1000
        if row.get("warm") is not None:
            warm_ms = row["warm"] / n_pts * 1000
            lines.append(
                f"{label:<32} & {_fmt_time(row['cold']):>6} & {cold_ms:>5.1f}"
                f" & {_fmt_time(row['warm']):>6} & {warm_ms:>5.1f} \\\\\n"
            )
        else:
            lines.append(
                f"{label:<32} & {_fmt_time(row['cold']):>6} & {cold_ms:>5.1f} & \\multicolumn{{2}}{{c}}{{--}} \\\\\n"
            )
    return "".join(lines)


def write_table_defs(path, panels) -> None:
    r"""Write \\def macros (one per panel) to a .tex file for use inside tabular.

    The generated file must be \\input-ted OUTSIDE any tabular environment so that
    the macros are defined before the table is typeset.  Inside the tabular, call
    each macro by name — it expands inline with no file boundary, which means
    booktabs rules (\\midrule, \\bottomrule) and panel headers (\\multicolumn) can
    safely live in the static .tex file right before and after each macro call.

    panels: list of dicts with keys:
      macro_name       str   LaTeX control sequence name (without leading \\)
      results          dict  solver name -> {"time_s", "outer", "inner"}
      ref_key          str
      footnote_methods set | None
      method_order     list | None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for panel in panels:
        rows = _format_benchmark_rows(
            panel["results"],
            panel["ref_key"],
            panel.get("footnote_methods"),
            panel.get("method_order"),
        )
        chunks.append(f"\\def\\{panel['macro_name']}{{%\n{rows}}}\n")
    path.write_text("".join(chunks))


def write_frontier_def(path, macro_name, rows, n_pts) -> None:
    r"""Write a single \\def macro for frontier sweep rows.

    Same rationale as write_table_defs: \\input outside tabular, use macro inside.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _format_frontier_rows(rows, n_pts)
    path.write_text(f"\\def\\{macro_name}{{%\n{content}}}\n")
