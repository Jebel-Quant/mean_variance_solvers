"""Tests for the timing/formatting helpers in ``util/runner.py``."""

from __future__ import annotations

from util.runner import (
    _fmt_time,
    _format_benchmark_rows,
    _format_frontier_rows,
    run_timed,
    write_frontier_def,
    write_table_defs,
)


def test_run_timed_returns_result_and_best_time():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result, best = run_timed(fn, repeats=3)
    assert result == "ok"
    assert len(calls) == 3  # ran `repeats` times
    assert best >= 0.0


def test_fmt_time_bands():
    assert _fmt_time(42.0) == "42.0"  # >= 10 -> one decimal
    assert _fmt_time(1.2345) == "1.234"  # >= 0.1 -> three decimals
    assert _fmt_time(0.01234) == "0.0123"  # small -> four decimals


def test_format_benchmark_rows_speedup_and_order():
    results = {
        "ref": {"time_s": 1.0, "outer": None, "inner": 10},
        "fast": {"time_s": 0.25, "outer": 3, "inner": None},
    }
    rows = _format_benchmark_rows(results, ref_key="ref", method_order=["fast", "ref"])
    lines = rows.strip().splitlines()
    assert lines[0].startswith("fast")  # method_order respected
    assert "4.0x" in lines[0]  # 1.0 / 0.25
    assert "1.0x" in lines[1]
    assert "10" in lines[1]  # inner shown when outer is None


def test_format_benchmark_rows_footnote_marker():
    results = {"m": {"time_s": 1.0, "outer": 1, "inner": None}}
    rows = _format_benchmark_rows(results, ref_key="m", footnote_methods={"m"})
    assert "$^\\dagger$" in rows


def test_format_frontier_rows_warm_and_cold():
    rows = [
        {"label": "cold-only", "cold": 2.0, "warm": None},
        {"label": "both", "cold": 2.0, "warm": 0.5},
    ]
    out = _format_frontier_rows(rows, n_pts=10)
    lines = out.strip().splitlines()
    assert "multicolumn" in lines[0]  # cold-only -> "--" for warm columns
    assert "multicolumn" not in lines[1]


def test_write_table_defs(tmp_path):
    path = tmp_path / "defs.tex"
    write_table_defs(
        path,
        [
            {
                "macro_name": "myMacro",
                "results": {"m": {"time_s": 1.0, "outer": 1, "inner": None}},
                "ref_key": "m",
            }
        ],
    )
    text = path.read_text()
    assert "\\def\\myMacro{%" in text
    assert text.rstrip().endswith("}")


def test_write_frontier_def(tmp_path):
    path = tmp_path / "frontier.tex"
    write_frontier_def(path, "dataFrontier", [{"label": "x", "cold": 1.0, "warm": 0.5}], n_pts=5)
    text = path.read_text()
    assert "\\def\\dataFrontier{%" in text
