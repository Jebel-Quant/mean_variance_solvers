"""End-to-end smoke tests for the four figure/table scripts.

Each script is a top-level program (no ``main``), so it is exercised by running
it in a subprocess against the current interpreter and asserting it exits 0 and
regenerates its declared outputs. These are marked ``slow`` (a couple of minutes
in total, dominated by the n=3000 scaling sweeps); run just the fast unit tests
with ``pytest -m 'not slow'`` (``make test-fast``).

The scripts read the committed ``data/*.parquet`` inputs, so no network or
``fast_minimum_variance`` is required — the study now runs on ``nncg`` and the
local ``baselines`` alone.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

# script -> a few output files it must (re)write on a successful run
SCRIPTS = {
    "experiment.py": ["tables/sp500_defs.tex", "tables/ftse_defs.tex"],
    "experiment_synthetic.py": [
        "graphs/minvar_scaling.pdf",
        "graphs/minvar_iters.pdf",
        "tables/frontier_def.tex",
    ],
    "experiment_oos.py": ["tables/oos_defs.tex", "graphs/minvar_oos.pdf"],
    "experiment_rmt.py": [
        "tables/rmt_solver_comparison.tex",
        "graphs/rmt_frontier.pdf",
    ],
}


@pytest.mark.slow
@pytest.mark.parametrize("script", sorted(SCRIPTS), ids=lambda s: s)
def test_experiment_runs_and_writes_outputs(script):
    outputs = [EXPERIMENT_DIR / rel for rel in SCRIPTS[script]]
    # Delete first so the assertion proves the run wrote them (not a stale file).
    for out in outputs:
        out.unlink(missing_ok=True)

    proc = subprocess.run(
        [sys.executable, script],
        cwd=EXPERIMENT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    for out in outputs:
        assert out.exists() and out.stat().st_size > 0, f"{script} did not write {out}"
