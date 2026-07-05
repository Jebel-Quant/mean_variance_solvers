"""End-to-end smoke tests for the four figure/table scripts.

Each script is a top-level program (no ``main``), so it is exercised by running
it in a subprocess and asserting it exits 0 and writes its declared outputs.

The runs go through in smoke mode (``EXPERIMENT_SMOKE=1``): every sweep and the
scaling ``n`` list are shrunk and timing repeats capped to one, so the suite
exercises each code path in seconds rather than reproducing the paper-scale
runs. Outputs are redirected to a scratch directory (``EXPERIMENT_OUT``) so a
smoke run never overwrites the committed full-resolution figures and tables.

The scripts read the committed ``data/*.parquet`` inputs, so no network or
``fast_minimum_variance`` is required — the study now runs on ``nncg`` and the
local ``baselines`` alone.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

# module (run as `python -m <module>` from experiment/) -> output files it must
# (re)write on a successful run
SCRIPTS = {
    "cg.experiment": ["tables/sp500_defs.tex", "tables/ftse_defs.tex"],
    "cg.experiment_synthetic": [
        "graphs/cg_scaling.pdf",
        "graphs/cg_iters.pdf",
        "tables/frontier_def.tex",
    ],
    "cg.experiment_oos": ["tables/oos_defs.tex", "graphs/cg_oos.pdf"],
    "rmt.experiment_rmt": [
        "tables/rmt_solver_comparison.tex",
        "tables/rmt_oos.tex",
        "graphs/rmt_frontier.pdf",
    ],
    # companion "Non-Negative Conjugate Gradients" note (nncg package + nncg_note.problems)
    "nncg_note.experiment_nncg": ["graphs/nncg_kappa.pdf", "tables/nncg_synthetic.tex"],
    "nncg_note.experiment_nncg_bench": ["graphs/nncg_bench.pdf", "tables/nncg_bench_defs.tex"],
    "nncg_note.experiment_nncg_regu": ["tables/nncg_regu.tex", "tables/nncg_regu_defs.tex"],
    "nncg_note.experiment_nncg_deblur": ["graphs/nncg_deblur.pdf", "tables/nncg_deblur_defs.tex"],
    "nncg_note.experiment_nncg_hyperspectral": [
        "graphs/nncg_hyperspectral.pdf",
        "tables/nncg_hyperspectral_defs.tex",
    ],
}


@pytest.mark.slow
@pytest.mark.parametrize("module", sorted(SCRIPTS), ids=lambda s: s)
def test_experiment_runs_and_writes_outputs(module, tmp_path):
    env = {**os.environ, "EXPERIMENT_SMOKE": "1", "EXPERIMENT_OUT": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=EXPERIMENT_DIR,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"{module} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    for rel in SCRIPTS[module]:
        out = tmp_path / rel
        assert out.exists() and out.stat().st_size > 0, f"{module} did not write {out}"
