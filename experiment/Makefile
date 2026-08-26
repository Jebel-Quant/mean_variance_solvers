# Regenerate the figures (graphs/) and tables (tables/) that the papers
# include. The scripts are nested by paper into importable packages — cg/
# (Shrinkage-as-Preconditioning), rmt/ (Eigenvalue-Cleaning), nncg_note/
# (Non-Negative Conjugate Gradients), quadprog_note/ (the dual active-set
# note, which verifies its own identities) — over a shared common/ (baselines,
# simulate, util, data fetchers). They are run as modules from this directory
# (`uv run python -m <pkg>.<module>`) against the .venv that `make venv` builds
# from pyproject.toml, so imports resolve across packages. The committed
# data/*.parquet inputs are produced by the fetch_* scripts (`make data`).

.DEFAULT_GOAL := help

.PHONY: help venv data figures test test-fast clean

help:  ## Show this overview of available commands
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk -F':.*## ' '{ printf "  %-10s %s\n", $$1, $$2 }'

venv:  ## Build a persistent .venv from pyproject.toml (uv sync)
	uv sync
	@echo "Activate with: source .venv/bin/activate"

test:  ## Run the full test suite with coverage (unit + end-to-end; fails under 100%)
	uv run pytest --cov --cov-report=term-missing

test-fast:  ## Run only the fast unit tests, with coverage (skips the slow experiment runs)
	uv run pytest -m "not slow" --cov --cov-report=term-missing

data:  ## Download the raw S&P 500 / FTSE 100 return data into data/
	uv run python -m common.fetch_sp500
	uv run python -m common.fetch_ftse100

figures:  ## Regenerate every graph and table for every paper
	uv run python -m cg.experiment
	uv run python -m cg.experiment_synthetic
	uv run python -m cg.experiment_oos
	uv run python -m rmt.experiment_rmt
	uv run python -m nncg_note.experiment_nncg
	uv run python -m nncg_note.experiment_nncg_certificate
	uv run python -m nncg_note.experiment_nncg_precond
	uv run python -m nncg_note.experiment_nncg_frontier
	uv run python -m nncg_note.experiment_nncg_bench
	uv run python -m nncg_note.experiment_nncg_regu
	uv run python -m nncg_note.experiment_nncg_deblur
	uv run python -m nncg_note.experiment_nncg_hyperspectral_library
	uv run python -m quadprog_note.experiment_qp_identities
	uv run python -m quadprog_note.experiment_qp_pdas
	uv run python -m quadprog_note.experiment_qp_sweep
	uv run python -m quadprog_note.experiment_qp_compare
	uv run python -m quadprog_note.experiment_qp_portfolio

clean:  ## Remove the persistent virtual environment
	rm -rf .venv uv.lock
