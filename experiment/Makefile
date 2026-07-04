# Regenerate the figures (graphs/) and tables (tables/) that both papers
# include. Each script is a self-contained PEP 723 program: `uv run` resolves
# its pinned dependencies (including fast-minimum-variance) into a throwaway
# environment, so `make data`/`make figures` need no setup. For interactive
# work (a REPL, an IDE, running a script under a debugger) `make venv` builds
# one persistent .venv from pyproject.toml instead. The committed
# data/*.parquet inputs are produced by the fetch_* scripts (`make data`).

.DEFAULT_GOAL := help

.PHONY: help venv data figures test test-fast clean

help:  ## Show this overview of available commands
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk -F':.*## ' '{ printf "  %-10s %s\n", $$1, $$2 }'

venv:  ## Build a persistent .venv from pyproject.toml (uv sync)
	uv sync
	@echo "Activate with: source .venv/bin/activate"

test:  ## Run the full test suite (unit + end-to-end experiment runs; ~2 min)
	uv run pytest

test-fast:  ## Run only the fast unit tests (skips the slow experiment runs)
	uv run pytest -m "not slow"

data:  ## Download the raw S&P 500 / FTSE 100 return data into data/
	uv run fetch_sp500.py
	uv run fetch_ftse100.py

figures:  ## Regenerate every graph and table for both papers
	uv run experiment.py
	uv run experiment_synthetic.py
	uv run experiment_oos.py
	uv run experiment_rmt.py

clean:  ## Remove the persistent virtual environment
	rm -rf .venv uv.lock
