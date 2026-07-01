# Regenerate the figures (graphs/) and tables (tables/) that both papers
# include. Each script is a self-contained PEP 723 program: `uv run` resolves
# its pinned dependencies (including fast-minimum-variance from its git repo)
# into a throwaway environment, so no separate setup is needed. The committed
# data/*.parquet inputs are produced by the fetch_* scripts (`make data`).

.DEFAULT_GOAL := help

.PHONY: help data figures

help:  ## Show this overview of available commands
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk -F':.*## ' '{ printf "  %-10s %s\n", $$1, $$2 }'

data:  ## Download the raw S&P 500 / FTSE 100 return data into data/
	uv run fetch_sp500.py
	uv run fetch_ftse100.py

figures:  ## Regenerate every graph and table for both papers
	uv run experiment.py
	uv run experiment_synthetic.py
	uv run experiment_oos.py
	uv run experiment_rmt.py
