# Build the non-negativity paper ("Non-Negative Conjugate Gradients"). It cites
# the shared bibliography that lives with the minimum-variance paper
# (../matrix_free/bib/refs.bib). Unlike the two finance papers, this note owns a
# self-contained synthetic study under experiment/ that writes its figures to
# graphs/ and its tables to tables/ (no ../experiment symlinks); `make figures`
# regenerates them. All shared build logic lives in ../common.mk; only the
# per-paper variables differ. Run from this directory, e.g. via the
# repository-root `make compile`.

DOC := nncg_paper
BIB := ../matrix_free/bib/refs.bib

include ../common.mk

.PHONY: figures

figures:  ## Regenerate the synthetic-study figures (graphs/) and tables (tables/)
	cd experiment && uv run experiment_nncg.py
	cd experiment && uv run experiment_nncg_bench.py
