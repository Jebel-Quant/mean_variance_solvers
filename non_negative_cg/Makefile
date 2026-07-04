# Build the non-negativity paper ("Non-Negative Conjugate Gradients"). It cites
# the shared bibliography that lives with the minimum-variance paper
# (../matrix_free/bib/refs.bib). Its synthetic study now lives in the shared
# ../experiment package alongside the finance experiments (the experiment_nncg*
# scripts); graphs/ and tables/ are symlinks into ../experiment, and the
# repository-root `make figures` regenerates them. All shared build logic lives
# in ../common.mk; only the per-paper variables differ. Run from this
# directory, e.g. via the repository-root `make compile`.

DOC := nncg_paper
BIB := ../matrix_free/bib/refs.bib

include ../common.mk
