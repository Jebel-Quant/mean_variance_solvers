# Build the quadprog paper ("The Mathematics of a Dual Active-Set Quadratic
# Programming Solver"), a derivation of the Goldfarb/Idnani method as the
# cvx-quadprog package (https://github.com/Jebel-Quant/quadprog) implements it.
# The companion implementation paper lives in that repository under
# docs/paper/; this one carries the mathematics it takes as read.
#
# It cites the shared bibliography that lives with the minimum-variance paper
# (../matrix_free/bib/refs.bib). No figures or tables: every claim here is a
# derivation, so there is nothing for ../experiment to regenerate. All shared
# build logic lives in ../common.mk; only the per-paper variables differ. Run
# from this directory, e.g. via the repository-root `make compile`.

DOC := quadprog_paper
BIB := ../matrix_free/bib/refs.bib

include ../common.mk
