# Build the short note ("Guessing the active set: a rank obstruction on general
# convex quadratic programs"), the negative result extracted from the quadprog
# working note in ../quadprog. That note derives the Goldfarb/Idnani method in
# full and documents the cvx-quadprog package; this one carries only the claim
# about primal-dual active-set exchange on general constraints, the certificate
# that replaces the guarantee, and the measurement of what the exposure costs.
#
# It cites the shared bibliography that lives with the minimum-variance paper
# (../matrix_free/bib/refs.bib), and takes its one figure and its generated
# numbers from ../experiment via the graphs/ and tables/ symlinks, which the
# repository-root `make figures` regenerates. All shared build logic lives in
# ../common.mk; only the per-paper variables differ.

DOC := quadprog_neg_paper
BIB := ../matrix_free/bib/refs.bib

include ../common.mk
