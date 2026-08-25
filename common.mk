# Shared build logic for every paper. A paper's Makefile sets the handful of
# per-paper variables below and then `include ../common.mk`. Nothing here is
# paper-specific — edit a target once and every paper picks it up.
#
# Required / optional variables (set before the include):
#   DOC     basename of the main .tex / .pdf (e.g. cg_paper, rmt_paper).
#   BIB     path to the BibTeX database the paper cites, relative to the paper
#           folder (default bib/refs.bib). Only used to assemble the arXiv
#           tarball; the .tex's own \bibliography path drives compilation.
#   VENDOR  space-separated class/style/bst files to bundle into the arXiv
#           tarball (default: none — arXiv's TeX Live already ships siam.bst,
#           the plain style, etc.).
#
# Section files live in sections/ (the main .tex \inputs sections/s*.tex).
# Figures (graphs/) and tables (tables/) are symlinks into ../experiment, which
# the experiment/ Python package regenerates — see the repository-root
# `make figures`.

LATEX     := pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error
BIB       ?= bib/refs.bib
VENDOR    ?=

# Where `stage` drops this paper's PDF. Overridden by the repository-root
# `pdfs` target with an absolute path, because make -C has already changed
# directory by the time this is expanded.
OUTDIR    ?= ../out

.DEFAULT_GOAL := help

.PHONY: help compile stage clean arxiv

# Self-documenting help: lists every target with a `## description` comment.
help:  ## Show this overview of available commands
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| sort \
		| awk -F':.*## ' '{ printf "  %-10s %s\n", $$1, $$2 }'
	@echo ""

compile: $(DOC).pdf  ## Build the paper PDF

# Publish this paper's PDF by name, so that a caller collecting every paper's
# output does not have to know what the name is. `DOC` is the one place a
# paper's basename is written down, and this is what lets the CI publish step
# discover papers instead of carrying a hardcoded list that a new paper silently
# falls out of -- which is exactly what happened when quadprog/ was added.
stage: $(DOC).pdf  ## Copy the paper PDF into $(OUTDIR)
	@mkdir -p $(OUTDIR)
	cp $(DOC).pdf $(OUTDIR)/

$(DOC).pdf: $(DOC).tex $(wildcard sections/s*.tex) $(BIB) $(wildcard tables/*.tex)
	$(LATEX) $(LATEXOPTS) $(DOC).tex
	bibtex $(DOC)
	$(LATEX) $(LATEXOPTS) $(DOC).tex
	$(LATEX) $(LATEXOPTS) $(DOC).tex
	$(MAKE) clean

# Assemble a self-contained source tree and tar it. When a paper has figures or
# tables, graphs/ and tables/ are symlinks into ../experiment; the globs below
# resolve through them and copy the real files, so the tarball carries no
# symlinks. A paper with neither (e.g. nncg) has no such symlinks, so those
# copies are guarded and simply skipped. The bibliography is flattened
# to bib/refs.bib and the \bibliography path rewritten to match (rmt cites the
# shared ../matrix_free/bib/refs). This does not run LaTeX: arXiv compiles server-side,
# so the .bib travels along and arXiv runs BibTeX itself.
ARXIV_DIR     := arxiv-src
ARXIV_TARBALL := $(DOC).tar.gz

arxiv: $(ARXIV_TARBALL)  ## Package the self-contained arXiv source tarball

$(ARXIV_TARBALL): $(DOC).tex $(wildcard sections/s*.tex) $(BIB) $(VENDOR) \
                  $(wildcard graphs/*.pdf) $(wildcard tables/*.tex)
	rm -rf $(ARXIV_DIR)
	mkdir -p $(ARXIV_DIR)/sections $(ARXIV_DIR)/bib
	cp $(DOC).tex $(ARXIV_DIR)/
	cp sections/*.tex $(ARXIV_DIR)/sections/
	cp $(BIB) $(ARXIV_DIR)/bib/refs.bib
	@if ls graphs/*.pdf >/dev/null 2>&1; then \
		mkdir -p $(ARXIV_DIR)/graphs && cp graphs/*.pdf $(ARXIV_DIR)/graphs/; \
	fi
	@if ls tables/*.tex >/dev/null 2>&1; then \
		mkdir -p $(ARXIV_DIR)/tables && cp tables/*.tex $(ARXIV_DIR)/tables/; \
	fi
	@[ -z "$(VENDOR)" ] || cp $(VENDOR) $(ARXIV_DIR)/
	sed -i.bak 's|\\bibliography{[^}]*}|\\bibliography{bib/refs}|' $(ARXIV_DIR)/$(DOC).tex
	rm -f $(ARXIV_DIR)/$(DOC).tex.bak
	tar czf $(ARXIV_TARBALL) -C $(ARXIV_DIR) .
	tar tzf $(ARXIV_TARBALL)

clean:  ## Remove LaTeX build artifacts (keeps the PDF)
	rm -rf $(ARXIV_DIR) $(ARXIV_TARBALL)
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
		$(DOC).bbl $(DOC).blg $(DOC).fls $(DOC).fdb_latexmk $(DOC).thm
