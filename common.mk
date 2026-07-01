# Shared build logic for every paper. A paper's Makefile sets the handful of
# per-paper variables below and then `include ../common.mk`. Nothing here is
# paper-specific — edit a target once and every paper picks it up.
#
# Required / optional variables (set before the include):
#   DOC     basename of the main .tex / .pdf (e.g. minvar_paper, rmt_paper).
#   BIB     path to the BibTeX database the paper cites, relative to the paper
#           folder (default bib/refs.bib). Only used to assemble the arXiv
#           tarball; the .tex's own \bibliography path drives compilation.
#   VENDOR  space-separated class/style/bst files to bundle into the arXiv
#           tarball (default: none — arXiv's TeX Live already ships siam.bst,
#           the plain style, etc.).
#
# Section files are s*.tex at the paper root. Figures (graphs/) and tables
# (tables/) are symlinks into ../experiment, which the experiment/ Python
# package regenerates — see the repository-root `make figures`.

LATEX     := pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error
BIB       ?= bib/refs.bib
VENDOR    ?=

.DEFAULT_GOAL := help

.PHONY: help compile clean arxiv

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

$(DOC).pdf: $(DOC).tex $(wildcard s*.tex) $(BIB) $(wildcard tables/*.tex)
	$(LATEX) $(LATEXOPTS) $(DOC).tex
	bibtex $(DOC)
	$(LATEX) $(LATEXOPTS) $(DOC).tex
	$(LATEX) $(LATEXOPTS) $(DOC).tex
	$(MAKE) clean

# Assemble a self-contained source tree and tar it. graphs/ and tables/ are
# symlinks into ../experiment; the globs below resolve through them and copy the
# real files, so the tarball carries no symlinks. The bibliography is flattened
# to bib/refs.bib and the \bibliography path rewritten to match (rmt cites the
# shared ../paper/bib/refs). This does not run LaTeX: arXiv compiles server-side,
# so the .bib travels along and arXiv runs BibTeX itself.
ARXIV_DIR     := arxiv-src
ARXIV_TARBALL := $(DOC).tar.gz

arxiv: $(ARXIV_TARBALL)  ## Package the self-contained arXiv source tarball

$(ARXIV_TARBALL): $(DOC).tex $(wildcard s*.tex) $(BIB) $(VENDOR) \
                  $(wildcard graphs/*.pdf) $(wildcard tables/*.tex)
	rm -rf $(ARXIV_DIR)
	mkdir -p $(ARXIV_DIR)/graphs $(ARXIV_DIR)/tables $(ARXIV_DIR)/bib
	cp $(DOC).tex $(ARXIV_DIR)/
	cp s*.tex $(ARXIV_DIR)/
	cp $(BIB) $(ARXIV_DIR)/bib/refs.bib
	cp graphs/*.pdf $(ARXIV_DIR)/graphs/
	cp tables/*.tex $(ARXIV_DIR)/tables/
	@[ -z "$(VENDOR)" ] || cp $(VENDOR) $(ARXIV_DIR)/
	sed -i.bak 's|\\bibliography{[^}]*}|\\bibliography{bib/refs}|' $(ARXIV_DIR)/$(DOC).tex
	rm -f $(ARXIV_DIR)/$(DOC).tex.bak
	tar czf $(ARXIV_TARBALL) -C $(ARXIV_DIR) .
	tar tzf $(ARXIV_TARBALL)

clean:  ## Remove LaTeX build artifacts (keeps the PDF)
	rm -rf $(ARXIV_DIR) $(ARXIV_TARBALL)
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
		$(DOC).bbl $(DOC).blg $(DOC).fls $(DOC).fdb_latexmk $(DOC).thm
