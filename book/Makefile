# Build the textbook ("The Active Set: A Graduate Course in Convex Quadratic
# Optimization"). Unlike the papers in this repository the book does not include
# ../common.mk: it has chapters/ rather than sections/, needs an index pass, and
# needs the extra pdflatex run that a book-class table of contents requires.
# The target names match the papers' (help/compile/stage/clean) so that the
# repository-root `make compile` and `make pdfs`, which discover every folder
# holding a Makefile, pick the book up with no change.

DOC       := book
LATEX     := pdflatex
LATEXOPTS := -interaction=nonstopmode -halt-on-error
BIB       := bib/refs.bib
OUTDIR    ?= ../out

.DEFAULT_GOAL := help

.PHONY: help compile stage bib clean

help:  ## Show this overview of available commands
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| sort \
		| awk -F':.*## ' '{ printf "  %-10s %s\n", $$1, $$2 }'
	@echo ""

compile: $(DOC).pdf  ## Build the book PDF

stage: $(DOC).pdf  ## Copy the book PDF into $(OUTDIR)
	@mkdir -p $(OUTDIR)
	cp $(DOC).pdf $(OUTDIR)/

# Four LaTeX passes, not the papers' three: the first records citations and
# index entries, bibtex and makeindex turn them into files, and the last two
# fold both in and settle the table of contents and the page cross-references,
# which move once the bibliography and index have taken their pages.
$(DOC).pdf: $(DOC).tex $(wildcard chapters/*.tex) $(wildcard frontmatter/*.tex) $(BIB)
	$(LATEX) $(LATEXOPTS) $(DOC).tex
	bibtex $(DOC)
	-makeindex $(DOC).idx
	$(LATEX) $(LATEXOPTS) $(DOC).tex
	$(LATEX) $(LATEXOPTS) $(DOC).tex
	$(MAKE) clean

bib:  ## Rebuild bib/refs.bib from the companion papers' bibliographies
	python3 scripts/merge_bib.py

clean:  ## Remove LaTeX build artifacts (keeps the PDF)
	rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
		$(DOC).bbl $(DOC).blg $(DOC).fls $(DOC).fdb_latexmk \
		$(DOC).idx $(DOC).ilg $(DOC).ind $(DOC).lof $(DOC).lot
	rm -f chapters/*.aux frontmatter/*.aux
