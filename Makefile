# Repository-root Makefile. Each paper has its own self-contained Makefile
# (paper/Makefile builds cg_paper.pdf, paper_rmt/Makefile builds
# rmt_paper.pdf); the compile, clean, and arxiv targets here run every paper.
# The two finance papers' figures and tables come from the experiment/ Python
# package; a paper that owns a self-contained study instead (non_negative_cg)
# exposes its own `figures` target, which the figures target below also runs.

.DEFAULT_GOAL := help

# Every subdirectory with a Makefile is a paper, except experiment/ (the figure
# and table generator). Add a new paper by dropping in its folder; no target
# below needs editing.
PAPERS := $(filter-out experiment,$(patsubst %/Makefile,%,$(wildcard */Makefile)))

.PHONY: help compile pdfs figures test clean arxiv check

# Self-documenting help: lists every target with a `## description` comment.
help:  ## Show this overview of available commands
	@echo "Usage: make <target>"
	@echo ""
	@echo "Papers: $(PAPERS)"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| sort \
		| awk -F':.*## ' '{ printf "  %-10s %s\n", $$1, $$2 }'
	@echo ""

compile:  ## Build every paper
	@$(foreach p,$(PAPERS),$(MAKE) -C $(p) compile &&) true

# Collect every paper's PDF under out/. Each paper's own `stage` target does the
# copying, so this needs to know which folders are papers but not what any of
# them calls its document -- that lives in the paper's DOC. OUTDIR is absolute
# because make -C has already changed directory when it is expanded.
#
# This is what the CI publish step runs. It used to `cp` three PDFs by name,
# which meant a fourth paper (quadprog/) compiled and gated in CI but was never
# published, and nothing failed to say so. Anything that discovers papers should
# discover them the way PAPERS above does.
pdfs:  ## Collect every paper's PDF under out/
	@mkdir -p out
	@$(foreach p,$(PAPERS),$(MAKE) -C $(p) stage OUTDIR=$(CURDIR)/out &&) true
	@ls -l out

figures:  ## Regenerate every paper's figures and tables (runs the experiment)
	$(MAKE) -C experiment figures
	@for p in $(PAPERS); do \
		if grep -qE '^figures:' $$p/Makefile; then \
			echo "==> $(MAKE) -C $$p figures"; $(MAKE) -C $$p figures; \
		fi; \
	done

test:  ## Run the experiment test suite (baselines, cg, and end-to-end runs)
	$(MAKE) -C experiment test

check:  ## Run the CI quality gate (undefined refs, dash asides, bib placeholders, stale solver rows)
	./scripts/check_papers.sh

clean:  ## Remove every paper's LaTeX build artifacts
	@$(foreach p,$(PAPERS),$(MAKE) -C $(p) clean &&) true
	rm -rf out

arxiv:  ## Build every paper's arXiv tarball and collect them under out/arxiv/
	@mkdir -p out/arxiv
	@$(foreach p,$(PAPERS),$(MAKE) -C $(p) arxiv && cp $(p)/*.tar.gz out/arxiv/ &&) true
	@ls -l out/arxiv
