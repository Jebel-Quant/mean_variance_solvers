# mean_variance_solvers

LaTeX source for two companion working papers on fast, matrix-free solvers for
the long-only minimum-variance / mean-variance portfolio problem, by Thomas
Schmelzer (Jebel Quant Research), Martin Stoll (TU Chemnitz), and Michael Wolf
(University of Zurich / ADIA Lab):

- **Matrix-Free Methods for Long-Only Portfolio Optimization** — under
  [`matrix_free/`](matrix_free/). Casts the long-only minimum-variance problem
  so that covariance shrinkage acts as a preconditioner, and solves it with
  matrix-free iterative methods that never form the dense covariance matrix.
- **From Marchenko–Pastur to Woodbury: Direct Solvers for Long-Only
  Mean-Variance Portfolios** — under [`rmt/`](rmt/). Uses random-matrix theory
  (Marchenko–Pastur) to motivate a low-rank-plus-diagonal covariance model that
  a Woodbury identity turns into a fast direct solver.

Both papers share one bibliography
([`matrix_free/bib/refs.bib`](matrix_free/bib/refs.bib))
and draw every figure and table from the same numerical experiments in
[`experiment/`](experiment/).

## Building

A `Makefile` at the repository root drives everything; run `make` (or
`make help`) for the list of targets:

```sh
make compile   # build both papers (matrix_free/minvar_paper.pdf and rmt/rmt_paper.pdf)
make figures   # regenerate both papers' figures and tables (runs the experiment)
make arxiv     # assemble a self-contained arXiv source tarball per paper
make clean     # remove both papers' LaTeX build artifacts (keeps the PDFs)
```

`make compile` runs `pdflatex → bibtex → pdflatex → pdflatex` per paper: the
first pass records the citations, BibTeX turns them into a formatted
bibliography, and the final two passes fold it in and resolve all
cross-references. The minimum-variance paper uses the standard `siam`
bibliography style; the RMT paper uses `plain`. Both ship with any full TeX
Live install, so no class or style files need to be vendored.

`make figures` delegates to [`experiment/`](experiment/), whose scripts are
self-contained [PEP 723](https://peps.python.org/pep-0723/) programs run with
[`uv`](https://docs.astral.sh/uv/); their pinned dependencies include the
[`fast-minimum-variance`](https://github.com/Jebel-Quant/fast_minimum_variance)
package, resolved directly from its git repository. The committed figure PDFs
and table `.tex` files mean the papers compile without ever running the
experiment.

## Layout

```
Makefile                 root entry point: delegates to the per-paper Makefiles
common.mk                shared build logic (compile / arxiv / clean)
matrix_free/
  Makefile               builds minvar_paper.pdf (include ../common.mk)
  minvar_paper.tex       main file: preamble + \input of the sections
  sections/              one .tex per section (s0_abstract … s8_conclusions)
  bib/refs.bib           shared BibTeX database
  siam/                  vendored SIAM class/style (reference copy)
  graphs -> ../experiment/graphs    figure PDFs (symlink)
  tables -> ../experiment/tables    table .tex fragments (symlink)
rmt/
  Makefile               builds rmt_paper.pdf (cites ../matrix_free/bib/refs.bib)
  rmt_paper.tex          main file: preamble + \input of the sections
  sections/              one .tex per section (s0_abstract … s8_conclusions)
  graphs -> ../experiment/graphs
  tables -> ../experiment/tables
experiment/              numerical experiments that generate graphs/ and tables/
  experiment*.py         PEP 723 scripts (real-data, synthetic, OOS, RMT)
  fetch_*.py             download the raw S&P 500 / FTSE 100 return data
  data/                  committed return data (*.parquet)
  graphs/                generated figure PDFs (committed)
  tables/                generated table .tex fragments (committed)
  util/                  shared table/timing helpers
```

Adding a new paper is a matter of dropping in a folder with a `Makefile` that
sets `DOC` and `include ../common.mk`; the root `Makefile` discovers it
automatically (the `experiment/` folder is excluded).

## Continuous integration

GitHub Actions workflows (`.github/workflows/`):

- `build` — compiles both papers (`make compile`) and publishes
  `minvar_paper.pdf` and `rmt_paper.pdf` to the `pdf` branch; on tags it also
  attaches the PDFs and the arXiv source tarballs to a GitHub release.
- `arxiv` — assembles a self-contained arXiv source tarball for each paper
  (`make arxiv`) and publishes both to the `arxiv` branch.

## Branches

- `main` — primary development branch.
- `pdf` — orphan branch holding the compiled PDFs; never merged into `main`.
- `arxiv` — orphan branch for arXiv submission artifacts; never merged into `main`.
