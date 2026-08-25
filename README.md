# mean_variance_solvers

LaTeX source for a set of companion papers on fast, matrix-free solvers
for the long-only minimum-variance / mean-variance portfolio problem, by Thomas
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
- **Non-Negative Conjugate Gradients** — under
  [`non_negative_cg/`](non_negative_cg/), published as
  [arXiv:2607.22121](https://arxiv.org/abs/2607.22121) (Schmelzer and Stoll,
  24 July 2026). A domain-neutral technical note that
  isolates the computational kernel shared by the two papers above: solving the
  bound-constrained SPD quadratic `min_{x≥0} ½xᵀAx − bᵀx` (and its
  least-squares / equality-augmented variants) by wrapping matrix-free
  conjugate gradients in a primal-dual active-set loop, with an unconditional
  finite-termination guarantee and an operator abstraction that admits dense,
  Gram, factor/Woodbury, and regularised backends. It contains no finance
  material; the other two papers are, in its terms, two backends of one solver.

All three papers share one bibliography
([`matrix_free/bib/refs.bib`](matrix_free/bib/refs.bib)). The two finance papers
draw every figure and table from the same numerical experiments in
[`experiment/`](experiment/); the non-negativity note is self-contained and
draws its synthetic-study figures and tables from its own
[`non_negative_cg/experiment/`](non_negative_cg/experiment/) (NumPy only, no
finance data).

## Building

A `Makefile` at the repository root drives everything; run `make` (or
`make help`) for the list of targets:

```sh
make compile   # build every paper (cg_paper.pdf, rmt_paper.pdf, nncg_paper.pdf)
make figures   # regenerate every paper's figures and tables (finance experiment + nncg study)
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
  Makefile               builds cg_paper.pdf (include ../common.mk)
  cg_paper.tex       main file: preamble + \input of the sections
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
non_negative_cg/
  Makefile               builds nncg_paper.pdf (cites ../matrix_free/bib/refs.bib)
  nncg_paper.tex         main file: preamble + \input of the sections
  sections/              one .tex per section (s0_abstract … s8_conclusions)
  experiment/            self-contained synthetic study (nncg.py + experiment_nncg.py,
                         NumPy only) and an external-solver benchmark
                         (experiment_nncg_bench.py; SciPy + Clarabel); the algorithms
                         are released as the pip package
                         https://github.com/Jebel-Quant/nncg
  graphs/                generated figure PDFs (committed; `make figures`)
  tables/                generated table + \newcommand fragments (committed)
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
  `cg_paper.pdf` and `rmt_paper.pdf` to the `pdf` branch; on tags it also
  attaches the PDFs and the arXiv source tarballs to a GitHub release.
- `arxiv` — assembles a self-contained arXiv source tarball for each paper
  (`make arxiv`) and publishes both to the `arxiv` branch.

## Branches

- `main` — primary development branch.
- `pdf` — orphan branch holding the compiled PDFs; never merged into `main`.
- `arxiv` — orphan branch for arXiv submission artifacts; never merged into `main`.
