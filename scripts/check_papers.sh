#!/usr/bin/env bash
#
# CI quality gate for the papers (issue #28). Four cheap compile-time and
# static checks that would have caught defects fixed by hand:
#
#   (1) undefined references / citations in the final pdflatex log;
#   (2) spaced ` -- ` dash asides in sections/ (en-dash ranges like 6--18 are
#       fine — only the space-flanked pattern is flagged);
#   (3) placeholder patterns (XXXX) in the BibTeX databases;
#   (4) a solver name in a paper's tables that the paper's prose never mentions
#       (catches stale rows left behind when a generated table is regenerated).
#
# Runs every check, then exits non-zero if any failed. Check (1) needs a LaTeX
# toolchain; when pdflatex is absent it is skipped with a warning (CI ships
# TeX Live, so it always runs there) and the static checks (2)-(4) still run.

set -uo pipefail

cd "$(dirname "$0")/.."

fail=0
report() { printf '  FAIL: %s\n' "$1"; fail=1; }

# Papers = every top-level subdirectory with a Makefile except experiment/
# (the figure/table generator). Mirrors the repository-root Makefile.
papers=()
for mk in */Makefile; do
	d=${mk%/Makefile}
	[ "$d" = experiment ] && continue
	papers+=("$d")
done

doc_of() { sed -n 's/^DOC[[:space:]]*:=[[:space:]]*//p' "$1/Makefile" | head -1; }

# The .tex files that make up a paper's prose: the main document plus sections/.
prose_files() { echo "$1"/*.tex "$1"/sections/*.tex; }

# ---------------------------------------------------------------------------
# (1) Undefined references and citations.
# ---------------------------------------------------------------------------
echo "[1/4] Undefined references and citations"
if command -v pdflatex >/dev/null 2>&1; then
	for d in "${papers[@]}"; do
		doc=$(doc_of "$d")
		# Full build (bibtex + two extra passes) so cross-references resolve;
		# a partial build would report every reference as undefined.
		(
			cd "$d" &&
				pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 &&
				bibtex "$doc" >/dev/null 2>&1 &&
				pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 &&
				pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1
		)
		log="$d/$doc.log"
		if [ -f "$log" ] && grep -qE "(Reference|Citation) \`.*' .*undefined|There were undefined references" "$log"; then
			report "$d: undefined references/citations in $doc.log:"
			grep -oE "(Reference|Citation) \`[^']*' .*undefined" "$log" | sort -u | sed 's/^/         /'
		fi
		make -s -C "$d" clean >/dev/null 2>&1
	done
else
	echo "  pdflatex not found — skipping (CI provides TeX Live)"
fi

# ---------------------------------------------------------------------------
# (2) Spaced ` -- ` dash asides in a document's prose. En-dash ranges (6--18)
#     have no surrounding spaces and are not matched; full-line comments are
#     ignored. A paper keeps its prose in sections/; the book keeps its own in
#     chapters/ and frontmatter/, so both directory names are checked and a
#     document is skipped only when it has neither.
# ---------------------------------------------------------------------------
echo "[2/4] Spaced -- dash asides in prose"
for d in "${papers[@]}"; do
	prose_dirs=()
	for sub in sections chapters frontmatter; do
		[ -d "$d/$sub" ] && prose_dirs+=("$d/$sub"/*.tex)
	done
	[ "${#prose_dirs[@]}" -eq 0 ] && continue
	hits=$(grep -rnE '[^%]* -- ' "${prose_dirs[@]}" | grep -vE '^[^:]+:[0-9]+:[[:space:]]*%')
	if [ -n "$hits" ]; then
		report "$d: dash asides in prose:"
		echo "$hits" | sed 's/^/         /'
	fi
done

# ---------------------------------------------------------------------------
# (3) Placeholder patterns in the BibTeX databases.
# ---------------------------------------------------------------------------
echo "[3/4] Placeholder patterns in .bib files"
while IFS= read -r bib; do
	hits=$(grep -nE 'XXXX' "$bib")
	if [ -n "$hits" ]; then
		report "$bib: placeholder pattern:"
		echo "$hits" | sed 's/^/         /'
	fi
done < <(git ls-files '*.bib')

# ---------------------------------------------------------------------------
# (4) Stale solver rows: a solver named in a paper's tables but never in its
#     prose. Curated vocabulary of solver display names; extend when a new
#     solver is added to the experiments.
# ---------------------------------------------------------------------------
echo "[4/4] Solver names present in tables but absent from prose"
solvers=("Clarabel" "OSQP" "Woodbury" "Proximal" "Lawson" "Interior point" "Active set")
for d in "${papers[@]}"; do
	# Table files the paper actually \input's, resolved through the tables/
	# symlink (or real directory) to their real paths.
	tbl_files=()
	while IFS= read -r t; do
		[ -n "$t" ] || continue
		f="$d/tables/$t"
		[ -f "$f" ] || f="$f.tex"
		[ -f "$f" ] && tbl_files+=("$f")
	done < <(
		grep -rhoE '\\input\{tables/[^}]+\}' $(prose_files "$d") 2>/dev/null |
			sed -E 's/.*\{tables\/([^}]+)\}/\1/'
	)
	[ "${#tbl_files[@]}" -eq 0 ] && continue
	for s in "${solvers[@]}"; do
		if grep -qiF -- "$s" "${tbl_files[@]}" && ! grep -qiF -- "$s" $(prose_files "$d"); then
			report "$d: solver '$s' appears in tables but not in the prose"
		fi
	done
done

echo
if [ "$fail" -ne 0 ]; then
	echo "Quality gate FAILED."
	exit 1
fi
echo "Quality gate passed."
