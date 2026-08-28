#!/usr/bin/env python3
"""Rebuild book/bib/refs.bib from the companion papers' bibliographies.

The book cites across two repositories, which between them carry three BibTeX
databases with overlapping keys (markowitz1956, efron2004, cvxcla, ... appear in
more than one). Rather than maintain a fourth by hand, this concatenates them and
keeps the first definition of each key, then appends bib/book-local.bib for the
handful of entries no source defines. Run from book/:

    python3 scripts/merge_bib.py
"""
import re
import pathlib

SOURCES = [
    "../matrix_free/bib/refs.bib",
    "../../homotopy/statsci/refs/references.bib",
    "../../homotopy/cla/refs/references.bib",
    # Book-only entries, appended last so a source database always wins a key.
    "bib/book-local.bib",
]

HEADER = """% Bibliography for 'The Active Set'.
% Merged from the three source bibliographies of the companion papers:
%   mean_variance_solvers/matrix_free/bib/refs.bib
%   homotopy/statsci/refs/references.bib
%   homotopy/cla/refs/references.bib
% plus the book-only entries in bib/book-local.bib.
% Duplicate keys keep the first definition. Regenerate with book/scripts/merge_bib.py.

"""


def main() -> None:
    seen: set[str] = set()
    entries: list[str] = []
    for source in SOURCES:
        text = pathlib.Path(source).read_text(encoding="utf-8", errors="replace")
        for chunk in re.split(r"(?m)^(?=@)", text):
            match = re.match(r"@[a-zA-Z]+\s*\{\s*([^,\s]+)\s*,", chunk)
            if match is None or match.group(1) in seen:
                continue
            seen.add(match.group(1))
            entries.append(chunk.rstrip() + "\n")
    pathlib.Path("bib/refs.bib").write_text(HEADER + "\n".join(entries), encoding="utf-8")
    print(f"entries: {len(entries)}")


if __name__ == "__main__":
    main()
