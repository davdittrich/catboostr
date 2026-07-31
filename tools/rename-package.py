#!/usr/bin/env python3
"""Rename the vendored upstream R package `catboost` -> `catboostr`.

Applies the four package-name spellings that appear in upstream's
catboost/R-package sources.  Idempotent and re-runnable: none of the four
patterns can match text this script has already rewritten, so running it twice
is a no-op.  This is what makes the strict-superset regression gate meaningful
-- the fork's R sources are upstream's sources plus exactly these
substitutions, reproducible from a fresh copy of the pinned tree.

NEVER point this at vendor/catboost/ -- that tree is a read-only pin.  Copy out
of it first, then run this on the copy.

Usage:  python3 tools/rename-package.py [ROOT]     (default: the repo root)
"""

import re
import sys
from pathlib import Path

# (label, compiled pattern, replacement)
SUBS = [
    ("library",     re.compile(r"library\(catboost\)"),        "library(catboostr)"),
    ("colons",      re.compile(r"\bcatboost::"),               "catboostr::"),
    ("test_check",  re.compile(r'test_check\("catboost"\)'),   'test_check("catboostr")'),
    ("package_arg", re.compile(r'(package\s*=\s*)"catboost"'), r'\1"catboostr"'),
]

# Everything the rename must reach.  man/ is included deliberately:
# R CMD check --as-cran runs the Rd examples, so a stale
# system.file(..., package = "catboost") there is a check ERROR.
TARGETS = ["R", "man", "tests/testthat", "tests/testthat.R", "NAMESPACE"]


def files(root: Path):
    for t in TARGETS:
        p = root / t
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.is_file())


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
    if "vendor/catboost" in str(root.resolve()):
        print(f"refusing to write into the read-only pin: {root}", file=sys.stderr)
        return 2

    totals = {label: 0 for label, _, _ in SUBS}
    changed = []
    for f in files(root):
        text = original = f.read_text(encoding="utf-8")
        for label, pat, repl in SUBS:
            text, n = pat.subn(repl, text)
            totals[label] += n
        if text != original:
            f.write_text(text, encoding="utf-8")
            changed.append(f.relative_to(root))

    for label, _, _ in SUBS:
        print(f"{label}: {totals[label]}")
    print(f"files changed: {len(changed)}")
    for c in changed:
        print(f"  {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
