#!/usr/bin/env python3
"""Generate the catboostr known-parameter list and its roxygen reference
block from the machine-generated capability inventory
(tests/fixtures/parity/matrix.dispositioned.json), not by hand.

Regenerates the content between the BEGIN/END GENERATED markers in
R/catboost.R:
  1. `.catboostr_known_params`: the flat vector of the 139 canonical
     hyperparameter names used by catboost.train/catboost.cv's unknown-key
     validation gate (see process_synonyms()).
  2. A roxygen `\\describe{}` block enumerating every one of those names for
     catboost.train's help page, annotated with how catboostr actually
     accepts each one (as a params-list key, via a dedicated Pool/function
     argument, or "no catboostr equivalent").

Run: python3 tools/parity/gen_param_reference.py
(from the catboostr package root; edits R/catboost.R in place)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "tests" / "fixtures" / "parity" / "matrix.dispositioned.json"
CATBOOST_R = ROOT / "R" / "catboost.R"

KNOWN_PARAMS_BEGIN = "# BEGIN GENERATED KNOWN_PARAMS (tools/parity/gen_param_reference.py) -- DO NOT EDIT BY HAND"
KNOWN_PARAMS_END = "# END GENERATED KNOWN_PARAMS"
# `#' % ...`: a roxygen doc line whose body is an Rd comment (`%` starts a
# to-end-of-line Rd comment per the R extensions manual), so the block stays
# contiguous for roxygen2 (a plain `#` line here would split it into a
# separate, name-less block) while the marker text itself renders invisibly.
DOC_BEGIN = "#' % BEGIN GENERATED PARAM REFERENCE (tools/parity/gen_param_reference.py) -- DO NOT EDIT BY HAND"
DOC_END = "#' % END GENERATED PARAM REFERENCE"

# Names that are not native training params reachable through catboost.train's
# `params` list in catboostr, because catboostr exposes them through a
# dedicated argument or Pool-construction path instead. Classification is by
# name against catboostr's actual R surface (catboost.load_pool formals,
# catboost.train formals), not guesswork about the Python core.
DATA_ARGS = {"X", "y"}
POOL_ARGS = {
    "cat_features", "text_features", "embedding_features", "group_id",
    "group_weight", "subgroup_id", "pairs", "pairs_weight", "baseline",
    "sample_weight", "column_description", "graph",
}
DEDICATED_ARG = {
    "init_model": "the \\code{init_model} argument of \\code{catboost.train}",
    "eval_set": "the \\code{test_pool} argument of \\code{catboost.train}",
}
NO_R_EQUIVALENT = {
    "callback", "callbacks", "plot", "plot_file", "log_cout", "log_cerr",
    "silent",
}
SELF_REFERENTIAL = {"params"}

# Params catboostr already accepted pre-P5.5 (proven by a passing regression
# test) but that the machine-generated 139-row capability inventory does not
# list -- a gap in the inventory's Python-surface scan, not in catboostr.
# Included in .catboostr_known_params so the new validation gate cannot
# regress pre-existing behavior (found via a full-suite sweep after adding
# the gate; see tests/testthat/test_pool_embeddings.R).
EXTRA_KNOWN_PARAMS = {"embedding_processing"}


def parse_synonym_groups(src: str):
    """Extract the process_synonyms_in_one_group(c(...), params) alias
    groups straight from the R source -- single source of truth, no
    hand-duplicated list to drift out of sync."""
    groups = []
    for m in re.finditer(
        r"process_synonyms_in_one_group\(c\(([^)]*)\),\s*params\)", src
    ):
        names = [n.strip().strip("'\"") for n in m.group(1).split(",")]
        groups.append(names)
    return groups


def parse_load_pool_args(src: str):
    m = re.search(r"catboost\.load_pool <- function\(([^)]*(?:\)[^)]*)*?)\)\s*\{", src, re.S)
    if not m:
        # formals span multiple lines with nested parens (defaults); fall back
        # to a line-based scan up to the function body brace.
        start = src.index("catboost.load_pool <- function(")
        depth = 0
        i = start + len("catboost.load_pool <- function")
        args_start = i
        for j in range(i, len(src)):
            if src[j] == "(":
                depth += 1
                if depth == 1:
                    args_start = j + 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    arg_text = src[args_start:j]
                    break
        else:
            raise RuntimeError("could not find end of catboost.load_pool formals")
    else:
        arg_text = m.group(1)
    names = []
    for part in arg_text.split(","):
        name = part.strip().split("=")[0].strip()
        if name:
            names.append(name)
    return names


def classify(name, alias_of, pool_arg_names):
    if name in SELF_REFERENTIAL:
        return "Refers to the \\code{params} argument itself."
    if name in DEDICATED_ARG:
        return "Supplied via " + DEDICATED_ARG[name] + ", not the \\code{params} list."
    if name in DATA_ARGS:
        return "Supplied via \\code{learn_pool}/\\code{test_pool} (see \\code{catboost.load_pool}), not the \\code{params} list."
    if name in POOL_ARGS or name in pool_arg_names:
        return "Supplied via \\code{catboost.load_pool} (Pool construction), not the \\code{params} list."
    if name in NO_R_EQUIVALENT:
        return "Python-only; no catboostr equivalent."
    if alias_of is not None:
        return "Alias of \\code{%s} (see process_synonyms); resolved automatically." % alias_of
    return "Native training parameter accepted in the \\code{params} list."


def main():
    matrix = json.loads(MATRIX_PATH.read_text())
    names = sorted(
        row["family_id"][len("param:"):]
        for row in matrix
        if row.get("kind") == "parameter"
    )
    if len(names) != 139:
        print(
            "WARNING: expected 139 kind:parameter rows, found %d -- "
            "regenerating anyway (do not force the number)." % len(names),
            file=sys.stderr,
        )

    src = CATBOOST_R.read_text()
    groups = parse_synonym_groups(src)
    alias_of = {}
    for group in groups:
        canonical = group[0]
        for alias in group[1:]:
            alias_of[alias] = canonical
    pool_arg_names = set(parse_load_pool_args(src))

    # 1. .catboostr_known_params vector (139 inventory names + the small
    # EXTRA_KNOWN_PARAMS backward-compat allowlist above)
    accepted_names = sorted(set(names) | EXTRA_KNOWN_PARAMS)
    quoted = ", ".join("\"%s\"" % n for n in accepted_names)
    lines = [KNOWN_PARAMS_BEGIN]
    lines.append("# Canonical + alias hyperparameter names from the machine-generated")
    lines.append("# capability inventory (tests/fixtures/parity/matrix.dispositioned.json,")
    lines.append("# kind:\"parameter\" rows), plus EXTRA_KNOWN_PARAMS (see")
    lines.append("# tools/parity/gen_param_reference.py). Used by validate_params_keys() below.")
    wrapped = []
    cur = ".catboostr_known_params <- c("
    for i, part in enumerate(quoted.split(", ")):
        piece = part + (", " if i < len(accepted_names) - 1 else "")
        if len(cur) + len(piece) > 90:
            wrapped.append(cur)
            cur = "    "
        cur += piece
    cur += ")"
    wrapped.append(cur)
    lines.extend(wrapped)
    lines.append(KNOWN_PARAMS_END)
    known_params_block = "\n".join(lines)

    # 2. roxygen \describe{} reference block
    doc_lines = [DOC_BEGIN]
    doc_lines.append("#' @section Full Parameter Reference (generated):")
    doc_lines.append("#' All %d hyperparameters from the machine-generated capability" % len(names))
    doc_lines.append("#' inventory, and how catboostr accepts each one.")
    doc_lines.append("#' \\describe{")
    for name in names:
        note = classify(name, alias_of.get(name), pool_arg_names)
        doc_lines.append("#'   \\item{%s}{%s}" % (name, note))
    doc_lines.append("#' }")
    doc_lines.append(DOC_END)
    doc_block = "\n".join(doc_lines)

    def replace_between(text, begin, end, block):
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
        if not pattern.search(text):
            raise RuntimeError("markers not found: %s / %s" % (begin, end))
        return pattern.sub(lambda _m: block, text, count=1)

    src = replace_between(src, KNOWN_PARAMS_BEGIN, KNOWN_PARAMS_END, known_params_block)
    src = replace_between(src, DOC_BEGIN, DOC_END, doc_block)
    CATBOOST_R.write_text(src)
    print("Wrote %d parameter names into %s" % (len(names), CATBOOST_R))


if __name__ == "__main__":
    main()
