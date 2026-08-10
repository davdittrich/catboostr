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
VENDOR_PLAIN_OPTIONS_HELPER = (
    ROOT / "vendor" / "catboost" / "catboost" / "private" / "libs" / "options"
    / "plain_options_helper.cpp"
)

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
    "callback", "plot", "plot_file", "log_cout", "log_cerr",
}
SELF_REFERENTIAL = {"params"}

# catboost-8z4.122: names with real catboostr-specific behavior that the
# generic per-bucket notes above don't capture (not simply "Python-only" and
# not a plain pass-through native option either) -- kept as a small,
# hand-written exception list rather than teaching classify() a one-off
# bucket for two names.
CUSTOM_NOTES = {
    "callback": (
        "Python-only (a native training-progress convenience distinct from "
        "\\code{callbacks}); no catboostr equivalent."
    ),
    "callbacks": (
        "A list of functions, each taking a single \\code{info} argument (a "
        "list with \\code{iteration} and \\code{metrics} elements, the "
        "latter shaped like \\code{\\link{catboost.get_evals_result}}'s "
        "return value) and returning \\code{TRUE} to continue training or "
        "\\code{FALSE} to stop it early. Training continues only while "
        "every callback returns \\code{TRUE}, matching Python's "
        "\\code{callbacks} kwarg."
    ),
    "silent": (
        "Client-side convenience translated to \\code{logging_level} "
        "(\"Silent\" if \\code{TRUE}, \"Verbose\" if \\code{FALSE}) before "
        "export to native, matching Python's \\code{silent} kwarg. Mutually "
        "exclusive with \\code{logging_level}/\\code{verbose}/"
        "\\code{verbose_eval}."
    ),
}

# Params catboostr already accepted pre-P5.5 (proven by a passing regression
# test) but that the machine-generated 139-row capability inventory does not
# list -- a gap in the inventory's Python-surface scan, not in catboostr.
# Included in .catboostr_known_params so the new validation gate cannot
# regress pre-existing behavior (found via a full-suite sweep after adding
# the gate; see tests/testthat/test_pool_embeddings.R).
# `embedding_processing` and `embedding_calcers` are not literal
# CopyOption(plainOptions, "...") names in plain_options_helper.cpp (both are
# routed through ParseEmbeddingProcessingOptionsFromPlainJson instead, a
# distinct code path that accepts either key as a mutually-exclusive
# alternate encoding of the same embedding-calcer descriptor -- see
# embedding_processing_options.cpp), so neither can be picked up by
# parse_native_copyoption_names() below and both stay hand-kept exceptions
# here.
EXTRA_KNOWN_PARAMS = {"embedding_processing", "embedding_calcers"}


def parse_native_copyoption_names(vendor_cpp_path):
    """Every option name vendor-native's plain-options parser accepts, read
    straight from CopyOption(plainOptions, "name", ...) and
    CopyOptionWithNewKey(plainOptions, "name", "newName", ...) call sites in
    plain_options_helper.cpp -- the authoritative superset of the 139-name
    Python-surface inventory above (catboost-8z4 Phase 5 whole-branch review
    finding: the inventory only scans Python bindings, so 59 vendor-valid
    names silently failed .catboostr_known_params). CopyOptionWithNewKey is
    a distinct helper (renames the key while copying, e.g. plain "od_pval"
    becomes odConfig's "stop_pvalue"); its first-arg-literal-"plainOptions"
    call sites accept external names just like CopyOption's, so both are
    scanned here (catboost-8z4.67: 8 names -- od_pval, od_wait, od_type,
    bootstrap_type, ctr_target_border_count, feature_border_type,
    device_config, pinned_memory_size -- were previously missed because only
    CopyOption was scanned). The regex anchors on the literal identifier
    "plainOptions" as the first argument, so it does not pick up the
    function's other call sites that copy the opposite direction (options
    struct back into plainOptionsJson for serialization, e.g.
    CopyOptionWithNewKey(odConfig, "type", "od_type", &plainOptionsJson, ...)
    around line 648) -- those name the *destination* plain key, not an
    accepted input name. Not hand-typed: any future vendor pin bump just
    needs a re-run of this script.
    """
    if not vendor_cpp_path.is_file():
        sys.exit(
            "vendor/catboost is absent. Run tools/vendor/acquire.sh before "
            "running this script (needed to derive the native option-name "
            "superset from plain_options_helper.cpp)."
        )
    src = vendor_cpp_path.read_text()
    names = set(re.findall(r'CopyOption\(plainOptions,\s*"([^"]+)"', src))
    names |= set(re.findall(r'CopyOptionWithNewKey\(plainOptions,\s*"([^"]+)"', src))

    # Coverage guard (catboost-8z4.67 fix round 1): the two regexes above
    # name the *specific* CopyOption-family helpers this script knows about.
    # This broader, function-name-agnostic regex matches any
    # CopyOption<Whatever>(plainOptions, "name", ...) call site -- i.e. any
    # existing or future member of the CopyOption* family, not just the two
    # named above -- and its results must be a subset of what the two named
    # regexes already captured. This is exactly the check that would have
    # failed before this fix round: with only the CopyOption regex active,
    # CopyOptionWithNewKey(plainOptions, "od_pval", ...) etc. would show up
    # here but not in `names`, so `missing` would be non-empty and this
    # would sys.exit instead of silently under-scanning. Deliberately scoped
    # to the CopyOption* family (not a bare `Copy\w*`), because
    # plain_options_helper.cpp also has CopyCtrDescription /
    # CopyPerFeatureCtrDescription / CopyPerFloatFeatureQuantization --
    # differently-shaped helpers already covered via the 139-name
    # Python-surface inventory (see EXTRA_KNOWN_PARAMS comment above) that
    # would otherwise be flagged as false-positive gaps.
    family_names = set(
        re.findall(r'CopyOption\w*\(plainOptions,\s*"([^"]+)"', src)
    )
    missing = family_names - names
    if missing:
        sys.exit(
            "gen_param_reference.py coverage gap: %s appear as literal "
            "CopyOption*(plainOptions, \"...\") names in "
            "plain_options_helper.cpp but weren't captured by any regex in "
            "parse_native_copyoption_names(). Vendor added a new "
            "CopyOption-family helper -- add a matching regex line here."
            % sorted(missing)
        )
    return names


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
    if name in CUSTOM_NOTES:
        return CUSTOM_NOTES[name]
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

    # 1. .catboostr_known_params vector: the 139 Python-surface inventory
    # names, unioned with every name vendor-native's CopyOption(plainOptions,
    # ...) calls accept (plain_options_helper.cpp) -- not just the
    # Python-surface subset -- plus the small EXTRA_KNOWN_PARAMS backward-compat
    # allowlist above for the one name that reaches neither list literally.
    native_names = parse_native_copyoption_names(VENDOR_PLAIN_OPTIONS_HELPER)
    accepted_names = sorted(set(names) | native_names | EXTRA_KNOWN_PARAMS)
    quoted = ", ".join("\"%s\"" % n for n in accepted_names)
    lines = [KNOWN_PARAMS_BEGIN]
    lines.append("# Canonical + alias hyperparameter names from the machine-generated")
    lines.append("# capability inventory (tests/fixtures/parity/matrix.dispositioned.json,")
    lines.append("# kind:\"parameter\" rows), unioned with every name vendor-native's")
    lines.append("# CopyOption(plainOptions, ...) calls accept (plain_options_helper.cpp;")
    lines.append("# see parse_native_copyoption_names() in this script), plus")
    lines.append("# EXTRA_KNOWN_PARAMS. Used by validate_params_keys() below.")
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
    doc_lines.append("#'")
    doc_lines.append("#' Unrecognized \\code{params} keys are rejected with an error naming the")
    doc_lines.append("#' offending key(s). To use a newer vendor-core parameter not yet listed")
    doc_lines.append("#' below, set \\code{options(catboostr.allow_unknown_params = TRUE)} to bypass")
    doc_lines.append("#' this check.")
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
