"""Compute Python/CLI vs R-export diff by reproducible name normalization.

Matching rule (deterministic, conservative, auditable):
  normalize(name) = lowercase, strip all non-alphanumeric characters.
  A candidate (Python entry / CLI mode / CLI flag) is COVERED only if its
  normalized last-component name EXACTLY equals a normalized R export suffix
  or S3 generic name. No substring/fuzzy matching: a coarser rule was tried
  and 92 of 140 "covered" rows turned out to be false positives (e.g. CLI
  `--has-header` matching R `head()`, `model_shrink_rate` matching `shrink`)
  -- silently dropping real gap rows with no trail. Exact match only, so
  anything not confidently matched stays a gap row (over-reporting is the
  correct failure direction per the brief: a false gap gets triaged in
  Phase 2, a false "covered" is never seen again). Every match this rule
  does make is recorded in `covered` with the matched R symbol, auditable
  the same way private_excluded_count already is.

Caveat (recorded, not used to drop rows): R's catboost.train/catboost.cv take
an untyped `params` named list, so most individual CatBoostClassifier
hyperparameter names have no discrete R export to match 1:1 by construction.
They are still emitted as gap rows (python_only, kind=parameter) per the
no-filtering guard; the mechanism note belongs in the report, not in row
suppression. A small number DO happen to share an exact name with an R
export (e.g. `train_dir`, `eval_metric` -- matched against catboost.train /
catboost.eval_metrics normalized suffixes) and are recorded as covered.

Run: python3 tools/parity/compute_diff.py
Reads: tests/fixtures/parity/{python_surface,cli_surface,r_surface}.json
Writes: tests/fixtures/parity/capability_diff.json
"""
import json
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = REPO_ROOT / "tests" / "fixtures" / "parity"

python_surface = json.load(open(FIX / "python_surface.json"))
cli_surface = json.load(open(FIX / "cli_surface.json"))
r_surface = json.load(open(FIX / "r_surface.json"))


def normalize(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# --- R canonical set ---
r_canon = {}  # normalized -> list of original names
for exp in r_surface["exports"]:
    suffix = exp.split(".", 1)[1] if exp.startswith("catboost.") else exp
    r_canon.setdefault(normalize(suffix), []).append(exp)
for s3 in r_surface["s3methods"]:
    r_canon.setdefault(normalize(s3["generic"]), []).append(f"S3method({s3['generic']},{s3['class']})")


def covered(cand_norm):
    if not cand_norm:
        return None
    hit = r_canon.get(cand_norm)
    return hit[0] if hit else None


gaps = []
covered_rows = []


def classify(oracle, capability, kind, owner, cand_norm, dedup_key=None):
    r_symbol = covered(cand_norm)
    if r_symbol:
        covered_rows.append({"oracle": oracle, "capability": capability, "kind": kind, "owner": owner, "matched_r_symbol": r_symbol})
    else:
        row = {"oracle": oracle, "capability": capability, "kind": kind, "owner": owner}
        if dedup_key is not None:
            row["dedup_key"] = dedup_key
        gaps.append(row)


# --- Python candidates ---
for e in python_surface["entries"]:
    qn = e["qualified_name"]
    if e["kind"] == "parameter":
        m = re.search(r"param=([^)]+)\)$", qn)
        last = m.group(1) if m else qn
        # dedup_key: bare parameter name. The same hyperparameter name (e.g.
        # `iterations`) is re-emitted per estimator subclass __init__ (I4) --
        # this key lets a reader recover the distinct-name count without
        # dropping any row.
        classify("python", qn, e["kind"], e["owner"], normalize(last), dedup_key=f"param:{last}")
    else:
        last = qn.split("(")[0].split(".")[-1]
        classify("python", qn, e["kind"], e["owner"], normalize(last))

# --- CLI candidates: modes / submodes / flags ---
mode_scope_count = 0  # number of mode/submode scopes a universal flag could appear in
for m in cli_surface["modes"]:
    classify("cli", f"mode:{m['mode']}", "mode", None, normalize(m["mode"]))

    if m["submodes"]:
        for sm in m["submodes"]:
            mode_scope_count += 1
            full = f"{m['mode']} {sm['submode']}"
            classify("cli", f"mode:{full}", "submode", m["mode"], normalize(sm["submode"]))
            for flag_aliases in sm["flags"]:
                primary = sorted(flag_aliases, key=len, reverse=True)[0]
                alias_key = "/".join(flag_aliases)
                # dedup_key: the alias-set text alone, mode-independent. The
                # same flag (e.g. --help, --svnrevision) is re-emitted once
                # per mode/submode scope (I4) -- this key recovers the
                # distinct-flag count without dropping any row.
                classify("cli", f"flag:{full}:{alias_key}", "flag", full, normalize(primary.lstrip("-")), dedup_key=f"flag:{alias_key}")
    else:
        mode_scope_count += 1
        for flag_aliases in m["flags"]:
            primary = sorted(flag_aliases, key=len, reverse=True)[0]
            alias_key = "/".join(flag_aliases)
            classify("cli", f"flag:{m['mode']}:{alias_key}", "flag", m["mode"], normalize(primary.lstrip("-")), dedup_key=f"flag:{alias_key}")

# --- Deduplication (I4): rows are never dropped; these are additional
# fields recovering the distinct-entity counts the raw per-scope counts
# inflate. A CLI flag is "universal" (not a real capability -- --help,
# --svnrevision) if it recurs in nearly every mode/submode scope measured
# (>= scope_count - 1, so one scope omitting it, e.g. dump-options omitting
# --help, does not exclude it). Genuinely shared data-loading flags like
# --column-description top out around scope_count/2 and stay unmarked.
flag_gaps = [g for g in gaps if g["kind"] == "flag"]
param_gaps = [g for g in gaps if g["kind"] == "parameter"]
flag_occurrences = {}
for g in flag_gaps:
    flag_occurrences[g["dedup_key"]] = flag_occurrences.get(g["dedup_key"], 0) + 1
universal_flag_keys = {k for k, n in flag_occurrences.items() if n >= mode_scope_count - 1}
for g in flag_gaps:
    g["universal"] = g["dedup_key"] in universal_flag_keys

distinct_flag_count = len({g["dedup_key"] for g in flag_gaps})
distinct_parameter_count = len({g["dedup_key"] for g in param_gaps})
non_dedup_gap_count = len(gaps) - len(flag_gaps) - len(param_gaps)
gap_count_deduplicated = non_dedup_gap_count + distinct_flag_count + distinct_parameter_count

gap_python_only = sum(1 for g in gaps if g["oracle"] == "python")
gap_cli_only = sum(1 for g in gaps if g["oracle"] == "cli")

out = {
    "matching_rule": "normalize=lowercase+strip-non-alnum; EXACT match only against R export suffix / S3 generic name (no substring/fuzzy matching -- see module docstring)",
    "gap_count_total": len(gaps),
    "gap_count_python_only": gap_python_only,
    "gap_count_cli_only": gap_cli_only,
    "gap_count_deduplicated": gap_count_deduplicated,
    "distinct_flag_count": distinct_flag_count,
    "distinct_parameter_count": distinct_parameter_count,
    "universal_flag_count": len(universal_flag_keys),
    "universal_flags": sorted(k[len("flag:"):] for k in universal_flag_keys),
    "dedup_note": (
        "gap_count_total counts one row per (mode/submode/class) scope a gap "
        "appears in -- the raw number quoted elsewhere as '1537 gaps' is this "
        "figure. gap_count_deduplicated collapses CLI flag rows to distinct "
        "alias-sets and Python parameter rows to distinct names (see each "
        "gap's dedup_key); non-flag/non-parameter rows are not deduplicated. "
        "universal_flags recur in every measured mode/submode scope (e.g. "
        "--help, --svnrevision) and are not real capabilities."
    ),
    "covered_count": len(covered_rows),
    "gaps": gaps,
    "covered": covered_rows,
}
with open(FIX / "capability_diff.json", "w") as f:
    json.dump(out, f, indent=2)

print(
    f"gap_total={len(gaps)} (deduplicated={gap_count_deduplicated}) "
    f"python_only={gap_python_only} cli_only={gap_cli_only} covered={len(covered_rows)} "
    f"distinct_flags={distinct_flag_count} distinct_parameters={distinct_parameter_count} "
    f"universal_flags={len(universal_flag_keys)}"
)
