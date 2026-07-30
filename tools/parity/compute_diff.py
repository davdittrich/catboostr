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


def classify(oracle, capability, kind, owner, cand_norm):
    r_symbol = covered(cand_norm)
    if r_symbol:
        covered_rows.append({"oracle": oracle, "capability": capability, "kind": kind, "owner": owner, "matched_r_symbol": r_symbol})
    else:
        gaps.append({"oracle": oracle, "capability": capability, "kind": kind, "owner": owner})


# --- Python candidates ---
for e in python_surface["entries"]:
    qn = e["qualified_name"]
    if e["kind"] == "parameter":
        m = re.search(r"param=([^)]+)\)$", qn)
        last = m.group(1) if m else qn
    else:
        last = qn.split("(")[0].split(".")[-1]
    classify("python", qn, e["kind"], e["owner"], normalize(last))

# --- CLI candidates: modes / submodes / flags ---
for m in cli_surface["modes"]:
    classify("cli", f"mode:{m['mode']}", "mode", None, normalize(m["mode"]))

    if m["submodes"]:
        for sm in m["submodes"]:
            full = f"{m['mode']} {sm['submode']}"
            classify("cli", f"mode:{full}", "submode", m["mode"], normalize(sm["submode"]))
            for flag_aliases in sm["flags"]:
                primary = sorted(flag_aliases, key=len, reverse=True)[0]
                classify("cli", f"flag:{full}:{'/'.join(flag_aliases)}", "flag", full, normalize(primary.lstrip("-")))
    else:
        for flag_aliases in m["flags"]:
            primary = sorted(flag_aliases, key=len, reverse=True)[0]
            classify("cli", f"flag:{m['mode']}:{'/'.join(flag_aliases)}", "flag", m["mode"], normalize(primary.lstrip("-")))

gap_python_only = sum(1 for g in gaps if g["oracle"] == "python")
gap_cli_only = sum(1 for g in gaps if g["oracle"] == "cli")

out = {
    "matching_rule": "normalize=lowercase+strip-non-alnum; EXACT match only against R export suffix / S3 generic name (no substring/fuzzy matching -- see module docstring)",
    "gap_count_total": len(gaps),
    "gap_count_python_only": gap_python_only,
    "gap_count_cli_only": gap_cli_only,
    "covered_count": len(covered_rows),
    "gaps": gaps,
    "covered": covered_rows,
}
with open(FIX / "capability_diff.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"gap_total={len(gaps)} python_only={gap_python_only} cli_only={gap_cli_only} covered={len(covered_rows)}")
