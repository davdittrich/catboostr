"""Compute Python/CLI vs R-export diff by reproducible name normalization.

Matching rule (deterministic, no judgment calls on importance):
  normalize(name) = lowercase, strip all non-alphanumeric characters.
  A candidate (Python entry / CLI mode / CLI flag) is COVERED if its
  normalized last-component name either exactly equals, or is a substring
  (len>=4) of / contains, a normalized R export suffix or S3 generic name.
  Otherwise it is a GAP, tagged with the oracle it came from.

Caveat (recorded, not used to drop rows): R's catboost.train/catboost.cv take
an untyped `params` named list, so individual CatBoostClassifier hyperparameter
names have no discrete R export to match 1:1 by construction. They are still
emitted as gap rows (python_only, kind=parameter) per the no-filtering guard;
the mechanism note belongs in the report, not in row suppression.

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

r_norms = list(r_canon.keys())


def covered(cand_norm):
    if not cand_norm:
        return None
    if cand_norm in r_canon:
        return ("exact", r_canon[cand_norm][0])
    for rn in r_norms:
        if len(rn) >= 4 and (rn in cand_norm or cand_norm in rn):
            return ("substring", r_canon[rn][0])
    return None


gaps = []
covered_count = 0

# --- Python candidates ---
for e in python_surface["entries"]:
    qn = e["qualified_name"]
    if e["kind"] == "parameter":
        m = re.search(r"param=([^)]+)\)$", qn)
        last = m.group(1) if m else qn
    else:
        last = qn.split("(")[0].split(".")[-1]
    cand_norm = normalize(last)
    hit = covered(cand_norm)
    if hit:
        covered_count += 1
    else:
        gaps.append({
            "oracle": "python",
            "capability": qn,
            "kind": e["kind"],
            "owner": e["owner"],
        })

# --- CLI candidates: modes / submodes / flags ---
for m in cli_surface["modes"]:
    hit = covered(normalize(m["mode"]))
    if hit:
        covered_count += 1
    else:
        gaps.append({"oracle": "cli", "capability": f"mode:{m['mode']}", "kind": "mode", "owner": None})

    if m["submodes"]:
        for sm in m["submodes"]:
            full = f"{m['mode']} {sm['submode']}"
            hit = covered(normalize(sm["submode"]))
            if hit:
                covered_count += 1
            else:
                gaps.append({"oracle": "cli", "capability": f"mode:{full}", "kind": "submode", "owner": m["mode"]})
            for flag_aliases in sm["flags"]:
                primary = sorted(flag_aliases, key=len, reverse=True)[0]
                hit = covered(normalize(primary.lstrip("-")))
                if hit:
                    covered_count += 1
                else:
                    gaps.append({
                        "oracle": "cli",
                        "capability": f"flag:{full}:{'/'.join(flag_aliases)}",
                        "kind": "flag",
                        "owner": full,
                    })
    else:
        for flag_aliases in m["flags"]:
            primary = sorted(flag_aliases, key=len, reverse=True)[0]
            hit = covered(normalize(primary.lstrip("-")))
            if hit:
                covered_count += 1
            else:
                gaps.append({
                    "oracle": "cli",
                    "capability": f"flag:{m['mode']}:{'/'.join(flag_aliases)}",
                    "kind": "flag",
                    "owner": m["mode"],
                })

gap_python_only = sum(1 for g in gaps if g["oracle"] == "python")
gap_cli_only = sum(1 for g in gaps if g["oracle"] == "cli")

out = {
    "matching_rule": "normalize=lowercase+strip-non-alnum; exact or substring(>=4 chars) match against R export suffix / S3 generic name",
    "gap_count_total": len(gaps),
    "gap_count_python_only": gap_python_only,
    "gap_count_cli_only": gap_cli_only,
    "covered_count": covered_count,
    "gaps": gaps,
}
with open(FIX / "capability_diff.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"gap_total={len(gaps)} python_only={gap_python_only} cli_only={gap_cli_only} covered={covered_count}")
