"""Cross-check design spec Section 2 hand-written gap list against the
machine-generated inventory (step 6 of catboost-8z4.5). Reports two numbers:
spec claims the inventory could not confirm, and inventory gaps the spec
never mentioned.

Run: python3 tools/parity/spec_crosscheck.py
Reads: tests/fixtures/parity/{python_surface,cli_surface,capability_diff}.json
Writes: tests/fixtures/parity/spec_crosscheck.json
"""
import json
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = REPO_ROOT / "tests" / "fixtures" / "parity"


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


python_surface = json.load(open(FIX / "python_surface.json"))
cli_surface = json.load(open(FIX / "cli_surface.json"))
diff = json.load(open(FIX / "capability_diff.json"))

# Full searchable corpus: python qualified names + CLI mode/submode/flag names.
corpus = [(e["qualified_name"], norm(e["qualified_name"])) for e in python_surface["entries"]]
for m in cli_surface["modes"]:
    corpus.append((f"mode:{m['mode']}", norm(m["mode"])))
    for sm in m["submodes"]:
        corpus.append((f"mode:{m['mode']} {sm['submode']}", norm(sm["submode"])))
        for aliases in sm["flags"]:
            for a in aliases:
                corpus.append((f"flag:{m['mode']} {sm['submode']}:{a}", norm(a.lstrip("-"))))
    for aliases in m["flags"]:
        for a in aliases:
            corpus.append((f"flag:{m['mode']}:{a}", norm(a.lstrip("-"))))

# Section 2 claims, verbatim phrases from
# docs/superpowers/specs/2026-07-30-catboostr-design.md lines 25-30, split into
# discrete capability claims, each paired with a search key used to look for a
# matching program-derived symbol name (substring match on normalize()).
SPEC_CLAIMS = [
    ("embedding features", "embedding"),
    ("sparse/CSR pool input", "csr"),
    ("timestamps", "timestamp"),
    ("grid_search", "gridsearch"),
    ("randomized_search", "randomizedsearch"),
    ("select_features", "selectfeatures"),
    ("calc_feature_statistics", "calcfeaturestatistics"),
    ("ShapInteractionValues", "shapinteractionvalues"),
    ("PredictionDiff", "predictiondiff"),
    ("plot_tree", "plottree"),
    ("training-progress plotting", "plot"),
    ("model.compare", "compare"),
    ("explicit pool quantization", "quantiz"),
    ("text tokenizer/dictionary configuration", "tokenizer"),
    ("init_model continued training", "initmodel"),
    ("custom R loss/metric callbacks", "customobjective"),
    ("GPU training", "tasktype"),
    ("distributed training", "worker"),
]

spec_claims_not_found = []
matched_keys = set()
for label, key in SPEC_CLAIMS:
    hits = [qn for qn, n in corpus if key in n]
    if hits:
        matched_keys.add(key)
    else:
        spec_claims_not_found.append(label)


def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def token_exact_match(cap_tokens, key):
    """True only if `key` equals one whole token of the capability string, or
    the exact concatenation of a contiguous run of whole tokens. Rejects a
    key that merely occurs as a substring inside an unrelated longer token
    (e.g. key "csr" must not match some unrelated identifier that happens to
    contain the letters c-s-r) -- a bare `key in normalized_string` check, as
    this used to be, matched at that granularity and over-suppressed real gap
    rows in the direction that flatters the hand-written spec list, which is
    the wrong direction (see compute_diff.py's docstring for the same
    over-matching failure mode, fixed there the same way: exact match only)."""
    if key in cap_tokens:
        return True
    n = len(cap_tokens)
    for i in range(n):
        acc = ""
        for j in range(i, n):
            acc += cap_tokens[j]
            if acc == key:
                return True
            if len(acc) >= len(key):
                break
    return False


# gaps_missing_from_spec: every capability_diff.json gap row whose capability
# string does not token-exact-match any matched spec key (i.e. no plausible
# link back to a Section 2 term) -- capabilities the hand-written list never
# mentioned. Conservative on purpose (see token_exact_match docstring): a
# false "still missing" is a harmless triage item, a false "found" hides a
# real gap permanently.
gaps_missing_from_spec = []
for g in diff["gaps"]:
    cap_tokens = tokens(g["capability"])
    if not any(token_exact_match(cap_tokens, k) for k in matched_keys):
        gaps_missing_from_spec.append(f"{g['oracle']}:{g['capability']}")

out = {
    "spec_claims_total": len(SPEC_CLAIMS),
    "spec_claims_not_found": spec_claims_not_found,
    "spec_claims_not_found_count": len(spec_claims_not_found),
    "gaps_missing_from_spec_count": len(gaps_missing_from_spec),
    "gaps_missing_from_spec": gaps_missing_from_spec,
}
with open(FIX / "spec_crosscheck.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps({k: v for k, v in out.items() if k != "gaps_missing_from_spec"}, indent=2))
