#!/usr/bin/env python3
"""P2.3: bulk-disposition tool that classifies matrix.json's 770 rows into
the spec's four disposition categories (spec Sec 4.3/4.5, ticket
catboost-8z4.36).

Input:  tests/fixtures/parity/matrix.json (written by build_matrix.py)
        tests/fixtures/parity/disposition_judgments.json (human judgment
          payload: final_method/confidence/note for the 382 method/mode-shaped
          rows build_matrix.py left as method: null)
Output: tests/fixtures/parity/matrix.dispositioned.json

Four rules, applied in the same priority order as build_matrix.py's
determine_method() (universal checked first):

1. universal == true (or method == non_capability)  -> universal_flag
2. kind in (parameter, flag)                          -> parameter_flag_family
3. kind == enum_member                                -> enum_member_attachment
4. everything else                                    -> method_mode_shaped

For method_mode_shaped rows, "method" is resolved from null to a final Sec
4.3 output-type: the 8 rows build_matrix.py already tagged "structural" by
exact name-match (plot_tree/calc_feature_statistics) are mechanical and not
re-judged; the remaining 330 come from disposition_judgments.json. Once a
final method is assigned, needs_disposition is recomputed the same way
build_matrix.py computes it (method is None) so the two fields never
disagree. Any final_method == "elementwise" row also gets its tolerance
populated from build_matrix.DEFAULT_TOLERANCE (the two spec-fixed defaults;
no new value is invented).

After disposition, closure_overlay.json (catboost-8z4.73) is merged in by
inventory_row_id: closing tickets record their differential-test outcome
(state/test_id/tolerance_justification/tolerance/final_method_*) there
instead of hand-editing matrix.dispositioned.json, so a raw pipeline rerun
stays byte-for-byte reproducible without a manual by-id merge.
"""
import json
import sys
from pathlib import Path

import build_matrix as bm

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "tests/fixtures/parity/matrix.json"
JUDGMENTS_PATH = REPO_ROOT / "tests/fixtures/parity/disposition_judgments.json"
OUTPUT_PATH = REPO_ROOT / "tests/fixtures/parity/matrix.dispositioned.json"
# Closing tickets record their outcome here (by inventory_row_id) instead of
# hand-editing matrix.dispositioned.json directly, so a raw pipeline rerun
# (compute_diff -> build_matrix -> apply_disposition) stays byte-for-byte
# reproducible without a manual by-id merge (see catboost-8z4.66, which had
# to do that merge by hand because this mechanism didn't exist yet).
CLOSURE_OVERLAY_PATH = REPO_ROOT / "tests/fixtures/parity/closure_overlay.json"

# The only fields a closure ticket may set post-hoc: differential-test
# outcome bookkeeping, never the row's identity/kind/membership.
CLOSURE_OVERLAY_ALLOWED_FIELDS = frozenset({
    "state", "test_id", "tolerance_justification",
    "tolerance", "final_method_confidence", "final_method_note",
})

STRUCTURAL_NOTE = (
    "spec §4.3 names plot_tree/calc_feature_statistics explicitly; "
    "P2.1 matched by name"
)


def disposition_row(row: dict, judgments: dict) -> dict:
    row = dict(row)  # shallow copy; don't mutate the caller's matrix rows

    if row["universal"]:
        row["disposition"] = "universal_flag"
        row["disposition_rule"] = (
            "universal == true (or method == non_capability), checked first"
        )
        row["disposition_confidence"] = 100
    elif row["kind"] in ("parameter", "flag"):
        row["disposition"] = "parameter_flag_family"
        row["disposition_rule"] = "kind in (parameter, flag)"
        row["disposition_confidence"] = 100
        row["family_id"] = row["inventory_row_id"]
        row["family_size"] = len(row["members"])
    elif row["kind"] == "enum_member":
        row["disposition"] = "enum_member_attachment"
        row["disposition_rule"] = "kind == enum_member"
        row["disposition_confidence"] = 100
        row["attached_to"] = row["members"][0]["owner"]
    else:
        row["disposition"] = "method_mode_shaped"
        row["disposition_rule"] = (
            "everything else (method/mode/submode/function/class/property/"
            "attribute, or needs_disposition: true)"
        )
        row["disposition_confidence"] = 95

        if row["method"] == "structural":
            # P2.1's mechanical spec-named exception; not re-judged.
            row["final_method"] = "structural"
            row["final_method_confidence"] = 95
            row["final_method_note"] = STRUCTURAL_NOTE
        elif row["inventory_row_id"] in judgments:
            final_method, confidence, note = judgments[row["inventory_row_id"]]
            row["method"] = final_method
            row["final_method"] = final_method
            row["final_method_confidence"] = confidence
            row["final_method_note"] = note
            if final_method == "elementwise":
                row["tolerance"] = bm.DEFAULT_TOLERANCE[row["oracle"]]
        # else: no judgment authored yet (e.g. a row newly surfaced by a
        # pipeline fix) -- leave method: null / needs_disposition: true
        # rather than crashing. Disposition is deliberately out of scope
        # here (see follow-up tickets); this only stops the tool from
        # requiring an entry that doesn't exist yet.

        # Recompute the same way build_matrix.py does, so method and
        # needs_disposition can never disagree once a row is dispositioned.
        row["needs_disposition"] = row["method"] is None

    return row


def apply_disposition(rows: list, judgments: dict) -> list:
    return [disposition_row(row, judgments) for row in rows]


def apply_closure_overlay(rows: list, overlay: dict) -> list:
    """Merge closing-ticket outcomes recorded in closure_overlay.json onto
    the freshly dispositioned rows, by inventory_row_id. Only
    CLOSURE_OVERLAY_ALLOWED_FIELDS may be set this way -- anything else
    (kind/method/members/...) is a pipeline bug, not a closure outcome."""
    overlaid = []
    for row in rows:
        row_overlay = overlay.get(row["inventory_row_id"])
        if not row_overlay:
            overlaid.append(row)
            continue
        bad_fields = set(row_overlay) - CLOSURE_OVERLAY_ALLOWED_FIELDS
        if bad_fields:
            raise ValueError(
                f"row {row['inventory_row_id']!r}: closure_overlay.json sets "
                f"disallowed field(s) {sorted(bad_fields)}"
            )
        row = dict(row)
        row.update(row_overlay)
        overlaid.append(row)
    return overlaid


def summarize(rows: list) -> dict:
    counts: dict = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    return counts


def main() -> int:
    rows = json.loads(MATRIX_PATH.read_text())
    judgments = json.loads(JUDGMENTS_PATH.read_text())

    dispositioned = apply_disposition(rows, judgments)

    overlay = json.loads(CLOSURE_OVERLAY_PATH.read_text())
    dispositioned = apply_closure_overlay(dispositioned, overlay)

    unused = set(judgments) - {
        r["inventory_row_id"] for r in dispositioned if r["disposition"] == "method_mode_shaped"
    }
    if unused:
        print(f"unused judgment entries: {sorted(unused)}", file=sys.stderr)
        return 1

    OUTPUT_PATH.write_text(json.dumps(dispositioned, indent=2) + "\n")

    summary = summarize(dispositioned)
    print(f"wrote {len(dispositioned)} rows to {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
