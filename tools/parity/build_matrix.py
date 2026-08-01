#!/usr/bin/env python3
"""P2.1: join tool that merges capability_diff.json's 1537 raw gap rows down
to the spec's 725-row parity matrix (dedup_key grouping, spec Sec 4.3/4.5).

Input:  tests/fixtures/parity/capability_diff.json (written by compute_diff.py)
Output: tests/fixtures/parity/matrix.json

Scope (spec-derived, ticket catboost-8z4.34): builds the join/merge skeleton
and tags what is mechanically determinable (oracle, universal-flag exclusion,
parameter/flag/enum_member kind, the two spec-named structural capabilities).
Everything requiring semantic judgment (which output-type method a
method/mode/class/property/attribute/function row needs) is left
method: null, needs_disposition: true for P2.3.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "tests/fixtures/parity/capability_diff.json"
OUTPUT_PATH = REPO_ROOT / "tests/fixtures/parity/matrix.json"

# Spec Sec 4.3 tolerance defaults -- fixed policy, not invented here.
DEFAULT_TOLERANCE = {"python": 1e-12, "cli": 1e-9}
CLI_PRECISION_FLOOR = 1e-9  # spec Sec 4.3 precision ceiling: CLI can't declare tighter.

# Spec Sec 4.3 names these two explicitly as structural capabilities.
STRUCTURAL_NAME_EXCEPTIONS = ("plot_tree", "calc_feature_statistics")


def is_universal(row: dict) -> bool:
    return bool(row.get("universal", False))


def determine_method(kind: str, capability: str, universal: bool) -> str | None:
    """Spec Sec IV step 5, in priority order."""
    if universal:
        return "non_capability"
    if kind in ("parameter", "flag"):
        return "parameter_or_flag"
    if kind == "enum_member":
        return "enum_member"
    cap_lower = capability.lower()
    if any(name in cap_lower for name in STRUCTURAL_NAME_EXCEPTIONS):
        return "structural"
    return None


def validate_tolerance_justification(row: dict) -> None:
    """Non-default tolerance with empty justification is a schema violation,
    not a valid row (spec Sec 4.3 tolerance policy)."""
    tolerance = row["tolerance"]
    justification = row["tolerance_justification"]
    if tolerance is None:
        return
    oracle = row["oracle"]
    default = DEFAULT_TOLERANCE.get(oracle)
    if default is not None and tolerance != default and not justification:
        raise ValueError(
            f"row {row['inventory_row_id']!r}: non-default tolerance "
            f"{tolerance!r} requires a tolerance_justification"
        )


def validate_precision_ceiling(row: dict) -> None:
    """CLI oracle rows may never declare a tolerance tighter than the CLI's
    own measured ~10-significant-digit precision (relative 1e-9 floor)."""
    tolerance = row["tolerance"]
    if tolerance is None or row["oracle"] != "cli":
        return
    if tolerance < CLI_PRECISION_FLOOR:
        raise ValueError(
            f"row {row['inventory_row_id']!r}: CLI tolerance {tolerance!r} "
            f"is tighter than the precision ceiling {CLI_PRECISION_FLOOR!r}"
        )


def build_row(inventory_row_id: str, oracle: str, kind: str, universal: bool,
              members: list) -> dict:
    representative_capability = members[0]["capability"]
    method = determine_method(kind, representative_capability, universal)
    tolerance = None  # spec Sec 4.3: only "elementwise" carries a numeric
    # default, and no row in this ticket's scope is assigned "elementwise"
    # (parameter/flag/enum_member/non_capability rows are explicitly n/a;
    # every other row is method: null pending P2.3 disposition).
    tolerance_justification = ""
    row = {
        "inventory_row_id": inventory_row_id,
        "oracle": oracle,
        "kind": kind,
        "method": method,
        "tolerance": tolerance,
        "test_id": None,
        "state": "red",
        "tolerance_justification": tolerance_justification,
        "needs_disposition": method is None,
        "members": members,
        "universal": universal,
    }
    validate_tolerance_justification(row)
    validate_precision_ceiling(row)
    return row


def build_matrix(diff: dict) -> list:
    gaps = diff["gaps"]
    assert len(gaps) == 1537, f"expected 1537 raw gap rows, got {len(gaps)}"

    merged_groups: dict[str, list] = {}
    passthrough: list = []
    for gap in gaps:
        dedup_key = gap.get("dedup_key")
        if dedup_key is None:
            passthrough.append(gap)
        else:
            merged_groups.setdefault(dedup_key, []).append(gap)

    assert len(merged_groups) == 363, (
        f"expected 363 distinct dedup_key groups, got {len(merged_groups)}"
    )
    assert len(passthrough) == 362, (
        f"expected 362 passthrough rows, got {len(passthrough)}"
    )

    rows = []

    for dedup_key, group in merged_groups.items():
        oracles = {g["oracle"] for g in group}
        assert len(oracles) == 1, (
            f"dedup group {dedup_key!r} spans multiple oracles: {oracles}"
        )
        kinds = {g["kind"] for g in group}
        assert len(kinds) == 1, (
            f"dedup group {dedup_key!r} spans multiple kinds: {kinds}"
        )
        universal_flags = {is_universal(g) for g in group}
        assert len(universal_flags) == 1, (
            f"dedup group {dedup_key!r} has inconsistent 'universal' across members"
        )
        members = [{"capability": g["capability"], "owner": g["owner"]} for g in group]
        rows.append(build_row(
            inventory_row_id=dedup_key,
            oracle=next(iter(oracles)),
            kind=next(iter(kinds)),
            universal=next(iter(universal_flags)),
            members=members,
        ))

    seen_ids = {r["inventory_row_id"] for r in rows}
    for gap in passthrough:
        inventory_row_id = gap["capability"]
        assert inventory_row_id not in seen_ids, (
            f"passthrough capability {inventory_row_id!r} collides with an "
            "existing inventory_row_id"
        )
        seen_ids.add(inventory_row_id)
        members = [{"capability": gap["capability"], "owner": gap["owner"]}]
        rows.append(build_row(
            inventory_row_id=inventory_row_id,
            oracle=gap["oracle"],
            kind=gap["kind"],
            universal=is_universal(gap),
            members=members,
        ))

    assert len(rows) == 725, f"expected 725 output rows, got {len(rows)}"
    return rows, len(merged_groups), len(passthrough)


def summarize(rows: list, merged_count: int, passthrough_count: int, diff: dict) -> dict:
    rows_by_oracle: dict = {"python": 0, "cli": 0}
    rows_by_method: dict = {}
    rows_by_kind: dict = {}
    for row in rows:
        rows_by_oracle[row["oracle"]] = rows_by_oracle.get(row["oracle"], 0) + 1
        method_key = row["method"] if row["method"] is not None else "none"
        rows_by_method[method_key] = rows_by_method.get(method_key, 0) + 1
        rows_by_kind[row["kind"]] = rows_by_kind.get(row["kind"], 0) + 1

    distinct_parameter_count = sum(
        1 for r in rows if r["inventory_row_id"].startswith("param:")
    )
    distinct_flag_count = sum(
        1 for r in rows if r["inventory_row_id"].startswith("flag:")
    )
    universal_row_count = sum(1 for r in rows if r["universal"])

    discrepancies = []
    if distinct_parameter_count != diff["distinct_parameter_count"]:
        discrepancies.append(
            f"distinct_parameter_count: matrix={distinct_parameter_count} "
            f"vs capability_diff.json={diff['distinct_parameter_count']}"
        )
    if distinct_flag_count != diff["distinct_flag_count"]:
        discrepancies.append(
            f"distinct_flag_count: matrix={distinct_flag_count} "
            f"vs capability_diff.json={diff['distinct_flag_count']}"
        )
    if universal_row_count != len(diff["universal_flags"]):
        discrepancies.append(
            f"universal row count: matrix={universal_row_count} "
            f"vs capability_diff.json universal_flags={len(diff['universal_flags'])}"
        )

    return {
        "row_count": len(rows),
        "merged_row_count": merged_count,
        "passthrough_row_count": passthrough_count,
        "rows_by_oracle": rows_by_oracle,
        "rows_by_method": rows_by_method,
        "rows_by_kind": rows_by_kind,
        "discrepancies": discrepancies,
    }


def main() -> int:
    diff = json.loads(INPUT_PATH.read_text())
    rows, merged_count, passthrough_count = build_matrix(diff)
    summary = summarize(rows, merged_count, passthrough_count, diff)

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")
    print(json.dumps(summary, indent=2))
    if summary["discrepancies"]:
        print("DISCREPANCIES:", file=sys.stderr)
        for d in summary["discrepancies"]:
            print(f"  - {d}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
