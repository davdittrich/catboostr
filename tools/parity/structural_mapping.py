#!/usr/bin/env python3
"""Structural-output canonical-JSON serializer + field-mapping mechanism
(ticket catboost-8z4.35 / P2.2).

Spec ref: docs/superpowers/specs/2026-07-30-catboostr-design.md Sec 4.3 --
"Structural (plot_tree, calc_feature_statistics, progress logs): Both sides
serialize to a canonical JSON form with a field mapping declared per
capability in the parity matrix; comparison is field-by-field, numeric
leaves within tolerance ... a capability without a declared mapping cannot
be marked green."

This module defines the MECHANISM only:
  - to_canonical(): the serialization contract (key ordering, numeric leaf
    representation, NaN/Inf handling, nested-structure policy).
  - validate_mapping(): schema-level check that a mapping declaration is
    well-formed (every field has a comparison type; excluded fields have a
    reason).
  - validate_against_documents(): data-aware check that a mapping is
    actually usable against a concrete pair of canonical documents (paths
    resolve on both sides; declared comparison type matches the leaf's
    actual kind). Raises ValueError on a BROKEN mapping -- this is a bug in
    the declaration itself, not a differential-test result.
  - compare(): the field-by-field comparator. Returns (ok, mismatches) --
    this IS the differential-test result (a perturbed value is an expected
    "red", not an exception).

No real per-capability mappings are declared here (out of scope, see
ticket boundary). tools/parity/mappings/example_worked.json is the single
fabricated worked example proving the mechanism end-to-end.

Run self-test: python3 tools/parity/structural_mapping.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import build_matrix as bm  # same directory; reuses the one tolerance policy

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPPINGS_DIR = Path(__file__).resolve().parent / "mappings"

# Spec Sec 4.3: this IS the numeric-row tolerance policy, reused verbatim.
# No separate regime for structural leaves (ticket III + Sec 4.3 explicit).
DEFAULT_TOLERANCE = bm.DEFAULT_TOLERANCE
CLI_PRECISION_FLOOR = bm.CLI_PRECISION_FLOOR

_FLOAT_SPECIALS = {math.nan: "NaN", math.inf: "Infinity", -math.inf: "-Infinity"}


# --------------------------------------------------------------------------
# 1. Canonical-JSON serialization contract
# --------------------------------------------------------------------------
def to_canonical(obj: Any) -> Any:
    """Transform a native Python structure (as decoded from R or oracle
    output) into the canonical form used for comparison.

    Contract (confidence 90 -- mechanical, no ambiguity in the rules below):
      - dict keys: sorted lexicographically, recursively. Canonical string
        form is produced with json.dumps(..., sort_keys=True); key order
        carries no semantic meaning, so sorting makes the serialization
        deterministic regardless of R's/Python's native dict/list ordering.
      - lists: order is KEPT and is semantically significant (e.g. a tree's
        left/right child order, or calc_feature_statistics's per-bin
        sequence). Not sorted, not deduplicated.
      - numeric leaves: kept as Python int/float. NaN/+Inf/-Inf have no
        native JSON literal, so they are wrapped as
        {"__float__": "NaN" | "Infinity" | "-Infinity"} -- a tagged form
        distinguishable from a genuine string leaf, decoded back to a float
        by resolve_path() before comparison.
      - nested/recursive structures (e.g. a tree): kept NESTED, not
        flattened. A mapping field declares a dotted path with "[]" to
        broadcast over a list (see resolve_path), so arbitrarily deep
        recursive structures are addressable without a separate
        flattening pass. This is the design answer to ticket III's
        Anti-pivot guard: JSON's native nesting already represents a tree
        without loss, so no flattening step is required or offered.
      - everything else (str, bool, None, plain int) passes through
        unchanged.
    """
    if isinstance(obj, float):
        if obj in _FLOAT_SPECIALS or obj != obj:  # obj != obj catches nan
            return {"__float__": _FLOAT_SPECIALS.get(obj, "NaN")}
        return obj
    if isinstance(obj, dict):
        return {k: to_canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [to_canonical(v) for v in obj]
    return obj


def canonical_dumps(obj: Any) -> str:
    """Canonical string form -- stable across runs/hosts. allow_nan=False
    because to_canonical() already tags NaN/Inf; a raw float special
    reaching here means to_canonical() was skipped, which is a bug."""
    return json.dumps(to_canonical(obj), sort_keys=True, allow_nan=False)


def _is_float_leaf(value: Any) -> bool:
    return isinstance(value, float) or (
        isinstance(value, dict) and set(value) == {"__float__"}
    )


def _float_value(value: Any) -> float:
    if isinstance(value, dict):
        tag = value["__float__"]
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[tag]
    return value


# --------------------------------------------------------------------------
# 2. Path resolution -- addresses into a canonical document
# --------------------------------------------------------------------------
def resolve_path(doc: Any, path: str) -> list[Any]:
    """Resolve a dotted path into a canonical document. A segment suffixed
    "[]" broadcasts over a list, e.g. "children[].value" walks every
    element of the "children" list and reads "value" off each -- this is
    how a recursive/nested tree is addressed without flattening it.

    Returns the list of resolved leaf values (length 1 for a path with no
    "[]" segments; length N for one broadcast level; nested broadcasts
    flatten left-to-right). Raises KeyError/IndexError/TypeError if the
    path does not resolve -- callers catch this to report "field doesn't
    exist on one side" (ticket III Logic guard (a)).
    """
    contexts = [doc]
    for segment in path.split("."):
        broadcast = segment.endswith("[]")
        key = segment[:-2] if broadcast else segment
        next_contexts: list[Any] = []
        for ctx in contexts:
            value = ctx[key]
            if broadcast:
                next_contexts.extend(value)
            else:
                next_contexts.append(value)
        contexts = next_contexts
    return contexts


# --------------------------------------------------------------------------
# 3. Field-mapping declaration schema + validation
# --------------------------------------------------------------------------
_VALID_COMPARE = {"exact", "numeric"}


def validate_mapping(mapping: dict) -> None:
    """Schema-level check (ticket III Logic guard): every declared,
    non-excluded field must state its comparison type; every excluded
    field must state a reason. Raises ValueError, does not silently
    accept a partial declaration."""
    if "fields" not in mapping or not isinstance(mapping["fields"], list):
        raise ValueError("mapping must have a 'fields' list")
    for i, field in enumerate(mapping["fields"]):
        label = field.get("r_field", f"<fields[{i}]>")
        if field.get("exclude", False):
            if not field.get("exclude_reason"):
                raise ValueError(f"field {label!r}: excluded but no exclude_reason")
            continue
        if not field.get("r_field") or not field.get("oracle_field"):
            raise ValueError(f"field {label!r}: needs both r_field and oracle_field")
        compare = field.get("compare")
        if compare not in _VALID_COMPARE:
            raise ValueError(
                f"field {label!r}: compare must be one of {_VALID_COMPARE}, got {compare!r}"
            )
        if compare == "numeric":
            tol = field.get("tolerance")
            if not tol or "oracle" not in tol or "value" not in tol:
                raise ValueError(
                    f"field {label!r}: compare=numeric requires tolerance {{oracle, value}}"
                )
            oracle = tol["oracle"]
            default = DEFAULT_TOLERANCE.get(oracle)
            if oracle == "cli" and tol["value"] < CLI_PRECISION_FLOOR:
                raise ValueError(
                    f"field {label!r}: cli tolerance {tol['value']} tighter than "
                    f"precision floor {CLI_PRECISION_FLOOR} (Sec 4.3)"
                )
            if default is not None and tol["value"] != default and not field.get(
                "tolerance_justification"
            ):
                raise ValueError(
                    f"field {label!r}: non-default tolerance {tol['value']} "
                    f"(default {default}) needs tolerance_justification"
                )


def validate_against_documents(mapping: dict, r_doc: Any, oracle_doc: Any) -> None:
    """Data-aware check: every declared field pair must actually resolve
    on both canonical documents, and the declared compare type must match
    the leaf's real kind. Raises ValueError on a broken mapping -- this is
    the perturbation target for 'field missing on one side' and 'numeric
    leaf declared exact-match' (ticket step 3)."""
    validate_mapping(mapping)
    for field in mapping["fields"]:
        if field.get("exclude", False):
            continue
        label = field["r_field"]
        try:
            r_values = resolve_path(r_doc, field["r_field"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"field {label!r}: r_field does not resolve on R side: {exc}")
        try:
            o_values = resolve_path(oracle_doc, field["oracle_field"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"field {label!r}: oracle_field does not resolve on oracle side: {exc}"
            )
        if len(r_values) != len(o_values):
            raise ValueError(
                f"field {label!r}: r_field resolves to {len(r_values)} leaves, "
                f"oracle_field to {len(o_values)} -- structurally mismatched"
            )
        if field["compare"] == "exact":
            if any(_is_float_leaf(v) for v in r_values + o_values):
                raise ValueError(
                    f"field {label!r}: numeric leaf declared compare=exact "
                    "(use compare=numeric with a tolerance)"
                )


# --------------------------------------------------------------------------
# 4. Comparator -- the actual differential-test result
# --------------------------------------------------------------------------
def compare(mapping: dict, r_doc: Any, oracle_doc: Any) -> tuple[bool, list[str]]:
    """Field-by-field comparison. Raises ValueError if the mapping itself
    is broken (see validate_against_documents). Otherwise always returns
    (ok, mismatches) -- a perturbed value is a normal red result, not an
    exception."""
    validate_against_documents(mapping, r_doc, oracle_doc)
    r_doc = to_canonical(r_doc)
    oracle_doc = to_canonical(oracle_doc)
    mismatches: list[str] = []
    for field in mapping["fields"]:
        if field.get("exclude", False):
            continue
        label = field["r_field"]
        r_values = resolve_path(r_doc, field["r_field"])
        o_values = resolve_path(oracle_doc, field["oracle_field"])
        for idx, (r_v, o_v) in enumerate(zip(r_values, o_values)):
            if field["compare"] == "exact":
                if r_v != o_v:
                    mismatches.append(f"{label}[{idx}]: exact mismatch {r_v!r} != {o_v!r}")
            else:  # numeric
                r_f, o_f = _float_value(r_v), _float_value(o_v)
                tol = field["tolerance"]["value"]
                if not math.isclose(r_f, o_f, rel_tol=tol):
                    mismatches.append(
                        f"{label}[{idx}]: numeric mismatch {r_f!r} vs {o_f!r} "
                        f"(rel_tol={tol})"
                    )
    return (len(mismatches) == 0, mismatches)


def load_mapping(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Self-test / worked example (ticket step 4): fabricated data only.
# --------------------------------------------------------------------------
def _worked_example() -> None:
    mapping = load_mapping(MAPPINGS_DIR / "example_worked.json")

    r_doc = {
        "feature": "age",
        "bins": [
            {"lower": 0.0, "upper": 10.0, "mean_target": 0.31},
            {"lower": 10.0, "upper": 20.0, "mean_target": 0.42},
        ],
        "generated_at": "2026-08-02T00:00:00Z",  # excluded: non-comparable
    }
    oracle_doc = {
        "feature_name": "age",
        "buckets": [
            {"lo": 0.0, "hi": 10.0, "target_mean": 0.31},
            {"lo": 10.0, "hi": 20.0, "target_mean": 0.42},
        ],
        "generated_at": "2026-08-02T00:00:01Z",
    }

    ok, mismatches = compare(mapping, r_doc, oracle_doc)
    print(f"[match]      ok={ok} mismatches={mismatches}")
    assert ok and not mismatches, "worked example must pass on matching data"

    perturbed = json.loads(json.dumps(oracle_doc))
    perturbed["buckets"][1]["target_mean"] = 0.99  # deliberately wrong
    ok2, mismatches2 = compare(mapping, r_doc, perturbed)
    print(f"[perturbed]  ok={ok2} mismatches={mismatches2}")
    assert not ok2 and mismatches2, "perturbed data must fail comparison"

    # Broken-mapping perturbation (ticket step 3): field missing on one side.
    broken_missing = json.loads(json.dumps(mapping))
    broken_missing["fields"][0]["oracle_field"] = "feature_name_typo"
    try:
        compare(broken_missing, r_doc, oracle_doc)
        raise AssertionError("expected ValueError for missing field")
    except ValueError as exc:
        print(f"[broken: missing field]        raised ValueError: {exc}")

    # Broken-mapping perturbation: numeric leaf declared exact-match.
    broken_exact = json.loads(json.dumps(mapping))
    for f in broken_exact["fields"]:
        if f.get("r_field") == "bins[].mean_target":
            f["compare"] = "exact"
            f.pop("tolerance", None)
    try:
        compare(broken_exact, r_doc, oracle_doc)
        raise AssertionError("expected ValueError for numeric-as-exact")
    except ValueError as exc:
        print(f"[broken: numeric as exact]      raised ValueError: {exc}")


if __name__ == "__main__":
    _worked_example()
    print("OK: structural_mapping self-test passed")
