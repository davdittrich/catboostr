# tools/parity — capability inventory (spec §4.5)

Generates the machine-derived Python/CLI/R surface inventory that the design
spec (`docs/superpowers/specs/2026-07-30-catboostr-design.md` §4.5) commissions,
and diffs it against R and against the spec's own hand-written §2 claim list.
Nothing in this directory is hand-curated: every row traces to introspection
or `--help` parsing, not to a person's memory of what CatBoost does.

## Pipeline order

Each stage reads the previous stage's fixture and writes its own under
`tests/fixtures/parity/`. Run in this order (or via `run.sh`, below):

1. **`parse_namespace.py`** — reads the pinned upstream checkout at
   `vendor/catboost` (acquire it first via `tools/vendor/acquire.sh` if
   absent) and the fork's own `R-package/NAMESPACE`, and records every
   `export()` and registered S3 method plus the upstream tag/SHA actually
   used. Writes `tests/fixtures/parity/r_surface.json`.
2. **`introspect_python.py`** — introspects the installed `catboost` Python
   package (must run inside the pinned oracle venv: `uv run --frozen --project
   tools/oracle python tools/parity/introspect_python.py`). Every public
   module member, class method, property, enum member, and training
   parameter. Writes `tests/fixtures/parity/python_surface.json`.
3. **`enumerate_cli.py`** — parses `--help` output from the pinned CLI binary
   (`tools/oracle/cli/bin/catboost-v1.2.10`; acquire via
   `tools/oracle/cli/acquire.sh` if absent). Every mode, submode, and flag.
   Writes `tests/fixtures/parity/cli_surface.json`.
4. **`compute_diff.py`** — diffs Python ∪ CLI against R: every capability
   present in Python or CLI and absent from R becomes a gap row. Also reports
   deduplicated counts (`distinct_flag_count`, `distinct_parameter_count`,
   `gap_count_deduplicated`) because the raw row count double-counts a flag
   or parameter once per CLI mode/submode it appears in, and flags universal
   non-capability flags (`--help`, `--svnrevision`) rather than dropping
   them. Writes `tests/fixtures/parity/capability_diff.json`.
5. **`spec_crosscheck.py`** — cross-checks the spec's hand-written §2 claim
   list against `capability_diff.json`'s gap rows using exact whole-token
   matching (not substring matching — a bare substring match over-suppresses
   real gaps in the direction that flatters the hand-written list). Writes
   `tests/fixtures/parity/spec_crosscheck.json`.

Stage 1 needs `vendor/catboost`; stages 2-3 need their respective oracle
binaries acquired first (see `tools/vendor/acquire.sh`,
`tools/oracle/cli/acquire.sh`). Stages 4-5 need only the fixtures the earlier
stages already wrote.

## Regenerating everything

```
./tools/parity/run.sh
```

## What it found (measured, catboost-8z4.5)

- **1537** raw gap rows (719 Python-only, 818 CLI-only, 48 covered) —
  **inflated ~2x**: 800 CLI flag rows collapse to **224** distinct
  alias-sets (a flag repeated once per mode/submode it appears in), and 375
  Python parameter rows collapse to **139** distinct names (a parameter
  repeated once per class it appears on). Deduplicated total: **725** gap
  rows. `--help` and `--svnrevision` are universal CLI flags present in
  every mode's own `--help` output, not capability gaps — `compute_diff.py`
  marks them `"universal": true` rather than deleting the rows.
- `spec_crosscheck.py`: of the spec's 18 hand-written §2 claims, 17 are
  confirmed present in the generated inventory; 1 (`sparse/CSR pool input`)
  is not found by name-matching against the inventory (annotated in spec
  §2). Of the **raw** (non-deduplicated) 1537 gap rows, 1306 do not
  token-match anything in the spec's hand-written list — this is the entire
  point of §4.5: the hand-written list undercounts, and the generated
  inventory is what's authoritative. (`spec_crosscheck.py` runs against the
  raw gap list, not the deduplicated count, so this number and the 725
  deduplicated figure above are not directly comparable.)
