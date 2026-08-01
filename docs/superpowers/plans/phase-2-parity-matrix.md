# Plan: Phase 2 — Parity Matrix Foundation

Pointer plan. Each task's full spec lives in its beads ticket (hermetic
6-section format) — the implementer must run `bd show <id>` as its first
action and treat that as the sole source of requirements. Do not paste
ticket bodies into implementer dispatches; the brief below is a pointer,
not a copy.

## Global Constraints

- `vendor/catboost/` is a pinned read-only snapshot. Never edit any file
  under it. Every task must end with `git -C vendor/catboost status
  --porcelain` empty.
- Never run a writing git command as the implementer — the controller
  (this session) commits. (Note: per repo's standard SDD flow the
  implementer subagent does commit its own task's work locally in the
  worktree branch; "controller commits" here refers to not touching
  vendor/ or pushing/merging — resolved in favor of each ticket's own
  INVIOLABLE list, which is the authoritative text.)
- Inline `python3 -c '...'` may be blocked in this environment — write
  scripts to a `.py` file and run that file instead if so.
- No Phase 3+ epic or ticket may be created by any task in this plan
  (explicit INVIOLABLE in catboost-8z4.36 / P2.3) — that requires the
  user's own sign-off after this plan's work lands.
- Task 4 (catboost-8z4.36 / P2.3) is BLOCKED until Tasks 1 and 2
  (catboost-8z4.34 / P2.1, catboost-8z4.35 / P2.2) are both complete and
  reviewed clean — do not dispatch it earlier.

## Task 1: catboost-8z4.34 (P2.1) — Build the parity matrix join tool

Full spec: `bd show catboost-8z4.34`. No dependencies.

## Task 2: catboost-8z4.35 (P2.2) — Structural-output serializer and field mapping

Full spec: `bd show catboost-8z4.35`. No dependencies.

## Task 3: catboost-8z4.37 (P2.4) — Root-cause the multi-target and get_object_importance breakage reports

Full spec: `bd show catboost-8z4.37`. No dependencies (independent of
Tasks 1/2/4 — uses `tools/oracle/`'s pinned Python environment only).

## Task 4: catboost-8z4.36 (P2.3) — Apply the bulk-disposition rule and get explicit sign-off

Full spec: `bd show catboost-8z4.36`. BLOCKED on Tasks 1 and 2
(`tests/fixtures/parity/matrix.json` from Task 1, field-mapping mechanism
from Task 2). Do not dispatch before both are complete.
