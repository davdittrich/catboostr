# Plan: catboost-8z4.100 + catboost-8z4.101

**Mechanism:** Execute 2 already-hermetic, already-validated beads tickets (filed after the phase6-followups batch's final whole-batch review) via subagent-driven-development, same process as the two prior batches on this branch. No new planning content per ticket -- each is self-contained.

**Forbidden:** No new abstractions beyond what each ticket specifies. catboost-8z4.101 must not touch plain `#` code comments (internal-only, never exported to `.Rd`) -- only roxygen `#'` blocks.

**Audit:** Each ticket's own DoD (full test suite green, no `vendor/` edits, .100's new tests, .101's zero-remaining-ticket-ID grep check).

## Execution order

Both independent (no functional dependency), sequenced only to avoid same-file (`R/catboost.R`) collision:

1. `catboost-8z4.100` -- add params validation to `catboost.eval_feature`/`catboost.model_based_eval` (code + 4 new tests: unknown-key and duplicate-key case for each function)
2. `catboost-8z4.101` -- strip internal ticket-ID references from `man/*.Rd`-facing roxygen text (docs-only, no test changes expected)

Each: dispatch implementer -> task review -> fix loop (SDD process). Full suite must stay green after each (`[ FAIL 0 | WARN 0 | SKIP 2 | PASS 819 ]` baseline; expect +4 after .100, unchanged after .101).
