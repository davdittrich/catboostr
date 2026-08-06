# Active Plan
<!-- approved: 2026-08-06 -->
<!-- gate-iterations: 2 -->
<!-- user-approved: yes -->
<!-- status: in-progress -->

Phase 7 (Parity-cleanup follow-ups, from Phase 6) executing on phase-0 via
subagent-driven-development. Plan file:
docs/superpowers/plans/2026-08-06-phase7-parity-cleanup-followups.md. SDD
ledger: .superpowers/sdd/2026-08-06-phase7-parity-cleanup-followups/progress.md.

Gate: task-specific — catboost-8z4.73 needs tools/parity Python suite green
only; catboost-8z4.74/.75/.76-.82 need R CMD INSTALL --preclean clean + full
testthat green (and .76-.82 also need tools/parity Python suite green);
vendor/catboost/ untouched throughout.

Execution order (bd blocks: .76-.82 all depend on .73):

1. catboost-8z4.73 — fix/retire stale test_apply_disposition.py self-consistency assertion
2. catboost-8z4.74 — CatBoostPoolNumTrees_R exports/init.c registration mismatch
3. catboost-8z4.75 — catboost.save_model toJSON digits=4 truncation (same class as .65)
4. catboost-8z4.76 — flag:--cv row closure (bulk parameter/flag family)
5. catboost-8z4.77 — mode:eval-metrics row closure
6. catboost-8z4.78 — training entry points (catboost.cv/sum_models/train)
7. catboost-8z4.79 — eval_metrics method x4 classes
8. catboost-8z4.80 — feature introspection x3 methods x4 classes (12 rows)
9. catboost-8z4.81 — prediction family x4 methods x4 classes (16 rows)
10. catboost-8z4.82 — persistence (load_model/save_model) x4 classes (8 rows)

catboost-8z4.76-.82 are the atomic per-capability split (spec §4.5, per
no-bundling rule) of the previously-bundled catboost-8z4.72, now CLOSED
pointing at these 7.

Plan review gate history: 2 iterations. Iter 1: Feasibility PASS,
Completeness PASS, Scope FAIL (catboost-8z4.72 bundled parameter/flag AND
method/mode-shaped rows in one ticket, violating spec §4.5) — fixed by
force-closing catboost-8z4.72 and filing catboost-8z4.76-.82. Iter 2: 3/3
PASS, only a trivial "8 vs 7 tickets" wording fix in Global Constraints.

Prior phase: Phase 6 (Parity debt cleanup) merged to phase-0, gate met (7/7
tasks, final whole-branch review 0 Critical/Important, 6 Minor, one fix
wave). Epic catboost-8z4's own history is in git log / bd show — not
restated here.
