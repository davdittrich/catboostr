# Active Plan
<!-- approved: 2026-08-06 -->
<!-- gate-iterations: 2 -->
<!-- user-approved: yes -->
<!-- status: merged-to-phase-0 -->

Parity-cleanup follow-ups (batch 2, follow-ups from the parity debt cleanup
batch) merged to phase-0 (fast-forward, e1eb88d..3b0eca1, 13 commits). All
10 tasks complete, final whole-branch review verdict: ready to merge (0
Critical, 0 Important, several Minor/Nit -- one addressed directly by the
controller, rest filed as follow-ups or left as informational). Merged-result
rebuild and full test suite re-verified green (R: 734 PASS, 0 FAIL, 2
pre-existing documented skips; tools/parity: 47/47). Worktree and branch
removed. Plan file:
docs/superpowers/plans/2026-08-06-parity-debt-cleanup-followups-2.md.

Note: this is NOT spec Phase 6 (custom R loss/metric callback bridge) or
spec Phase 7 (GPU parity) -- those real numbered phases (docs/superpowers/
specs/2026-07-30-catboostr-design.md §5 phase table) are untouched and still
provisional/not started. This batch is unplanned debt cleanup found during
real Phase 5/6 work; it carries no phase number.

Gate: task-specific -- catboost-8z4.73 needed tools/parity Python suite
green only; catboost-8z4.74/.75/.76-.82 needed R CMD INSTALL --preclean
clean + full testthat green (and .76-.82 also needed tools/parity Python
suite green); vendor/catboost/ untouched throughout. Gate met.

Execution order (all complete, bd blocks: .76-.82 all depended on .73):

1. catboost-8z4.73 -- fix/retire stale test_apply_disposition.py self-consistency assertion (closure-overlay mechanism introduced)
2. catboost-8z4.74 -- CatBoostPoolNumTrees_R exports/init.c registration mismatch
3. catboost-8z4.75 -- catboost.save_model toJSON digits=4 truncation (same class as .65)
4. catboost-8z4.76 -- flag:--cv row (left red, genuine mechanism mismatch, documented)
5. catboost-8z4.77 -- mode:eval-metrics row closure
6. catboost-8z4.78 -- training entry points (catboost.cv/sum_models/train)
7. catboost-8z4.79 -- eval_metrics method x4 classes
8. catboost-8z4.80 -- feature introspection x3 methods x4 classes (12 rows; 1 fix round)
9. catboost-8z4.81 -- prediction family x4 methods x4 classes (16 rows; 1 fix round)
10. catboost-8z4.82 -- persistence (load_model/save_model) x4 classes (8 rows)

catboost-8z4.76-.82 are the atomic per-capability split (spec §4.5, per
no-bundling rule) of the previously-bundled catboost-8z4.72, now CLOSED
pointing at these 7.

Plan review gate history: 2 iterations. Iter 1: Feasibility PASS,
Completeness PASS, Scope FAIL (catboost-8z4.72 bundled parameter/flag AND
method/mode-shaped rows in one ticket, violating spec §4.5) -- fixed by
force-closing catboost-8z4.72 and filing catboost-8z4.76-.82. Iter 2: 3/3
PASS.

Follow-up tickets filed during this batch, none blocking: catboost-8z4.83
(R's drop_unused_features has dead ntree_end/ntree_start args Python's
equivalent doesn't have), catboost-8z4.84 (default prediction_type differs
between R's catboost.predict() and Python's per-subclass defaults),
catboost-8z4.85 (dedupe ~6x duplicated dataset/param preamble across
tools/oracle/gen_*_classes_fixture.py scripts).

Final matrix state: 770 rows total (unchanged count), 220 green / 550 red
(up from 176 green / 594 red at batch start) -- 44 of the 45 newly-surfaced
rows closed green with real differential tests, 1 (flag:--cv) left red with
a documented mechanism-level reason per spec §4.4's no-fabrication rule.

Prior batch: Parity debt cleanup (catboost-8z4.65-.71, also no phase
number) merged to phase-0, gate met (7/7 tasks, final whole-branch review 0
Critical/Important, 6 Minor, one fix wave). Epic catboost-8z4's own history
is in git log / bd show -- not restated here.
