# Active Plan
<!-- approved: 2026-08-06 -->
<!-- gate-iterations: 1 -->
<!-- user-approved: yes -->
<!-- status: in-progress -->

Follow-up triage batch (catboost-8z4.83, .84, .85 -- all filed during the
parity-cleanup-followups-2 batch's task/final reviews, none blocking) on
phase-0 (8fafcb7). Plan file:
.superpowers/sdd/2026-08-06-followup-triage-review-plan.md.

Gate: task-specific -- catboost-8z4.83/.84 need R CMD INSTALL --preclean
clean + full testthat green; catboost-8z4.85 needs tools/parity Python
suite green + full R testthat suite green + byte-identical fixture diffs.
vendor/catboost/ untouched throughout.

Execution order (independent, no bd blocking edges, numeric order):

1. catboost-8z4.83 -- investigate and fix drop_unused_features's dead ntree_end/ntree_start args
2. catboost-8z4.84 -- determine whether default prediction_type divergence is real parity debt
3. catboost-8z4.85 -- dedupe duplicated dataset/param preamble across gen_*_classes_fixture.py scripts

Plan review gate history: 1 iteration, 3/3 PASS (no revision needed).

Prior batch: parity-cleanup-followups-2 (catboost-8z4.73-.82, also no
phase number) merged to phase-0, gate met (10/10 tasks, final whole-branch
review 0 Critical/Important). These 3 tickets are its own follow-ups.
Epic catboost-8z4's own history is in git log / bd show -- not restated
here.
