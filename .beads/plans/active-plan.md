# Active Plan
<!-- approved: 2026-08-06 -->
<!-- gate-iterations: 1 -->
<!-- user-approved: yes -->
<!-- status: merged-to-phase-0 -->

Follow-up triage batch (catboost-8z4.83, .84, .85) merged to phase-0
(fast-forward, 8158156..648262b, 5 commits). All 3 tasks complete, final
whole-branch review found 1 Important finding (14 stale R/catboost.R line
citations in comments, caused by earlier line shifts within this same
batch) -- fixed and re-reviewed clean. Merged-result rebuild and full test
suite re-verified green (R: 736 PASS, 0 FAIL, 2 pre-existing documented
skips; tools/parity: 47/47). Worktree and branch removed. Plan file:
docs/superpowers/plans/2026-08-06-followup-triage.md.

Outcomes:
1. catboost-8z4.83 -- removed dead ntree_end/ntree_start args from
   catboost.drop_unused_features (no native tree-range support exists);
   now matches Python's argument-free signature.
2. catboost-8z4.84 -- investigated R-vs-Python default prediction_type
   divergence; documented as intentional API difference (R has no
   per-subclass hook), not a behavior fix.
3. catboost-8z4.85 -- extracted tools/oracle/_classes_common.py, deduping
   dataset/param preamble across 7 fixture-generator scripts (pure
   refactor, all fixtures verified byte-identical).

Follow-up ticket filed during this batch, non-blocking: catboost-8z4.86
(roxygen2/OpenSSL build-environment issue that forced a hand-edited .Rd
file in task 1).

Plan review gate history: 1 iteration, 3/3 PASS (no revision needed).

Prior batch: parity-cleanup-followups-2 (catboost-8z4.73-.82) merged to
phase-0, gate met (10/10 tasks). Epic catboost-8z4's own history is in git
log / bd show -- not restated here.
