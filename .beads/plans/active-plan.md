# Active Plan
<!-- approved: 2026-08-08 -->
<!-- gate-iterations: 2 -->
<!-- user-approved: yes -->
<!-- status: complete-not-merged -->

catboost-8z4.100 + catboost-8z4.101. Plan file:
docs/superpowers/plans/2026-08-08-post-followups-cleanup.md. Worktree:
.claude/worktrees/phase6-custom-loss-metric, branch
worktree-phase6-custom-loss-metric.

Execution order (both closed):
1. catboost-8z4.100 (params validation for eval_feature/model_based_eval)
   -- commit 2b8b4b4
2. catboost-8z4.101 (strip internal ticket IDs from CRAN-facing man/ docs)
   -- commit 198c1a8

Plan review gate: APPROVED iteration 2/3 (iteration 1 Completeness FAIL --
catboost-8z4.101's verification grep pattern false-positived on legitimate
@seealso \url{} doc-site links; fixed by tightening the ticket's own
Reference Data/DoD pattern to catboost-8z4\.[0-9]+|catboost-upw, verified
7/7 real matches, 0/6 false positives; re-validated).

Final whole-batch review (2026-08-08, opus): APPROVED, no findings.

Full suite as of last commit: [ FAIL 0 | WARN 0 | SKIP 2 | PASS 823 ].

--- Prior batches on this branch (complete, not merged) ---
phase6-followups (catboost-8z4.95-.99 + catboost-upw), commits
8974a29..50822d1. Final whole-batch review (opus): NEEDS_FIXES -> fixed
(01cd939) -> PASS. Full suite: PASS 819.
Phase 6 (catboost-8z4.88-.92/.93/.94/.91), commits e79671c..034ccd5. Final
whole-branch review (opus): APPROVED. Full suite: PASS 816.
None of the three batches yet merged to phase-0 -- awaiting user go-ahead.

Prior-prior batches (roxygen2-dll-reload, followup-triage) merged to
phase-0; history is in git log / bd show, not restated here.
