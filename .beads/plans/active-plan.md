# Active Plan
<!-- approved: 2026-08-08 -->
<!-- gate-iterations: 2 -->
<!-- user-approved: yes -->
<!-- status: complete-not-merged -->

Phase 6 follow-ups: catboost-8z4.95-.99 + catboost-upw. Plan file:
docs/superpowers/plans/2026-08-08-phase6-followups.md. Worktree:
.claude/worktrees/phase6-custom-loss-metric, branch
worktree-phase6-custom-loss-metric.

Execution order (all closed):
1. catboost-8z4.98 (duplicate-params-key detection) -- commit 8974a29
2. catboost-8z4.97 (helper-call ordering fix) -- commit 1d87fae
3. catboost-8z4.96 (interrupt-deviation docs) -- commit 316532d, closes
   catboost-upw via its own mandated bd comment+close step
4. catboost-8z4.99 (test fixture dedup) -- commit 242d261
5. catboost-8z4.95 (C++ .Call-site dedupe) -- commit be2b786

Plan review gate: APPROVED iteration 2/3 (iteration 1 Completeness FAIL --
catboost-upw closure was plan-level prose only, fixed by adding a mandatory
bd comment/bd close step to catboost-8z4.96's own ticket body; re-validated).

Final whole-batch review (2026-08-08, opus): NEEDS_FIXES -> fixed -> PASS.
Found a load-bearing gap in catboost-8z4.98: the duplicate-key check ran
too late for 24 process_synonyms alias-group names (iterations, depth,
loss_function, ...), silently collapsing a duplicate instead of erroring.
Fix commit 01cd939 (hoisted the check to the top of process_synonyms(),
before alias resolution runs); scoped re-review PASS, no findings.
3 non-blocking observations filed as follow-up tickets: catboost-8z4.100
(eval_feature/model_based_eval skip params validation entirely),
catboost-8z4.101 (pre-CRAN sweep to strip internal ticket IDs from man/).

Full suite as of last commit: [ FAIL 0 | WARN 0 | SKIP 2 | PASS 819 ].

Not yet merged to phase-0 -- awaiting user go-ahead.

--- Phase 6 (previous batch on this branch, complete, not merged) ---
catboost-8z4.88-.92/.93/.94/.91, commits e79671c..034ccd5. Final whole-branch
review (2026-08-08, opus): APPROVED. Full suite: [ FAIL 0 | WARN 0 | SKIP 2 |
PASS 816 ]. Not yet merged to phase-0 -- awaiting user go-ahead.

Prior batches (roxygen2-dll-reload, followup-triage) merged to phase-0;
history is in git log / bd show, not restated here.
