# Active Plan
<!-- approved: 2026-08-03 -->
<!-- gate-iterations: 3 -->
<!-- user-approved: yes -->
<!-- status: merged-to-phase-0 -->

Phase 5 (Training-control parity, spec line 639) merged to phase-0 at
a5272a0 (merge commit; whole-branch fix wave on top of task work). All 7
tasks complete, gate condition (differential tests green) met: 120/139
generated parameter matrix rows green with independently-verified
coverage, 19 red with documented blockers (GPU-only, native rejection, or
no R equivalent). Merged-result rebuild and full test suite re-verified
green (only the 2 pre-existing documented skips). Worktree and branch
removed. Plan file:
docs/superpowers/plans/2026-08-03-phase5-training-control-parity.md. SDD
ledger: .superpowers/sdd/2026-08-03-phase5-training-control-parity/progress.md.
Execution order was (serial, P5.1 then P5.5 share catboost.train/
catboost.cv — enforced by a beads blocks dependency, catboost-8z4.62
depends on catboost-8z4.58):

- catboost-8z4.58 (P5.1) — init_model support (continue training)
- catboost-8z4.59 (P5.2) — grid_search / randomized_search
- catboost-8z4.60 (P5.3) — select_features
- catboost-8z4.61 (P5.4) — virtual ensembles verified end-to-end
- catboost-8z4.62 (P5.5) — generated parameter documentation, validation,
  and family-level differential closure of all 139 param:* matrix rows
- catboost-8z4.63 (P5.6) — CLI metadata mode parity
- catboost-8z4.64 (P5.7) — CLI normalize-model mode parity

Follow-up tickets filed during Phase 5, none blocking: catboost-8z4.65
(outbound params JSON truncation in catboost.train/cv), catboost-8z4.66
(Phase 2 inventory pipeline gap), catboost-8z4.67 (minor regex-robustness
gap in the whitelist generator).

Prior phase: Phase 4 (Analysis parity) merged to phase-0 at 1acd501, 7/8
gate items green, mode:model-based-eval stays red (GPU-only, documented).
Follow-up ticket catboost-jpp (deferred Minor findings) still open, not
blocking.
