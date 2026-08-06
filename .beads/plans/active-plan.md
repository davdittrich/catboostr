# Active Plan
<!-- approved: 2026-08-07 -->
<!-- gate-iterations: 1 -->
<!-- user-approved: yes -->
<!-- status: in-progress -->

roxygen2 DLL-reload investigation (catboost-8z4.86 -- filed during the
parity-cleanup-followups-2 batch's task review, non-blocking) on phase-0
(e0cdcbc). Plan file:
.superpowers/sdd/2026-08-07-roxygen2-dll-reload-review-plan.md.

Note: the ticket's original filing mis-diagnosed the cause as an "OpenSSL
environment-variable issue." The controller independently reproduced the
failure before rewriting the ticket and found this diagnosis wrong: the
full rebuild (~15 min, including the pinned OpenSSL static build)
completes successfully; the real failure is
`Error in getDLLRegisteredRoutines.DLLInfo(dll, addNames = FALSE) : must
specify DLL via a "DLLInfo" object`, occurring AFTER install, during
roxygen2/pkgload's native-routine introspection step
(`assignNativeRoutines -> getDLLRegisteredRoutines.DLLInfo`). Feasibility
reviewer additionally found a corroborating clue: the generated NAMESPACE
has no `useDynLib` directive despite `R/catboost.R:7`'s roxygen tag
(`@useDynLib libcatboostr, .registration = TRUE`) -- worth checking first
in execution.

Gate: dev-tooling only, no shipped-behavior change expected. Verification:
regenerate at least one real .Rd file via the fixed/documented path and
diff against the currently-correct, hand-edited catboost.drop_unused_features.Rd
(from catboost-8z4.83) to confirm the fix/workaround actually works.
vendor/catboost/ untouched.

Execution order: single ticket, no dependencies.

1. catboost-8z4.86 -- investigate and fix/document-workaround for roxygen2's post-install DLL-reload failure

Plan review gate history: 1 iteration, 3/3 PASS (no revision needed).

Prior batch: followup-triage (catboost-8z4.83-.85) merged to phase-0, gate
met (3/3 tasks, final whole-branch review found and fixed 1 Important
finding -- stale line citations). Epic catboost-8z4's own history is in
git log / bd show -- not restated here.
