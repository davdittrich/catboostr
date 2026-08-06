# Follow-up triage (catboost-8z4.83, .84, .85) — implementation plan

Epic: catboost-8z4. Base branch: phase-0 (8fafcb7, parity-cleanup-followups-2 merge).

## Global Constraints

- **Mechanism:** direct-fix/investigation tickets, follow-ups filed during the parity-cleanup-followups-2 batch's task and final whole-branch reviews. No new R public API surface beyond what's explicitly scoped per ticket (catboost-8z4.83 may change one existing function's signature; catboost-8z4.84 may add roxygen docs or, if a genuine gap is confirmed, a scoped fix; catboost-8z4.85 is a pure internal refactor, zero public API change).
- **Forbidden:** no touching vendor/catboost/ (read-only); no fabricated parity-matrix rows (spec §4.4); no re-litigating any pre-existing matrix row's state; no scope creep beyond each ticket's named investigation/fix.
- **Audit:** each ticket is investigation-first (read real source before deciding fix direction) — no guessing at behavior. catboost-8z4.83/.84 both require reading actual C++/Python source before choosing a fix vs. document-only direction. catboost-8z4.85 requires byte-identical fixture diffs after refactor (regression check, not behavioral trust).
- **Dependency:** none — all 3 tickets are independent (no bd blocking edges). Execution order below is arbitrary/numeric, not a dependency chain.
- **Verification method, task-specific:**
  - catboost-8z4.83 (R/catboost.R, src/catboostr.cpp possibly): R CMD INSTALL --preclean . + full testthat suite green.
  - catboost-8z4.84 (R/catboost.R, investigation + doc or scoped fix): R CMD INSTALL --preclean . + full testthat suite green.
  - catboost-8z4.85 (tools/oracle/*.py, pure refactor): tools/parity Python suite green + full R testthat suite green (fixtures are R-test-consumed) + byte-identical fixture diff check.
- vendor/catboost/ untouched in all 3 tasks.

## Execution Order

1. catboost-8z4.83 — investigate and fix drop_unused_features's dead ntree_end/ntree_start args
2. catboost-8z4.84 — determine whether default prediction_type divergence is real parity debt
3. catboost-8z4.85 — dedupe duplicated dataset/param preamble across gen_*_classes_fixture.py scripts

## Tickets (verbatim, hermetic — see `bd show <id>` for full body)

- catboost-8z4.83
- catboost-8z4.84
- catboost-8z4.85
