# Active Plan
<!-- approved: 2026-07-30 -->
<!-- gate-iterations: 3 -->
<!-- user-approved: pending -->
<!-- status: in-progress -->

Epic: catboost-8z4

Exec order (Phase 0):
1. catboost-8z4.1, catboost-8z4.2, catboost-8z4.3, catboost-8z4.4 — parallel, disjoint write scopes
2. catboost-8z4.5 — blocked by .3 and .4

Spec: docs/superpowers/specs/2026-07-30-catboostr-design.md
Recovery: bd prime --work-type recovery, then bd show on the IDs above.
