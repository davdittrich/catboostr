# Plan: Phase 8 -- distributed training parity (catboost-8z4.102/.103/.104)

**Mechanism:** Wire CatBoost's existing `run-worker` CLI mode and the
already-partially-supported `node_type`/`node_port`/`file_with_hosts`
master-side params through to a new R binding (`catboost.run_worker()`),
then prove the distributed path with a live worker+master differential
test, then document it. No new build/framework choice -- upstream's
distributed protocol (`NPar`/`library/cpp/par`) is fixed, read-only vendor
code; this plan only adds an R-facing entry point to a capability that
already exists in the vendored C++.

**Forbidden:** No CRAN-facing "distributed cluster orchestration" feature
(e.g. auto-provisioning workers, a cluster-manager integration) --
strictly the CLI's existing manual process-per-host model, mirrored 1:1 in
R. No reuse of Phase 6's `TRCallbackBridge`/`TRCustomCallbackWiring`
(RunWorker makes zero R callbacks). No new package dependency
(`processx` etc.) -- base R (`system2`, `tools::pskill`,
`socketConnection`) covers process spawn/kill and free-port selection.

**Audit:** Each ticket's own DoD: P8.1's background-process-startable
smoke test; P8.2's live worker+master differential test with guaranteed
teardown (verified once by deliberately breaking an assertion mid-
development and confirming no orphan process survives); P8.3's zero-
internal-ticket-ID grep over newly-touched `.Rd` files (same pattern
catboost-8z4.101 established) plus full suite green after each ticket.

## Scope basis

Parity matrix (`tests/fixtures/parity/matrix.dispositioned.json`) has
**4** red rows touching distributed training (corrected 2026-08-08 after
a plan-review-gate Scope reviewer caught a 3-row undercount in an earlier
draft): `mode:run-worker` (roundtrip, confidence 35, "no single-node
comparable output established yet"), `flag:--node-type` (`fit` +
`select-features`), `flag:--node-port` (`fit` + `select-features` +
`run-worker`), `flag:--file-with-hosts` (`fit` + `select-features`).
`node_type`/`node_port`/`file_with_hosts` are ALREADY in R/catboost.R's
generated known-params allowlist (`R/catboost.R:3077-3092`) and pass
through generically as JSON -- confirmed by grep, no code change needed
for the master side's param plumbing. The only missing capability is the
worker side, which has zero R binding today.

## Execution order

Strictly sequential (each depends on the prior being merged/present):

1. `catboost-8z4.102` (P8.1) -- add `catboost.run_worker()` binding
   (`src/catboostr.cpp` new `.Call` entry + registration + R wrapper).
2. `catboost-8z4.103` (P8.2) -- differential test AGAINST THE REAL CLI
   BINARY (`tools/oracle/cli/bin/catboost-v1.2.10`), per the spec's
   literal "Differential test against the CLI" gate text (an R-vs-R-only
   draft of this ticket was FAILED by a plan-review-gate Completeness
   reviewer 2026-08-08 for not driving the real CLI). The binary is
   gitignored/per-checkout-acquired (`tools/oracle/cli/acquire.sh`, same
   model as `vendor/catboost`) -- absent in this worktree, present in the
   main checkout; the ticket acquires it if missing rather than assuming
   presence (a plan-review-gate Feasibility reviewer caught an earlier
   draft's false "confirmed present" claim 2026-08-08). Two assertions
   against the same live CLI-binary worker: `catboost.train` (closes
   `mode:run-worker` + the `fit` members of the 3 flag families) and
   `catboost.select_features` (closes the `select-features` members).
   Closes all 4 rows via this project's established
   `closure_overlay.json` + `apply_disposition.py` mechanism
   (catboost-8z4.73) -- a plan-review-gate Completeness reviewer caught an
   earlier draft omitting the matrix-file update entirely 2026-08-08.
   This is the ticket that resolves the matrix's "no single-node
   comparable output established yet" gap.
3. `catboost-8z4.104` (P8.3) -- roxygen docs for the new function and the
   master-side distributed params, zero internal ticket-ID leakage.

Each: dispatch implementer -> task review -> fix loop (SDD process, same
as the three prior batches on this branch). Full suite must stay green
after each (`[ FAIL 0 | WARN 0 | SKIP 2 | PASS 823 ]` baseline).

## Task 1: catboost-8z4.102 (P8.1)

Full spec: `bd show catboost-8z4.102`. No dependencies.

## Task 2: catboost-8z4.103 (P8.2)

Full spec: `bd show catboost-8z4.103`. BLOCKED until Task 1
(catboost-8z4.102) complete and reviewed clean.

## Task 3: catboost-8z4.104 (P8.3)

Full spec: `bd show catboost-8z4.104`. BLOCKED until Task 2
(catboost-8z4.103) complete and reviewed clean.

## Alternatives considered

* **Static CLI-oracle fixture (`tools/oracle/cli/` pattern)** instead of a
  live CLI-binary worker process in P8.2 -- REJECTED for the fixture
  MECHANISM only, not for using the CLI binary at all: that pattern
  captures one-shot input->output CLI invocations as committed JSON;
  distributed training requires a live worker running concurrently with
  training, which cannot be pre-captured. P8.2 still drives the real CLI
  binary (`tools/oracle/cli/bin/catboost-v1.2.10`), just as a live
  background process (`catboost-v1.2.10 run-worker`) rather than a static
  fixture -- this satisfies the spec's literal "Differential test against
  the CLI" gate text.
* **`processx` package for worker process management** -- REJECTED: base
  R's `system2(..., wait = FALSE)` + `tools::pskill()` covers spawn/kill
  with zero new dependencies; `processx` would add a `Suggests` entry for
  functionality base R already provides at this scope (single background
  process, no streaming I/O needs).
* **Reusing Phase 6's callback-bridge machinery for the worker binding** --
  REJECTED: `RunWorker` makes no R callbacks; it is a plain blocking C++
  call, not a mechanism that needs marshaling C++ threads back onto R's
  main thread.
