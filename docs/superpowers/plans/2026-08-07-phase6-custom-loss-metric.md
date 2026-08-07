# Phase 6: Custom R loss/metric callback bridge — implementation plan (v3)

## v2 -> v3: what changed and why

Iteration 2 of plan-review-gate: all 3 reviewers FAIL again, on real, independently-verified findings (not process nitpicks). Fixed here, all corrections applied directly to the 7 tickets (source of truth; this doc summarizes):

1. **`select_features.h` claim was wrong.** v2 claimed it has neither descriptor. Actual: it accepts `TMaybe<TCustomMetricDescriptor>` (two overloads, `select_features.h:20,30`), already stubbed at `src/catboostr.cpp:1934`. Only the OBJECTIVE descriptor is genuinely absent (verified: zero matches for `TCustomObjectiveDescriptor` anywhere in `features_selection/`). catboost-8z4.94 now wires `select_features`'s custom METRIC (in scope) and excludes only custom OBJECTIVE (hard constraint, verified narrower).
2. **`catboost.eval_feature` was entirely missing from the plan.** `EvaluateFeatures(...)` (`eval_feature.h:87-91`) accepts BOTH descriptors, already stubbed at `src/catboostr.cpp:2077-2078`. Now in catboost-8z4.94's scope (both objective and metric).
3. **CatBoost's own logging (`Rprintf`, via `SetCustomLoggingFunction` in `R_API_BEGIN()`) would call an R API function from the new background thread on every verbose run** -- directly violating the plan's own no-R-API-off-main-thread rule. catboost-8z4.93 now routes logging through the same queue as callback requests.
4. **No handling for "background training thread throws/dies."** A drain loop with no producer would hang the R session forever. catboost-8z4.93 now requires bounded-time detection and clean propagation; catboost-8z4.91 now tests it explicitly (case f).
5. **`grid_search`/`randomized_search`'s `refit = TRUE` path (the default) silently dropped the custom objective/metric** on its internal re-entrant `catboost.train()` call. Fixed in catboost-8z4.94, tested in catboost-8z4.91 (case g).
6. **catboost-8z4.90 had silently dropped `TEvalMultiTargetFuncPtr`** from its implementation steps while still listing it in Reference Data. Restored; all six `TCustomMetricDescriptor` pointer members now explicitly required.
7. **New `custom_metric` argument name collided with the existing native `custom_metric` params-list key** (`R/catboost.R:2568,2787`, a string list of extra metrics to report -- unrelated meaning). Renamed to `custom_eval_metric_object`; the divergence from Python's naming is now a required doc item (catboost-8z4.92), following this repo's established convention for documenting R-vs-Python divergences.
8. **The multithreading-preservation assertion was an unresolved "wall-clock or counter" either/or**, with no CI-safe choice made and no instrumentation provisioned. Resolved: catboost-8z4.93 now exposes an atomic active-worker-count instrumentation hook; catboost-8z4.91's test asserts against that counter, not flaky wall-clock timing. catboost-8z4.88's own spike DoD was also reframed (see next point) to an honest end-to-end throughput comparison rather than a confounded intra-loop-overlap claim.
9. **catboost-8z4.88's spike success criterion was confounded.** It asked to prove "other TBB workers do non-callback work while one is blocked on the queue" -- but in the real derivative-computation call sites (`approx_calcer.cpp:131,208,298`), EVERY task in the relevant loop calls the R callback; there is no non-callback work in that specific loop to overlap. Reframed to the honest claim: end-to-end training throughput with `thread_count=M>1` is measurably faster than `thread_count=1` on the same custom-objective workload (the real parallelism gain is in other training phases -- histogram building, split search -- which are not gated on the R callback, plus TBB block-local setup overlapping across blocks; this is the same fundamental limitation Python's GIL-based bridge has for the callback portion itself).
10. **Stale ticket-ID cross-references** (`catboost-8z4.95`/`.96`, left over from an earlier numbering draft before the final IDs were assigned) corrected to the real IDs (`.94`, `.91`) throughout.
11. **`.93`'s own self-description mislabeled which ticket is the objective bridge vs. the metric bridge.** Corrected: catboost-8z4.89 is the objective bridge, catboost-8z4.90 is the metric bridge, catboost-8z4.94 is the cv/grid/randomized/select_features/eval_feature extension.
12. **No ticket required proving that omitting the new arguments reproduces today's exact behavior**, for the `cv` family specifically (train had it, `cv`/`grid_search`/`randomized_search`/`select_features`/`eval_feature` didn't). Added as an explicit backward-compatibility case to catboost-8z4.91 (case i), covering all six entry points.

The mechanism itself (background-thread + main-thread callback queue, no `thread_count=1`) is unchanged from v2 -- v3 corrects factual errors in scope/file claims and closes concurrency-safety gaps the mechanism itself created (logging, thread death, refit forwarding), it does not change the core approach.

## v3 -> v4: what changed and why

A user-requested informal 4th verification round (after the formal 3-iteration gate cap) found one more factual error and two test-coverage gaps, all fixed:

13. **`catboost.grid_search`/`catboost.randomized_search` do NOT route through `CrossValidate()`** — v3 (and every prior version) claimed they shared `CrossValidate`'s wiring point. They actually route through `GridSearch()`/`RandomizedSearch()` (`vendor/catboost/catboost/private/libs/hyperparameter_tuning/hyperparameter_tuning.h:46-52,60-68`), a separate pair of functions, declared in a header neither the plan nor catboost-8z4.94 previously cited. catboost-8z4.94 now cites the correct functions, headers, and exact `Nothing()` line numbers (`src/catboostr.cpp:1800-1801` for `GridSearch`, `:1870-1871` for `RandomizedSearch`) for both.
14. **Multi-class/multi-target descriptor variants were required by DoD but had zero test coverage.** catboost-8z4.89 requires `CalcDersMultiClass`/`CalcDersMultiTarget` wired; catboost-8z4.90 requires `TEvalMultiTargetFuncPtr` wired; every smoke test and every catboost-8z4.91 case only exercised single-target RMSE. catboost-8z4.91 now requires at least one differential case per non-single-target variant (case k).
15. **The logging-through-queue guard (v3 fix #3) had an implementation requirement but no test.** catboost-8z4.91 now requires a `verbose = TRUE` training run with a custom objective active, confirming log output appears correctly and the session does not crash (case l).
16. **`grid_search`/`randomized_search`'s actual descriptor wiring (not just the refit path) had no differential test** — catboost-8z4.91's case (g) tested only the refit-forwarding fix, not that the search itself uses the custom objective. Added as part of case (g)'s scope.

Epic: catboost-8z4. Base branch: phase-0 (135d84c, current tip). Spec:
docs/superpowers/specs/2026-07-30-catboostr-design.md §5 (Phase 6, "highest
risk; isolated deliberately"), §7 risk register ("Custom R loss/metric
callback infeasible" row), §9 descope trigger.

## v1 -> v2: what changed and why

v1 of this plan (plan-review-gate iteration 1, all 3 reviewers FAIL) proposed
forcing `thread_count = 1` for the whole training run whenever a custom R
objective/metric was set. User decision 2026-08-07, in response to that
gate's findings: **"multithreading is required, custom-objective support is
in scope."** `thread_count = 1` is REJECTED as a mechanism — it would kill
all of CatBoost's internal parallelism (histogram building, split search,
everything), not just the derivative computation. v2 replaces it with a
background-thread + main-thread callback-queue bridge (RcppThread's
documented pattern, hand-rolled, no new dependency) that keeps CatBoost's
own engine fully multithreaded and serializes only the R callback
invocation itself onto R's single-threaded main thread.

v1's reviewers also caught two independent, non-mechanism bugs, both fixed
here: (1) every file citation pointed at the gitignored, untracked
`vendor/catboost/catboost/R-package/src/catboostr.cpp` instead of the real,
tracked, fork-owned `/home/dd/Gemini/catboost/src/catboostr.cpp`; (2) the R
closure cannot travel through `catboost.train()`'s existing `params` ->
`jsonlite::toJSON()` path (no `asJSON` method for a function) — it must be a
new, dedicated `.Call` argument.

## Research findings (controller)

- **The C++ hook point needs no vendor/catboost edits.** `TrainModel()`
  (`vendor/catboost/catboost/libs/train_lib/train_model.h:152-165`) already
  accepts `const TMaybe<TCustomObjectiveDescriptor>&` and
  `const TMaybe<TCustomMetricDescriptor>&`. `CrossValidate()`
  (`vendor/catboost/catboost/libs/train_lib/cross_validation.h:165-166,
  174-175, 216-217`, used by `catboost.cv` ONLY) accepts the same two
  descriptors. `catboost.grid_search`/`catboost.randomized_search` do NOT
  share `CrossValidate`'s wiring point (an earlier draft claimed they did) —
  they route through `GridSearch()`/`RandomizedSearch()`
  (`vendor/catboost/catboost/private/libs/hyperparameter_tuning/hyperparameter_tuning.h:46-52,
  60-68`), a separate pair of functions in a separate header, each also
  already accepting both descriptors. `select_features`
  (`vendor/catboost/catboost/libs/features_selection/select_features.h:20,30`)
  accepts `TMaybe<TCustomMetricDescriptor>` at both overloads (in scope) but
  NO `TCustomObjectiveDescriptor` anywhere in that file or
  `recursive_features_elimination.h`/`.cpp` (confirmed via grep, zero
  matches for the objective type specifically — the earlier "neither"
  claim was wrong and is corrected here). `eval_feature`
  (`vendor/catboost/catboost/libs/train_lib/eval_feature.h:87-91`) accepts
  BOTH descriptors, fully in scope.
- **CPU training uses `NPar::TTbbLocalExecutor`, not `NPar::TLocalExecutor`.**
  `train_model.cpp:61-70`'s `CreateLocalExecutor` picks `TLocalExecutor` only
  on the GPU branch; the CPU branch (all of Phase 6's scope) uses
  `TTbbLocalExecutor<>(threadCount)`, an Intel TBB `task_arena` wrapper
  (`tbb_local_executor.cpp`: `ExecRange(WAIT_COMPLETE)` runs
  `TbbArena.execute([=]{ tbb::parallel_for(...) })`). v1's spike targeted the
  wrong class entirely; v2's spike (catboost-8z4.88) targets the real one.
- **R's C API is safe only from R's main thread** — confirmed via WebSearch
  against CRAN's Writing R Extensions manual and the RcppThread/RcppParallel
  authors' documentation: "It is not safe to call R's C API from multiple
  threads. It is safe, however, to call it from the main thread." Their
  documented solution: workers compute in pure C++, marshal any R-API call
  to the main thread via a queue. That is v2's mechanism, hand-rolled with
  `std::thread`/`std::mutex`/`std::condition_variable` (already-available
  stdlib, no new dependency — RcppThread itself is not adopted, since it
  requires Rcpp, contradicting spec §3's "raw `.Call` throughout, zero new
  mechanism" decision).
- **The real fork-owned glue file is `/home/dd/Gemini/catboost/src/catboostr.cpp`**
  (tracked; `CatBoostFit_R` at L1474, 4-arg `.Call` signature; `R_API_BEGIN`/
  `R_API_END` at L95/105; `TrainModel(...)` calls at L1501/L1516 currently
  passing `Nothing()` for both descriptors) — NOT
  `vendor/catboost/catboost/R-package/src/catboostr.cpp` (gitignored,
  untracked, forbidden to edit).
- **The closure must be a new `.Call` argument, not a `params` entry.**
  `R/catboost.R:2894-2923`'s `prepare_train_export_parameters` runs
  `jsonlite::toJSON(params, ...)` before the `.Call`; there is no `asJSON`
  method for an R function. `R/catboost.R:2840-2852`'s `validate_params_keys`
  also `stop()`s on unrecognized keys. Both are sidestepped by adding a
  genuinely new formal argument + `.Call` argument (extending
  `CatBoostFit_R`'s arity, `src/init.c:19`'s registration, and
  `src/catboostr.exports`), not by trying to route the closure through
  either existing path.

## Alternatives Considered

| Alternative | Status |
| :--- | :--- |
| **Background-thread + main-thread callback queue (chosen)** | Preserves CatBoost's full internal multithreading — only the R callback itself serializes onto R's main thread, which is unavoidable no matter the mechanism (R has no equivalent of Python's GIL that other threads can acquire). Zero new dependency; matches RcppThread's own documented pattern without adopting RcppThread/Rcpp. |
| **Force `thread_count = 1` for the whole run (v1's mechanism, REJECTED)** | User decision 2026-08-07: multithreading is required. Killed ALL of CatBoost's engine parallelism (histogram building, split search, etc.), not just the loss/metric computation — a strictly worse tradeoff than necessary once the queue pattern is available. |
| **RcppThread/RcppParallel as a dependency** | REJECTED. Same pattern as the chosen mechanism, but pulls in Rcpp as a transitive dependency, directly contradicting spec §3's "raw `.Call` throughout, zero new mechanism" decision (already-rejected cpp11 for the same reason). The pattern is adopted; the library is not. |
| **`reticulate`-style second interpreter / async job queue** | REJECTED, same class of rejection as spec §6's already-refuted `reticulate` alternative for the whole package. |

**Mechanism Justification** (ranked): 1. Performance — CatBoost's own engine keeps its configured thread count; only the R-callback portion serializes, which is the theoretical minimum cost of calling into a single-threaded interpreter, not an avoidable inefficiency. 2. Simplicity — one shared queue/condvar component (catboost-8z4.93) reused by four call sites (objective, metric, train, cv-family), versus four independent thread-forcing hacks. 3. Ecosystem — mirrors RcppThread's own documented, widely-used pattern for this exact class of problem. 4. Maintenance — all glue lives in fork-owned `src/` files; the C++ engine itself is untouched.

## Global Constraints

- **Mechanism:** `TrainModel()`/`CrossValidate()`/`GridSearch()`/`RandomizedSearch()`/`SelectFeatures()`/`EvaluateFeatures()`'s existing `TMaybe<TCustomObjectiveDescriptor>`/`TMaybe<TCustomMetricDescriptor>` parameters, wired from fork-owned `src/catboostr.cpp`; training runs on a background thread; R closures invoked via `R_tryEval` from a main-thread queue-drain loop (catboost-8z4.93's shared bridge).
- **Forbidden:** no edits to `vendor/catboost/` (read-only); no new R-level threading dependency (RcppThread/RcppParallel/Rcpp); no `reticulate`/second-interpreter approach; no calling any R API function from a thread that is not R's confirmed main thread; no attempting to wire `catboost.select_features`'s custom OBJECTIVE (its C++ engine has no descriptor hook for objectives specifically — a hard constraint, not a deferred task; its custom METRIC hook exists and IS in scope, wired by catboost-8z4.94).
- **Audit:** catboost-8z4.88's spike output is the audit artifact for the entire mechanism — it must show, by measurement, that the queue-based pattern round-trips correctly and produces a measurable end-to-end throughput improvement at `thread_count > 1` vs `thread_count = 1` on the same custom-objective workload (NOT literal non-callback-work overlap during a blocked callback wait within the same `parallel_for` loop — every task in the real derivative-computation call sites invokes the callback, so that specific framing is unmeasurable and is not the criterion), and handles R-level errors cleanly. catboost-8z4.93 onward is gated on catboost-8z4.88 passing; if it is REFUTED, stop and file a single infeasibility-recording ticket per spec §9's descope trigger instead of proceeding.
- **Verification method:** differential tests (catboost-8z4.91) comparing R custom-objective/custom-metric training runs against equivalent built-in-loss runs, across all six wired entry points (`catboost.train`, `cv`, `grid_search`, `randomized_search`, `select_features` [metric only], `eval_feature`), at default tolerance, per the project's existing oracle/differential-test convention. Multithreading preservation is verified via catboost-8z4.93's atomic active-worker-count instrumentation, not wall-clock timing or inferred from correctness alone. Interrupt-handling and background-thread-death guards (catboost-8z4.93) each require their own test in catboost-8z4.91, not just an implementation-side claim.

## Execution Order

1. **catboost-8z4.88** — Spike: verify the background-thread + main-thread callback-queue pattern against the real CPU executor (`TTbbLocalExecutor`). Gates everything below.
2. **catboost-8z4.93** — Build the shared callback-queue bridge infrastructure (fork-owned `src/` component), reused by all four downstream consumers.
3. **catboost-8z4.89** — Wire custom R objective into `catboost.train`.
4. **catboost-8z4.90** — Wire custom R metric into `catboost.train`.
5. **catboost-8z4.94** — Extend to `catboost.cv` (`CrossValidate`), `catboost.grid_search` (`GridSearch`), `catboost.randomized_search` (`RandomizedSearch` -- three distinct C++ functions, not one shared wiring point), `catboost.select_features` (metric only -- objective has no C++ hook), `catboost.eval_feature` (both descriptors). Fixes `grid_search`/`randomized_search`'s `refit = TRUE` forwarding gap.
6. **catboost-8z4.91** — Differential/parity tests across all wired entry points, including a multithreading-preservation assertion and error-propagation tests.
7. **catboost-8z4.92** — Documentation, including the `select_features` exclusion and the multithreading-preserving mechanism.

If catboost-8z4.88 disproves the queue-based hypothesis, tasks .93 onward are cancelled/closed as not-applicable and replaced by one infeasibility-recording ticket instead (per spec §9 descope trigger) — controller decides at that checkpoint.

## Tickets (verbatim, hermetic — see `bd show catboost-8z4.<n>` for full body)

- catboost-8z4.88
- catboost-8z4.93
- catboost-8z4.89
- catboost-8z4.90
- catboost-8z4.94
- catboost-8z4.91
- catboost-8z4.92
