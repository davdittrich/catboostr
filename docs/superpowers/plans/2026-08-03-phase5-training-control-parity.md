# Phase 5: Training-Control Parity — Implementation Plan

Epic: catboost-8z4. Spec: `docs/superpowers/specs/2026-07-30-catboostr-design.md`,
gate at line 639: "Training-control parity: `init_model`, grid/randomized
search, `select_features`, virtual ensembles verified end-to-end, the
generated parameter documentation and validation (§4.2). Includes the CLI's
`metadata` and `normalize-model` modes." Gate condition: "Differential tests
green."

Approved by plan-review-gate 2026-08-03 (3 iterations; Feasibility PASS all
3, Scope & Alignment PASS all 3, Completeness FAIL/FAIL/FAIL — three rounds
of real gaps found and fixed: P5.7's wrong "no Python equivalent" claim for
normalize-model plus incomplete matrix coverage; P5.5's missing backward-
compat handling for `process_synonyms_in_one_group` alias keys; P5.5's
missing family-level closure of all 139 `param:*` matrix rows per §4.5;
P5.1's missing explicit backward-compat guard for `init_model`; and a
missing beads dependency edge between P5.1 and P5.5, both of which edit
`catboost.train`/`catboost.cv`). Each task below is the authoritative source
for its own beads ticket (`bd show <id>`); the ticket and this plan file
must stay in sync — the ticket is canonical.

## Global Constraints

- **Read-only vendor pin.** Zero writes under `vendor/catboost/` in any task.
  Verify with `git -C vendor/catboost status --porcelain` before and after —
  must be empty both times.
- **No core patches.** Upstream C++ source is never patched locally. Anything
  needing a core change is out of scope / filed upstream, never worked around
  by editing `vendor/catboost/`.
- **Append-only R API.** Existing exported function signatures never change
  in an incompatible way — new arguments get defaults that preserve existing
  call sites. `catboost.train`'s new `init_model` argument (P5.1) and the
  new `params`-key validation gate (P5.5) both edit the same function bodies
  — P5.1 (catboost-8z4.58) MUST land and close before P5.5
  (catboost-8z4.62) starts; this is enforced by a beads `blocks` dependency
  (`bd show catboost-8z4.62` lists `catboost-8z4.58` under DEPENDS ON), not
  just this note.
- **§4.2 mapping rule.** Enum members become accepted *values* of an existing
  argument, never new exported functions. CLI modes and Python methods become
  a single `catboost.*` function (snake_case `verb_object`, never a dotted
  S3-colliding name like `model.compare`).
- **§4.3 verification.** Every capability is proven by a differential test
  against the pinned oracle (Python: relative tolerance `1e-12` default; CLI:
  relative tolerance `1e-9` default, never bit-exact, any override needs a
  first-principles justification recorded in the matrix).
- **§4.5 parameter/flag bulk-disposition rule.** The 139 `kind:"parameter"`
  matrix rows are dispositioned as a family, not individually — one
  differential test per parameter family, plus the generated documentation
  and validation. They never become per-name tickets or per-name tests.
  P5.5 (catboost-8z4.62) owns closing all 139 rows, reusing existing test
  coverage where it already exercises a family and adding the minimum new
  family-level tests where it doesn't.
- **Parity matrix.** `tests/fixtures/parity/matrix.dispositioned.json` (725
  rows) is updated to `state: green` with a `test_id` for every row a task
  closes. Do not flip a row green without a passing differential test backing
  it. Do not trust a row count stated in a ticket over your own search of
  the matrix file — several counts in these tickets were corrected during
  plan review and any implementer should re-verify, not copy stale numbers.
- **Test convention.** New tests go in `tests/testthat/test_<capability>.R`,
  following the differential-test-against-oracle pattern already used by
  every closed Phase 1-4 ticket.
- **Build/verify command.** `R CMD INSTALL --preclean .` (redirect output to
  a log file and check `$?` directly — never pipe through `tail`, which
  masks the real exit code) then run the new test file; quote exact output
  in the task report.
- **Isolation.** This plan executes entirely inside the
  `.worktrees/phase-5-training-control-parity` worktree (branch
  `phase-5-training-control-parity`, based on `phase-0`). Each task commits
  its own work on this branch.
- **Execution order.** Tasks run in the order below (P5.1 through P5.7), not
  in parallel — P5.1 and P5.5 share `catboost.train`/`catboost.cv`, and
  serial execution is the simplest way to avoid a merge conflict on the same
  function bodies.
- **No scope creep (gate-enforced).** Each task's own "Scope"/"Boundary"
  guard lists what it must NOT do. Do not re-expand scope during
  implementation without stopping to ask.

## Task 1: P5.1: init_model support (continue training) (catboost-8z4.58)

_Beads ticket: `bd show catboost-8z4.58`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._

# P5.1: init_model support (continue training)

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add `init_model` support to `catboost.train`/`catboost.cv` (continue training an existing model), verified by differential test against the Python oracle.
* **Why:** Phase 5 gate (spec line 639, "Training-control parity: `init_model`, ..."). Matrix row `param:init_model` — `CatBoost.fit(param=init_model)`, red, oracle python.
* **Reference Data:** Python `CatBoost.fit(..., init_model=...)` accepts a `CatBoost`/`CatBoostClassifier`/path/string. R's `catboost.train(learn_pool, test_pool, params)` signature (`R/catboost.R:2565`) has no such argument today — confirm exact current signature by reading before editing. Vendor core native training entry point already supports continuing from an existing model (used internally by Python) — locate and reuse it; do not reimplement training-continuation logic in R.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Python oracle `CatBoost.fit` signature/docstring for `init_model`, `R/catboost.R` current `catboost.train`/`catboost.cv` bodies, `src/catboostr.cpp` native training entry points.
* **Format:** Python introspection output, R source, C++ source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under vendor/catboost/. |
| **Scope** | Only `init_model`. Do not implement grid/randomized search, select_features, or param validation (separate tickets). |
| **Tolerance** | Model continuation must match Python's oracle within existing prediction-comparison tolerance (default per §4.3) — not a new tolerance class. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Backward compat** | `init_model` must be optional, default `NULL`/absent, so every existing `catboost.train`/`catboost.cv` call site with no `init_model` argument is byte-for-byte unaffected — write a regression test proving this, per epic's "strict superset, nothing that works today breaks" guard (§4.6). |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Read Python oracle's `CatBoost.fit` signature; quote exact `init_model` parameter type/behavior.
2. Read current `catboost.train`/`catboost.cv` R signatures (`R/catboost.R`) and the native `.Call` entry points they use; quote exact current code.
3. Locate the vendor core's model-continuation training entry point (grep training call sites for an existing-model/init-model argument); confirm reuse path, do not reimplement.
4. Add `init_model` argument to `catboost.train` as an optional, default-absent argument (accepting a `catboost.Model` object or file path, matching Python's accepted types), threading it through to the native call, positioned so existing positional call sites are unaffected.
5. Write a differential test: train a model, continue training it via `init_model`, compare predictions against the Python oracle's continued-model predictions at the declared tolerance.
6. Write a backward-compatibility regression test: call `catboost.train`/`catboost.cv` without `init_model` and assert output is unchanged from pre-change behavior (e.g. against a fixed fixture/golden prediction, or by re-running an existing pre-Phase-5 test unmodified).
7. Update matrix disposition (`param:init_model`) to green with test id.
8. Run `R CMD INSTALL --preclean .` (redirect output to a log file, check `$?` directly — never pipe through `tail`) then the new test file. Quote exact output.
9. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:
```toon
task_id: catboost-8z4.58
success: bool
data:
  function_signature_changed: bool
  differential_test_passing: bool
report_path: docs/phase-5/P5.1-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] Python oracle `init_model` behavior quoted in report.
- [ ] `catboost.train` accepts `init_model` as an optional, default-absent argument; reuses vendor's native continuation path (no reimplementation).
- [ ] Regression test proves existing `catboost.train`/`catboost.cv` calls without `init_model` are unaffected.
- [ ] Differential test against Python oracle passes at declared tolerance; matrix row flipped to green.
- [ ] git -C vendor/catboost status --porcelain empty.
- [ ] Report written with exact command output.


## Task 2: P5.2: grid_search / randomized_search (catboost-8z4.59)

_Beads ticket: `bd show catboost-8z4.59`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._

# P5.2: grid_search / randomized_search

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add R equivalents of Python's `CatBoost.grid_search` and `CatBoost.randomized_search` (hyperparameter search with optional CV), verified by differential test against the Python oracle.
* **Why:** Phase 5 gate (spec line 639, "grid/randomized search"). Matrix rows `CatBoost.grid_search` / `CatBoostClassifier.grid_search` / `CatBoostRegressor.grid_search` / `CatBoostRanker.grid_search` and the `randomized_search` equivalents — all red, oracle python.
* **Reference Data:** Python's `grid_search`/`randomized_search` internally repeat `fit`/`cv` over a parameter grid/sample and pick the best by a metric — confirm exact algorithm and return-value shape (best params, best score, per-fold results) by reading the Python oracle source, not by assumption. R already has `catboost.train`/`catboost.cv`; this ticket wraps them in a search loop, it does not add new native training paths.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Python oracle `grid_search`/`randomized_search` source and docstrings, existing `catboost.train`/`catboost.cv` R functions.
* **Format:** Python source, R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under vendor/catboost/. |
| **Scope** | `grid_search`/`randomized_search` only. Reuse `catboost.train`/`catboost.cv`; no new native entry points unless the Python implementation itself requires one you can point to in its source. |
| **Tolerance** | Selected best-params and best-score must match Python oracle within default tolerance (§4.3) on a fixed random seed/search space fixture. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Read Python oracle's `grid_search`/`randomized_search` implementation; quote the exact search/selection algorithm and return shape.
2. Design R functions `catboost.grid_search`/`catboost.randomized_search` (verb_object snake_case per §4.2) wrapping `catboost.train`/`catboost.cv` in a loop over the grid/sampled params.
3. Implement, matching Python's parameter names (`param_grid`, `n_iter` for randomized, `cv`, `search_by_train_test_split`, etc. — confirm exact names from step 1, do not guess).
4. Write a differential test: fixed param grid/search space and seed, compare best-params/best-score against the Python oracle at declared tolerance.
5. Update matrix dispositions (all 8 rows: grid_search x4 model classes, randomized_search x4) to green with test id(s).
6. Run `R CMD INSTALL --preclean .` (redirect to log file, check `$?` directly, never pipe through `tail`) then the new test file. Quote exact output.
7. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:
```toon
task_id: catboost-8z4.59
success: bool
data:
  grid_search_implemented: bool
  randomized_search_implemented: bool
  differential_test_passing: bool
report_path: docs/phase-5/P5.2-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] Python oracle algorithm/return-shape quoted in report.
- [ ] `catboost.grid_search`/`catboost.randomized_search` implemented, reusing existing train/cv.
- [ ] Differential test against Python oracle passes at declared tolerance; all 8 matrix rows flipped to green.
- [ ] git -C vendor/catboost status --porcelain empty.
- [ ] Report written with exact command output.


## Task 3: P5.3: select_features (catboost-8z4.60)

_Beads ticket: `bd show catboost-8z4.60`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._

# P5.3: select_features

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add an R equivalent of Python's `CatBoost.select_features` (algorithm-driven feature elimination during training) and the CLI's `select-features` mode as its oracle, verified by differential test.
* **Why:** Phase 5 gate (spec line 639, "select_features"). Matrix rows `CatBoost.select_features` / `CatBoostClassifier.select_features` / `CatBoostRegressor.select_features` / `CatBoostRanker.select_features` — red, oracle python. Note P4.7's `eval_feature` ticket explicitly excluded this as "a different algorithm" (`R/catboost.R:2712` docstring already references `CatBoost.select_features` as distinct from eval_feature) — do not conflate the two or reuse P4.7's `catboost.eval_feature` as this function's implementation.
* **Reference Data:** `EFeaturesSelectionAlgorithm` (vendor) drives this; per P4.7's report (`docs/phase-4/P4.7-report.md`) this was explicitly out of scope there. Read the Python oracle's `select_features` source for exact signature/algorithm choices before designing.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Python oracle `select_features` source/docstring, CLI `select-features` --help (pinned CLI oracle, secondary cross-check), vendor `EFeaturesSelectionAlgorithm` and its training-loop entry point.
* **Format:** Python source, CLI help text, C++ vendor source, R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under vendor/catboost/. |
| **Scope** | `select_features` only. Do not modify or reuse `catboost.eval_feature` (P4.7) as this function's body — different algorithm. |
| **Tolerance** | Python oracle is primary (default tolerance, §4.3); if no in-process Python-equivalent native path exists, fall back to CLI oracle at relative 1e-9 — document which was used and why. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Read Python oracle's `select_features` signature/algorithm; quote exact behavior (features_for_select, num_features_to_select, algorithm, steps, train_final_model, etc.).
2. Locate the vendor native entry point for feature selection (grep for `EFeaturesSelectionAlgorithm` / select-features training path); confirm it is linked and reachable in-process (same verification style as P4.7/P4.8 opus-tier design decisions), or document why not.
3. Design and implement `catboost.select_features` (verb_object snake_case), matching Python's parameter names.
4. Write a differential test comparing selected-features output against the Python oracle (or CLI oracle if no in-process Python path exists) at the declared tolerance.
5. Update matrix dispositions (all 4 rows) to green with test id.
6. Run `R CMD INSTALL --preclean .` (redirect to log file, check `$?` directly, never pipe through `tail`) then the new test file. Quote exact output.
7. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:
```toon
task_id: catboost-8z4.60
success: bool
data:
  in_process_python_path_found: bool
  function_implemented: bool
  differential_test_passing: bool
report_path: docs/phase-5/P5.3-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] Python oracle `select_features` signature/algorithm quoted in report.
- [ ] `catboost.select_features` implemented, distinct from `catboost.eval_feature`.
- [ ] Differential test passes at declared tolerance; all 4 matrix rows flipped to green.
- [ ] git -C vendor/catboost status --porcelain empty.
- [ ] Report written with exact command output.


## Task 4: P5.4: virtual ensembles verified end-to-end (catboost-8z4.61)

_Beads ticket: `bd show catboost-8z4.61`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._

# P5.4: virtual ensembles verified end-to-end

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Verify `catboost.virtual_ensembles_predict` (already implemented, `R/catboost.R:3269`) end-to-end with a real differential test against the Python oracle; fix any gap found, do not assume it already works untested.
* **Why:** Phase 5 gate (spec line 639, "virtual ensembles verified end-to-end"). No matrix row currently tracks this capability by name (confirmed: `virtual_ensemble`/`VirtEnsembles` absent from `tests/fixtures/parity/matrix.dispositioned.json` except `flag:--virtual-ensembles-count`, red, CLI `calc` mode) — this ticket must first determine why the capability inventory did not surface it as its own row (naming mismatch vs. genuinely missing from Phase 2's inventory) before writing the test.
* **Reference Data:** `R/catboost.R:3229-3284` — existing `catboost.virtual_ensembles_predict` implementation, `prediction_type = "VirtEnsembles"`. Python oracle's `virtual_ensembles_predict` top-level function is the primary comparison target.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Current `catboost.virtual_ensembles_predict` R source, Python oracle `virtual_ensembles_predict` source/docstring, `tests/fixtures/parity/matrix.dispositioned.json` inventory row search results.
* **Format:** R source, Python source, JSON, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under vendor/catboost/. |
| **Scope** | Verification and any bug-fix needed to reach parity for `virtual_ensembles_predict` only. Do not touch `flag:--virtual-ensembles-count` (CLI `calc` mode, separate capability) unless it is the same code path — confirm before touching. |
| **Tolerance** | Default tolerance per §4.3 (Python oracle, in-process). |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Search `tests/fixtures/parity/matrix.dispositioned.json` for every row plausibly related to virtual ensembles (by capability string and by `inventory_row_id`); quote exact search results, including "none found" if that's the case.
2. If no row exists, determine whether this is a genuine inventory gap (report it, do not silently add a row not backed by the machine-generated inventory process) or a naming mismatch (e.g., folded under a `prediction_type` enum row) — quote evidence either way.
3. Read the current `catboost.virtual_ensembles_predict` implementation and the Python oracle's equivalent; quote both signatures.
4. Write a differential test: train a model, predict with virtual ensembles via both R and Python oracle, compare shapes and values at declared tolerance.
5. If the test fails, root-cause and fix (do not paper over); if it passes, the ticket confirms existing correctness.
6. If step 2 found a genuine inventory gap, file a follow-up ticket for the Phase 2 inventory process rather than hand-adding a matrix row.
7. Run `R CMD INSTALL --preclean .` (redirect to log file, check `$?` directly, never pipe through `tail`) then the new test file. Quote exact output.
8. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:
```toon
task_id: catboost-8z4.61
success: bool
data:
  matrix_row_found: bool
  bug_found_and_fixed: bool
  differential_test_passing: bool
report_path: docs/phase-5/P5.4-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] Matrix search results (found/not-found, with evidence) quoted in report.
- [ ] Differential test against Python oracle written and passing at declared tolerance.
- [ ] Any bug found during verification is root-caused and fixed, not papered over.
- [ ] git -C vendor/catboost status --porcelain empty.
- [ ] Report written with exact command output.


## Task 5: P5.5: generated parameter documentation and validation (catboost-8z4.62)

_Beads ticket: `bd show catboost-8z4.62`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._

# P5.5: generated parameter documentation, validation, and family-level differential closure

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** (a) Generate `@param`-level roxygen documentation for all 139 distinct hyperparameters from the machine-generated capability inventory (not hand-written); (b) add validation in `catboost.train`/`catboost.cv` that rejects unknown `params` keys with a documented escape hatch, without breaking existing synonym/alias keys; (c) close all 139 `kind:parameter` matrix rows to green via family-level differential tests, per §4.5's disposition rule — reusing existing test coverage where it already exercises a parameter family, adding one minimal family-level test where it doesn't. Per §4.5 this is explicitly NOT 139 per-name tickets — one ticket, family-level closure.
* **Why:** Phase 5 gate (spec line 639, "the generated parameter documentation and validation (§4.2)") plus §4.5's Phase 2 bulk-disposition rule: "Parameter/flag rows are dispositioned as a family, not individually: one differential test per parameter family that reaches the core through `params`, plus the generated documentation and validation described in §4.2. They do not become per-name tickets." (found during plan review: an earlier draft of this ticket implemented only the docs+validation half and left all 139 parameter rows red, which fails Phase 5's own gate condition, "Differential tests green.")
* **Reference Data:** §4.5 machine-generated capability inventory is the source of the 139 hyperparameter (canonical) names — never hand-curate this list. `catboost.train` currently calls `process_synonyms_in_one_group(...)` (grep `R/catboost.R` — confirmed ~11-12 call sites around line 2600-2610, re-verify exact lines/count, may have shifted) accepting synonym/alias keys not equal to canonical names, e.g. `eta` for `learning_rate`, `n_estimators`/`num_boost_round`/`num_trees` for `iterations`, `max_depth` for `depth`, `reg_lambda` for `l2_leaf_reg`, `random_state` for `random_seed`, `min_child_samples` for `min_data_in_leaf`, `num_leaves` for `max_leaves`, `colsample_bylevel` for `rsm`, `max_bin` for `border_count`, `verbose_eval` for `verbose`, `objective` for `loss_function`. The generated 139-name list is canonical names only — the validation gate MUST also accept every existing synonym/alias key. This is a hard constraint. Existing test suite (all phases, `tests/testthat/`) already trains models exercising many hyperparameters and compares against the Python oracle — much of the 139-family closure is plausibly already satisfied by tests that were never linked to their matrix row via `test_id`; audit before writing new tests, do not duplicate coverage.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Machine-generated capability inventory (`kind:"parameter"` rows in `tests/fixtures/parity/matrix.dispositioned.json`, 139 rows per prior phase measurement — reconcile, don't assume), current `catboost.train`/`catboost.cv` R source, every `process_synonyms_in_one_group(...)` call site in `R/catboost.R`, existing `tests/testthat/*.R` files for parameter-exercising coverage.
* **Format:** JSON inventory, R source, generated `.Rd`/roxygen `@param` blocks, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under vendor/catboost/. |
| **Scope** | Documentation generation + validation gate + family-level matrix closure only. Do not hand-author the 139-item list; derive it programmatically from the inventory. Do not restructure `params = list()` into a typed/structured argument API. Do not spawn per-name tickets or per-name tests — §4.5 explicitly forbids this; group families sensibly (e.g. one test that trains with a representative combination covering multiple families at once is acceptable and preferred over 139 separate tests). |
| **Backward compatibility** | Every key accepted by an existing `process_synonyms_in_one_group(...)` call must still validate successfully after this change. Enumerate every call site and union their alias keys into the accepted-key set, or resolve synonyms to canonical names before the check runs — explicit and tested either way. |
| **Escape hatch** | Validation must not break forward-compatibility with a newer vendor core adding params — document the exact opt-out mechanism chosen. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Locate the 139-hyperparameter list in the machine-generated inventory (`kind:"parameter"` rows); quote the count found and confirm it matches 139 (if not, document the discrepancy, do not force the number).
2. Read current `catboost.train`/`catboost.cv` `params` handling; quote exact current code and line numbers (verify against the `:1550`/`:1622` line numbers cited in the spec, which may be stale).
3. Grep every `process_synonyms_in_one_group(...)` call site in `R/catboost.R`; quote the full list of synonym groups and every alias key they accept.
4. Write a generator (script under `tools/` or similar, matching existing inventory-generation conventions) that emits `@param`-level roxygen documentation for each canonical hyperparameter from the inventory data, not by hand.
5. Add a validation step in `catboost.train`/`catboost.cv`: check every key of the supplied `params` list against the union of (a) the generated canonical name list and (b) every alias key found in step 3; error on unknown keys unless the documented escape hatch is used.
6. Write validation tests: (a) valid canonical params pass through unchanged, (b) every existing synonym/alias key from step 3 still passes through unchanged, (c) a mistyped/unknown key errors with a clear message, (d) the escape hatch permits an unknown key when explicitly requested.
7. Audit the existing test suite (`tests/testthat/*.R`, across all phases) for which of the 139 parameter families are already exercised by a training call whose output is compared against the Python oracle; quote the mapping found (family → existing test file/test name), including gaps.
8. For every parameter family with no existing family-level differential-test coverage found in step 7, add the minimum number of new differential tests needed to cover it — batching multiple families into shared training runs where the parameters don't interact adversarially, per §4.5's "not per-name" instruction. Compare against the Python oracle at default tolerance (§4.3).
9. Update all 139 `param:*` matrix rows to green with the covering test id (existing or new) — record the family→test_id mapping in the report.
10. Run `roxygen2::roxygenise()` to regenerate `.Rd` files; confirm the new `@param` blocks render.
11. Run `R CMD INSTALL --preclean .` (redirect to log file, check `$?` directly, never pipe through `tail`) then the new/full test suite. Quote exact output.
12. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:
```toon
task_id: catboost-8z4.62
success: bool
data:
  hyperparameter_count_found: int
  synonym_alias_groups_found: int
  generator_script_path: string
  validation_added: bool
  escape_hatch_mechanism: string
  synonym_regression_tests_passing: bool
  parameter_families_closed: int
  new_tests_added: int
report_path: docs/phase-5/P5.5-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] Hyperparameter count from inventory quoted and reconciled against the spec's "139" figure.
- [ ] Generated (not hand-written) `@param` documentation covers every canonical hyperparameter.
- [ ] `catboost.train`/`catboost.cv` reject unknown `params` keys, with a documented, tested escape hatch.
- [ ] Every existing `process_synonyms_in_one_group` alias key still validates successfully, proven by a regression test per group.
- [ ] All 139 `param:*` matrix rows flipped to green, each with a recorded covering test_id (existing or new), per family — not per-name tickets/tests.
- [ ] Full test suite green including new validation, synonym-regression, and family-coverage tests.
- [ ] git -C vendor/catboost status --porcelain empty.
- [ ] Report written with exact command output.


## Task 6: P5.6: CLI metadata mode parity (catboost-8z4.63)

_Beads ticket: `bd show catboost-8z4.63`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._

# P5.6: CLI metadata mode parity

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add an R-reachable equivalent of the CLI's `metadata` mode (subcommands: `set`, `get`, `dump`, `dump-feature-names`) and of `CatBoost.get_metadata`, verified by differential test against the CLI oracle (and Python oracle for `get_metadata`).
* **Why:** Phase 5 gate (spec line 639, "Includes the CLI's `metadata` ... mode"). Matrix rows `mode:metadata`, `mode:metadata set`, `mode:metadata get`, `mode:metadata dump`, `mode:metadata dump-feature-names`, plus `CatBoost.get_metadata`/`CatBoostClassifier.get_metadata`/`CatBoostRegressor.get_metadata`/`CatBoostRanker.get_metadata` — all red.
* **Reference Data:** Model metadata (arbitrary string key/value pairs embedded in the saved model, e.g. training params, custom user data) is likely already reachable via existing model-save/-load native paths — check whether `catboost.Model` already exposes any metadata accessor before assuming none exists. CLI `--help` for `metadata set/get/dump/dump-feature-names` is the authoritative flag source for the CLI-mode functions.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** CLI `metadata` --help output (pinned CLI oracle, all 4 subcommands), Python oracle `get_metadata`/`set_metadata` source, current R `catboost.Model` accessors.
* **Format:** CLI help text, Python source, R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under vendor/catboost/. |
| **Scope** | `metadata` mode + `get_metadata` only. Do not implement `normalize-model` (separate ticket). |
| **Tolerance** | `get_metadata` (Python oracle): default tolerance. CLI `metadata` subcommands (CLI-only): relative 1e-9 / exact string match as appropriate, never bit-exact numeric comparison for text output. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Get CLI `metadata set`/`get`/`dump`/`dump-feature-names` --help output via pinned CLI oracle; quote exact output for each subcommand.
2. Read Python oracle's `get_metadata`/`set_metadata` (or equivalent) source; quote exact signature/behavior.
3. Check for an existing R-side metadata accessor on `catboost.Model` objects; quote result.
4. Implement `catboost.get_metadata` (and `catboost.set_metadata` if the Python oracle exposes a paired setter) matching Python's calling convention, reusing the native model-save/-load path's metadata storage.
5. Implement R-reachable equivalents of the CLI `metadata` subcommands only if no in-process Python-side equivalent covers them (e.g. `dump`/`dump-feature-names` may be CLI-only) — document which subcommands map to which oracle.
6. Write differential tests per subcommand/function at the declared tolerance.
7. Update all listed matrix rows to green with test id(s).
8. Run `R CMD INSTALL --preclean .` (redirect to log file, check `$?` directly, never pipe through `tail`) then the new test file. Quote exact output.
9. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:
```toon
task_id: catboost-8z4.63
success: bool
data:
  get_metadata_implemented: bool
  cli_subcommands_implemented: list
  differential_test_passing: bool
report_path: docs/phase-5/P5.6-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] CLI --help output for all 4 metadata subcommands quoted in report.
- [ ] `catboost.get_metadata` implemented and matches Python oracle.
- [ ] CLI-only subcommands implemented or explicitly documented as CLI-only with reasoning.
- [ ] Differential tests pass at declared tolerance; all listed matrix rows flipped to green.
- [ ] git -C vendor/catboost status --porcelain empty.
- [ ] Report written with exact command output.


## Task 7: P5.7: CLI normalize-model mode parity (catboost-8z4.64)

_Beads ticket: `bd show catboost-8z4.64`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._

# P5.7: CLI normalize-model mode parity

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add R equivalents of `CatBoost.get_scale_and_bias`/`CatBoost.set_scale_and_bias` (Python) and the CLI's `normalize-model` mode (`--set-scale`, `--set-bias`, `--print-scale-and-bias`), verified by differential test against both oracles.
* **Why:** Phase 5 gate (spec line 639, "Includes the CLI's ... `normalize-model` mode"). Matrix rows confirmed present in `tests/fixtures/parity/matrix.dispositioned.json`: `mode:normalize-model`, `flag:--set-scale`, `flag:--set-bias`, `flag:--print-scale-and-bias`, `flag:--input-path/-i`, `flag:--output-model`, `flag:--output-model-format`, plus `CatBoost.get_scale_and_bias`/`CatBoost.set_scale_and_bias` and their `CatBoostClassifier`/`CatBoostRegressor`/`CatBoostRanker` equivalents — all red. **Do not hard-code a row count in the implementation** (an earlier draft of this ticket said "12," which plan review found did not match a fresh enumeration of the matrix — treat any specific number in this ticket, including counts implied above, as a starting hint, not ground truth: search the matrix yourself and close every row your search finds).
* **Reference Data:** Correction to an earlier draft of this ticket: a Python-side equivalent DOES exist — `get_scale_and_bias`/`set_scale_and_bias` on `CatBoost` and its subclasses read/write the same scale-and-bias values the CLI's `normalize-model` mode manipulates. Treat Python as the primary oracle for `get_scale_and_bias`/`set_scale_and_bias`; treat the CLI as the oracle only for whatever `normalize-model` behavior (if any) is not reachable through those two Python methods (e.g. bulk rewrite of a saved model file via `--input-path`/`--output-model` without loading into a live object). Confirm this split empirically — read both the Python oracle's method source and the CLI's `normalize-model` --help — before designing.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Python oracle `get_scale_and_bias`/`set_scale_and_bias` source/docstrings (all 4 model classes), CLI `normalize-model` --help output (pinned CLI oracle), existing R `catboost.Model` save/load functions as calling-convention precedent.
* **Format:** Python source, CLI help text, R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under vendor/catboost/. |
| **Scope** | `get_scale_and_bias`/`set_scale_and_bias`/`normalize-model` only. |
| **Tolerance** | `get_scale_and_bias`/`set_scale_and_bias` (Python oracle, in-process): default tolerance per §4.3. Any CLI-only residual behavior not reachable via those two methods: relative 1e-9, never bit-exact. |
| **Matrix completeness** | Every row your own search of the matrix finds related to `normalize-model`/`get_scale_and_bias`/`set_scale_and_bias` (by grepping `inventory_row_id` and `members` for those strings, plus `mode:normalize-model` and its `flag:*` children) must be dispositioned. Do not rely on a fixed number from this ticket text — search and enumerate yourself; report the exact count found. Rows left red must be explicitly justified in the report, not silently dropped. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Read Python oracle's `get_scale_and_bias`/`set_scale_and_bias` source for all 4 model classes; quote exact signatures/behavior.
2. Get CLI `normalize-model` --help output via pinned CLI oracle; quote exact output including every flag.
3. Search `tests/fixtures/parity/matrix.dispositioned.json` yourself for every row related to `normalize-model`, `get_scale_and_bias`, `set_scale_and_bias`; quote the exact list found and its count (do not trust any number stated elsewhere in this ticket).
4. Determine which capability maps to which oracle: does `--set-scale`/`--set-bias`/`--print-scale-and-bias` map 1:1 onto `set_scale_and_bias`/`get_scale_and_bias`, or does the CLI mode do something (e.g. file-to-file rewrite via `--input-path`/`--output-model`) not reachable through the Python methods? Quote evidence for the split.
5. Implement `catboost.get_scale_and_bias`/`catboost.set_scale_and_bias` (verb_object snake_case), matching Python's calling convention, reusing the native model scale/bias storage already used by save/load.
6. If step 4 found CLI-only residual behavior, implement an R equivalent for that residual only (or document as a blocking design gap per the epic's "any deviation halts and reports" guard — do not silently skip).
7. Write differential tests: `get_scale_and_bias`/`set_scale_and_bias` against the Python oracle at default tolerance; any CLI-only residual against the CLI oracle at relative 1e-9.
8. Update every matrix row found in step 3 to green with test id(s); for any row left red, record the justification in the report.
9. Run `R CMD INSTALL --preclean .` (redirect to log file, check `$?` directly, never pipe through `tail`) then the new test file. Quote exact output.
10. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:
```toon
task_id: catboost-8z4.64
success: bool
data:
  python_equivalent_confirmed: bool
  get_set_scale_and_bias_implemented: bool
  cli_only_residual_found: bool
  differential_test_passing: bool
  matrix_rows_found: int
  matrix_rows_flipped: int
report_path: docs/phase-5/P5.7-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] Python oracle `get_scale_and_bias`/`set_scale_and_bias` signatures quoted in report; CLI `normalize-model` --help output quoted in report.
- [ ] Own matrix search performed and its count quoted, not taken from this ticket's text.
- [ ] `catboost.get_scale_and_bias`/`catboost.set_scale_and_bias` implemented and matched against Python oracle.
- [ ] Any CLI-only residual implemented or explicitly documented as a design gap, not silently skipped.
- [ ] Differential tests pass at declared tolerances; every matrix row found flipped to green or explicitly justified if left red.
- [ ] git -C vendor/catboost status --porcelain empty.
- [ ] Report written with exact command output.


