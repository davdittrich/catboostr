# Phase 7: Parity-cleanup follow-ups — implementation plan (iteration 2)

Epic: catboost-8z4. Base branch: phase-0 (c102350, Phase 6 merge).

## Global Constraints

- **Mechanism:** direct-fix tickets, follow-ups filed during/after Phase 6's execution/review. No new R public API surface, no new native entry points.
- **Forbidden:** no touching vendor/catboost/ (read-only); no fabricated matrix rows (spec §4.4); no re-litigating the pre-existing 770 rows' states; no bundling method/mode-shaped rows into one ticket (spec §4.5 — the original catboost-8z4.72 violated this and was split into 7 capability-themed tickets: .76-.82, closed and superseded).
- **Dependency:** catboost-8z4.76 through .82 (all 8 row-closure tickets) each depend on catboost-8z4.73 (bd blocks edges) — .73's chosen fix direction (option a: skip closed rows in the self-consistency test, or option b: introduce a closure-overlay file) determines how each closure ticket should record its row state changes. Execute .73 first, before any of .76-.82.
- **Verification method is task-specific, per what each task actually touches:**
  - catboost-8z4.73 (Python-only, tools/parity/*.py): tools/parity's own Python unit test suite green. No R rebuild required.
  - catboost-8z4.74 (src/init.c and/or src/catboostr.exports, C/C++): R CMD INSTALL --preclean . clean + full testthat suite green.
  - catboost-8z4.75 (R/catboost.R): R CMD INSTALL --preclean . clean + full testthat suite green.
  - catboost-8z4.76 through .82 (R test files + JSON matrix fixture): both R CMD INSTALL --preclean . + full testthat suite green AND tools/parity's Python unit test suite green.
- vendor/catboost/ untouched in all 10 tasks.

## Execution Order

1. catboost-8z4.73 — fix/retire the stale self-consistency test (design decision, Alternatives Considered required; blocks all of .76-.82)
2. catboost-8z4.74 — CatBoostPoolNumTrees_R exports/init.c registration mismatch (independent)
3. catboost-8z4.75 — catboost.save_model toJSON digits=4 (independent; note catboost-8z4.82 checks for interaction with this)
4. catboost-8z4.76 — flag:--cv (bulk-eligible family, trivial)
5. catboost-8z4.77 — mode:eval-metrics
6. catboost-8z4.78 — training entry points (catboost.cv/sum_models/train)
7. catboost-8z4.79 — eval_metrics method
8. catboost-8z4.80 — feature introspection (drop_unused_features/get_feature_importance/get_object_importance)
9. catboost-8z4.81 — prediction family (predict/staged_predict/virtual_ensembles_predict/shrink)
10. catboost-8z4.82 — persistence (load_model/save_model)

## Task Tickets (verbatim)

### catboost-8z4.73

# catboost-8z4.NN: Fix or retire test_apply_disposition.py's stale self-consistency assertion
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** `tools/parity/test_apply_disposition.py::test_regenerated_output_matches_committed_matrix_dispositioned_json` asserts a raw `apply_disposition()` rerun exactly reproduces the committed `matrix.dispositioned.json`. This has been false since the first hand-patch closed a row (Phase 3 era, commit `d8af454`/`5b668c8`), because closure work stamps `state: "green"`/`test_id` directly onto the committed file, outside the `compute_diff -> build_matrix -> apply_disposition` pipeline. Confirmed pre-existing (fails identically on `phase6-parity-debt-cleanup`'s base commit `b5041e8`, before catboost-8z4.66's pipeline fix), independent of that ticket's change. Pick and implement one fix direction: (a) change the test to only compare rows still in `red`/undispositioned status against a fresh rebuild (skip closed rows), or (b) introduce an explicit "closure overlay" file that closing tickets write to instead of hand-editing `matrix.dispositioned.json` directly, with the pipeline applying it as a final merge step — this would also make future pipeline reruns mechanically safe instead of requiring a manual by-id merge (as catboost-8z4.66 had to do by hand).
* **Why:** Found during catboost-8z4.66's task review; a real, currently-failing test in the repo (not a false-positive) that gives no actionable signal since it's structurally guaranteed to fail once any row is ever hand-closed. Filed as its own ticket since picking a fix direction is a design decision warranting its own "Alternatives Considered" evaluation, not something to bolt onto an unrelated diff.
* **Reference Data:** `tools/parity/test_apply_disposition.py` (the failing assertion), `tools/parity/apply_disposition.py`, `tests/fixtures/parity/matrix.dispositioned.json` (770 rows, closure history spans Phase 3-6).

## II. Input Specification
* **Expected Input:** `tools/parity/test_apply_disposition.py`, `tools/parity/apply_disposition.py`.
* **Format:** Python source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Alternatives** | Must evaluate both (a) and (b) above (or a third option if found) before picking one — this is a design decision per CLAUDE.md's Architectural Integrity gate, not a mechanical fix. |
| **No weakening** | The fixed test must still catch a genuine pipeline regression (e.g. a row that should still be red getting silently dropped or corrupted) — do not simply delete or vacuously pass the test. |
| **Scope** | Only this test and, if option (b) is chosen, the minimal pipeline change to support a closure-overlay file. Do not re-litigate or touch the 770 rows' existing states. |

## IV. Step-by-Step Logic
1. Read the failing test and `apply_disposition.py` in full; confirm the failure mechanism matches this ticket's Objective.
2. Evaluate options (a) and (b): which better serves future pipeline reruns (like catboost-8z4.66's) staying mechanically safe without manual by-id merges?
3. Implement the chosen option.
4. Verify the fixed test genuinely catches regressions: temporarily reintroduce a synthetic drop/corruption and confirm the test fails, then revert and confirm it passes.
5. Run the full `tools/parity` Python unit test suite; confirm green.

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  option_chosen: string
  regression_still_caught: bool
error_log: null | msg
```

## VI. Definition of Done
- [ ] Test no longer fails on the current committed matrix state.
- [ ] Test still catches a genuine synthetic regression (verified, not assumed).
- [ ] Full Python parity-tooling suite green.

### catboost-8z4.74

# catboost-8z4.NN: Resolve CatBoostPoolNumTrees_R's exports/init.c registration mismatch
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** `src/catboostr.exports` declares `C CatBoostPoolNumTrees_R` (alphabetical position among the `CatBoostPool*` block), but `src/init.c`'s `CallEntries[]` `R_CallMethodDef` table has no corresponding `{"CatBoostPoolNumTrees_R", (DL_FUNC) &..., n}` row — the opposite-direction mismatch from catboost-8z4.68's `.exports` sync (which only added missing-from-.exports entries, correctly leaving this direction untouched). Determine whether (a) the native C++ function `CatBoostPoolNumTrees_R` still exists in `src/catboostr.cpp` and should be registered in `init.c` (real gap, register it), or (b) the function was removed/renamed and the `.exports` line is a stale dead entry (real gap, delete it from `.exports`).
* **Why:** Found during catboost-8z4.68's task review (Phase 6 plan), confirmed as the ONLY such opposite-direction mismatch (verified via `comm -13` on the full init.c/.exports symbol sets). R's dynamic symbol lookup for an unregistered `.Call` name may still resolve via unregistered/dynamic lookup depending on `useDynamicSymbols`, but this is exactly the kind of inconsistency `R_CallMethodDef` registration exists to eliminate — a real registration bug, low priority (doesn't currently break the test suite).
* **Reference Data:** `src/catboostr.exports` (the declaring line), `src/init.c`'s `CallEntries[]` table, `src/catboostr.cpp` (check for a `CatBoostPoolNumTrees_R` definition to decide which side to fix).

## II. Input Specification
* **Expected Input:** `src/catboostr.exports`, `src/init.c`, `src/catboostr.cpp`.
* **Format:** C/C++ source, plain text.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Read `src/catboostr.cpp` first to confirm whether `CatBoostPoolNumTrees_R` is a real, currently-implemented native function before deciding which side to fix. |
| **Scope** | Only this one symbol's mismatch. Do not re-audit the rest of init.c/.exports (already done, clean, in catboost-8z4.68). |
| **Regression** | If registering in init.c, verify no signature/arity mismatch with the actual C++ function definition. If deleting from .exports, verify no other code path depends on the unregistered symbol being listed there. |

## IV. Step-by-Step Logic
1. Read `src/catboostr.cpp` for `CatBoostPoolNumTrees_R`'s definition (signature, arity).
2. If it exists and is a real, callable native function: add its `R_CallMethodDef` entry to `init.c`'s `CallEntries[]` table, matching the existing entries' format and correct arg count.
3. If it does not exist (renamed/removed): delete the `C CatBoostPoolNumTrees_R` line from `src/catboostr.exports`.
4. Run `R CMD INSTALL --preclean .` (redirect log, check `$?` and log tail directly) and the full `testthat` suite; confirm green (0 new FAIL/Error, same pre-existing skip count).

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  resolution: string
error_log: null | msg
```

## VI. Definition of Done
- [ ] `CatBoostPoolNumTrees_R` is either correctly registered in init.c's CallEntries[] table (matching a real cpp definition), or removed from .exports (if no longer real).
- [ ] `R CMD INSTALL --preclean .` succeeds, full suite green.

### catboost-8z4.75

# catboost-8z4.NN: catboost.save_model's export-parameters JSON uses jsonlite's default digits=4, same defect class as catboost-8z4.65
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** `R/catboost.R:3672`, `catboost.save_model`'s `jsonlite::toJSON(export_parameters, auto_unbox = TRUE)` call has no explicit `digits` argument, so jsonlite's default (`digits = 4`) applies — worse than the `digits = 10` truncation that catboost-8z4.65 just fixed on `prepare_train_export_parameters`, and inconsistent with that site's now-fixed `digits = NA`. Change to `digits = NA` for consistency and to close this instance of the same defect class.
* **Why:** Found during Phase 6's final whole-branch review (opus). Today's CoreML/PMML export parameters are all strings, so nothing currently truncates in practice — but this is now the only inconsistent JSON-precision site among the three (`prepare_train_export_parameters`, `prepare_grid_json`, and this one), and any future numeric export parameter would silently truncate to 4 significant digits. Filed per the no-bundling rule rather than folded into the whole-branch review's own fix wave (which was scoped to documentation-only fixes).
* **Reference Data:** `R/catboost.R:3672` (verify exact line fresh — may have shifted). `prepare_train_export_parameters` (catboost-8z4.65) and `prepare_grid_json` (Phase 5) both already use `digits = NA` as the established pattern.

## II. Input Specification
* **Expected Input:** `R/catboost.R`'s `catboost.save_model` function.
* **Format:** R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Scope** | Only this one `toJSON` call. Do not audit other `toJSON` sites in this ticket (grep for a definitive list if suspected, but fix only this one; file further tickets if more found). |
| **Regression test** | Add a regression test proving a numeric export parameter with >4 significant digits round-trips without truncation (currently all export parameters are strings, so this test may need to add a synthetic numeric parameter or verify via a lower-level unit test of the serialization call itself, not necessarily a full `catboost.save_model` round trip if the export parameter schema doesn't support numeric values today — check first). |

## IV. Step-by-Step Logic
1. Read `R/catboost.R`'s current `catboost.save_model`, confirm the `toJSON` call and its context.
2. Change `jsonlite::toJSON(export_parameters, auto_unbox = TRUE)` to add `digits = NA`.
3. Add a regression test per the Constraints guard above.
4. Run `R CMD INSTALL --preclean .` and the full test suite; confirm green.

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  digits_fixed: bool
  regression_test_added: bool
error_log: null | msg
```

## VI. Definition of Done
- [ ] `digits = NA` applied to the `catboost.save_model` toJSON call.
- [ ] Regression test proves no truncation.
- [ ] Full suite green.

### catboost-8z4.76

# catboost-8z4.NN: Close the flag:--cv parity matrix row (bulk parameter/flag family disposition)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** catboost-8z4.66's pipeline fix surfaced 45 previously-dropped rows in `tests/fixtures/parity/matrix.dispositioned.json` (725 -> 770). One of them, `flag:--cv`, is `kind: parameter_or_flag` and already mechanically classified (not a method/mode row) — per spec §4.5, parameter/flag rows are dispositioned as a family, one differential test per family, not per-name. Determine whether the existing CLI `--cv` flag / R's `catboost.cv` already has family-level test coverage (it likely does, since `catboost.cv` itself is closed separately in catboost-8z4.78 — check for overlap) and either sync `test_id` or write the one differential test this family needs.
* **Why:** Split from the original bundled catboost-8z4.72 follow-up per spec §4.5 ("parameter/flag rows... dispositioned as a family... do not become per-name tickets" — this genuinely qualifies, unlike the method/mode rows split into sibling tickets catboost-8z4.77 through .82).
* **Reference Data:** `tests/fixtures/parity/matrix.dispositioned.json` (770 rows; find the `flag:--cv` row by `inventory_row_id`). `tools/parity/apply_disposition.py`'s `parameter_or_flag` classification logic (already correctly auto-classified this row's `kind`).

## II. Input Specification
* **Expected Input:** `tests/fixtures/parity/matrix.dispositioned.json`, existing `tests/testthat/*.R` (especially any covering `catboost.cv`).
* **Format:** JSON, R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **No fabrication** | Per spec §4.4, only mark green with a real passing differential test backing it. |
| **Dependency** | This ticket depends on catboost-8z4.73 (fix/retire the stale self-consistency test) — .73's chosen closure-recording mechanism (direct hand-edit vs. closure-overlay file) determines how this row's closure should be written. Execute .73 first. |
| **Scope** | Only this one row. Do not touch other rows' state/test_id. |

## IV. Step-by-Step Logic
1. Read the `flag:--cv` row's current state in `matrix.dispositioned.json`.
2. Check `tests/testthat/` for existing `catboost.cv` coverage that would also validate `--cv`'s behavior; if adequate, sync `test_id`.
3. If not adequate, write one differential test (Python/CLI oracle vs R) for this family.
4. Flip `state` to `"green"`, set `test_id`, using catboost-8z4.73's chosen closure mechanism.
5. Run full R suite + `tools/parity` Python suite; confirm green.

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  row_closed: bool
  test_id: string
error_log: null | msg
```

## VI. Definition of Done
- [ ] `flag:--cv` row is green with a real passing differential test, or documented red reason if genuinely un-closeable.
- [ ] No other rows touched.
- [ ] Full suite green.

### catboost-8z4.77

# catboost-8z4.NN: Close the mode:eval-metrics parity matrix row (CLI eval-metrics mode parity)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** catboost-8z4.66's pipeline fix surfaced 45 previously-dropped rows in `tests/fixtures/parity/matrix.dispositioned.json` (725 -> 770). One is `mode:eval-metrics` — a CLI-mode-shaped capability, per spec §4.5 "the genuine feature work" requiring its own per-capability ticket (not bundled with the others). Write a differential test proving the CLI oracle's `eval-metrics` mode and R's equivalent (`catboost.eval_metrics`, if it exists — check `R/catboost.R`) produce matching output, or document why it cannot be closed.
* **Why:** Split from the originally-bundled catboost-8z4.72 per spec §4.5's explicit rule that method/mode-shaped rows must not be bulk-dispositioned together — found during this batch's plan-review-gate (Scope & Alignment reviewer).
* **Reference Data:** `tests/fixtures/parity/matrix.dispositioned.json` (770 rows; find `mode:eval-metrics` by `inventory_row_id`). `tests/fixtures/parity/disposition_judgments.json` (330 existing P2.3 judgments — this row needs an output-type judgment, elementwise/structural/other, added here, matching the existing pattern). Prior CLI-mode-parity precedent: catboost-8z4.55/.56/.57 (Phase 4, `P4.6/.7/.8: CLI <mode> mode parity`) for the expected ticket shape and test pattern.

## II. Input Specification
* **Expected Input:** `tests/fixtures/parity/matrix.dispositioned.json`, `tests/fixtures/parity/disposition_judgments.json`, `R/catboost.R`, CLI oracle tooling under `tools/oracle/cli/`.
* **Format:** JSON, R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **No fabrication** | Per spec §4.4, only mark green with a real passing differential test backing it; if genuinely no R/CLI equivalent exists or it's blocked (e.g. GPU-only), document the reason and leave red. |
| **Dependency** | This ticket depends on catboost-8z4.73 — execute .73 first; use its chosen closure-recording mechanism. |
| **Scope** | Only this one row. Do not touch other rows' state/test_id. |

## IV. Step-by-Step Logic
1. Read the `mode:eval-metrics` row's current state.
2. Add an output-type judgment to `tests/fixtures/parity/disposition_judgments.json`.
3. Check for existing test coverage of the CLI's `eval-metrics` mode; if adequate, sync `test_id`. If not, write a differential test.
4. If genuinely un-closeable, document the reason, leave red.
5. Flip `state` to `"green"` (or leave red with reason), set `test_id`, using catboost-8z4.73's chosen closure mechanism.
6. Run full R suite + `tools/parity` Python suite; confirm green.

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  row_closed: bool
  test_id: string | null
  red_reason: string | null
error_log: null | msg
```

## VI. Definition of Done
- [ ] `mode:eval-metrics` row is green with a real passing differential test, or documented red reason.
- [ ] No other rows touched.
- [ ] Full suite green.

### catboost-8z4.78

# catboost-8z4.NN: Close the training-entry-point parity matrix rows (catboost.cv, catboost.sum_models, catboost.train)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** catboost-8z4.66's pipeline fix surfaced 45 previously-dropped rows in `tests/fixtures/parity/matrix.dispositioned.json` (725 -> 770). Three are top-level training entry points — `catboost.cv`, `catboost.sum_models`, `catboost.train` — a coherent thematic group (core training/model-building surface), matching this project's precedent of grouping closely-related capabilities into one ticket (e.g. catboost-8z4.38's Pool accessor group). For each, check for existing R-side test coverage before writing net-new tests — `catboost.train`/`catboost.cv` almost certainly already have extensive test coverage from every prior phase; this is very likely a `test_id`-syncing task, not new-test-writing.
* **Why:** Split from the originally-bundled catboost-8z4.72 per spec §4.5's rule that method/function-shaped rows must not be bulk-dispositioned with parameter/flag rows — found during this batch's plan-review-gate (Scope & Alignment reviewer).
* **Reference Data:** `tests/fixtures/parity/matrix.dispositioned.json` (770 rows; find the 3 rows by `inventory_row_id`: `catboost.cv`, `catboost.sum_models`, `catboost.train`). `tests/fixtures/parity/disposition_judgments.json` (330 existing judgments — add output-type judgments for these 3 if not already present). `tests/testthat/test_train.R`, `test_cv.R`, or equivalent existing test files — check first.

## II. Input Specification
* **Expected Input:** `tests/fixtures/parity/matrix.dispositioned.json`, `tests/fixtures/parity/disposition_judgments.json`, existing `tests/testthat/*.R`.
* **Format:** JSON, R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **No fabrication** | Per spec §4.4, only mark green with a real passing differential test backing it. |
| **Dependency** | Depends on catboost-8z4.73 — execute .73 first; use its chosen closure-recording mechanism. |
| **Scope** | Only these 3 rows. Do not touch other rows' state/test_id. |
| **Reuse first** | These functions are core, heavily-tested surface from every prior phase — check `tests/testthat/` thoroughly before assuming zero coverage. |

## IV. Step-by-Step Logic
1. Read all 3 rows' current state.
2. Add output-type judgments to `tests/fixtures/parity/disposition_judgments.json` if not already present.
3. For each function, check existing test coverage; sync `test_id` if adequate, else write a differential test.
4. Flip each row's `state` to `"green"`, set `test_id`, using catboost-8z4.73's chosen closure mechanism (or leave red with documented reason if genuinely un-closeable).
5. Run full R suite + `tools/parity` Python suite; confirm green.

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  rows_closed: int
  rows_still_red: int
  red_reasons: [string]
error_log: null | msg
```

## VI. Definition of Done
- [ ] All 3 rows are green with real passing differential tests, or documented red reasons.
- [ ] No other rows touched.
- [ ] Full suite green.

### catboost-8z4.79

# catboost-8z4.NN: Close the eval_metrics method parity matrix rows (4 estimator classes)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** catboost-8z4.66's pipeline fix surfaced 45 previously-dropped rows in `tests/fixtures/parity/matrix.dispositioned.json` (725 -> 770). Four of them are `eval_metrics` on `CatBoost`/`CatBoostClassifier`/`CatBoostRegressor`/`CatBoostRanker` — one method, four classes, a single coherent capability (matching the granularity of prior per-capability tickets like catboost-8z4.51 "catboost.compare parity"). Write a differential test (or one shared test exercising all 4 classes) proving R's `eval_metrics` method matches the Python/CLI oracle.
* **Why:** Split from the originally-bundled catboost-8z4.72 per spec §4.5's rule — found during this batch's plan-review-gate (Scope & Alignment reviewer).
* **Reference Data:** `tests/fixtures/parity/matrix.dispositioned.json` (770 rows; find the 4 `eval_metrics` rows, one per class, by `inventory_row_id`). `tests/fixtures/parity/disposition_judgments.json` (330 existing judgments). Note: distinct from `mode:eval-metrics` (the CLI mode, closed separately in a sibling ticket) — this ticket is the R-method-level `eval_metrics`.

## II. Input Specification
* **Expected Input:** `tests/fixtures/parity/matrix.dispositioned.json`, `tests/fixtures/parity/disposition_judgments.json`, existing `tests/testthat/*.R`.
* **Format:** JSON, R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **No fabrication** | Per spec §4.4, only mark green with a real passing differential test backing it. |
| **Dependency** | Depends on catboost-8z4.73 — execute .73 first; use its chosen closure-recording mechanism. |
| **Scope** | Only these 4 rows. Do not touch other rows' state/test_id. |

## IV. Step-by-Step Logic
1. Read all 4 rows' current state.
2. Add an output-type judgment to `tests/fixtures/parity/disposition_judgments.json` (shared across the 4 classes if the method's output shape is consistent, or per-class if it differs).
3. Check existing test coverage; sync `test_id`s if adequate, else write differential test(s) covering all 4 classes.
4. Flip each row's `state` to `"green"`, set `test_id`, using catboost-8z4.73's chosen closure mechanism (or leave red with documented reason).
5. Run full R suite + `tools/parity` Python suite; confirm green.

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  rows_closed: int
  rows_still_red: int
  red_reasons: [string]
error_log: null | msg
```

## VI. Definition of Done
- [ ] All 4 `eval_metrics` rows are green with real passing differential tests, or documented red reasons.
- [ ] No other rows touched.
- [ ] Full suite green.

### catboost-8z4.80

# catboost-8z4.NN: Close the feature-introspection method parity matrix rows (drop_unused_features, get_feature_importance, get_object_importance)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** catboost-8z4.66's pipeline fix surfaced 45 previously-dropped rows in `tests/fixtures/parity/matrix.dispositioned.json` (725 -> 770). Twelve of them are `drop_unused_features`, `get_feature_importance`, `get_object_importance` on each of `CatBoost`/`CatBoostClassifier`/`CatBoostRegressor`/`CatBoostRanker` — three thematically-related feature-introspection methods (matching the grouping precedent of prior tickets like catboost-8z4.39 "Pool feature/shape introspection accessors"). `get_feature_importance` almost certainly already has extensive test coverage from earlier phases (core SHAP/importance surface) — check first.
* **Why:** Split from the originally-bundled catboost-8z4.72 per spec §4.5's rule — found during this batch's plan-review-gate (Scope & Alignment reviewer).
* **Reference Data:** `tests/fixtures/parity/matrix.dispositioned.json` (770 rows; find the 12 rows: 3 methods x 4 classes, by `inventory_row_id`). `tests/fixtures/parity/disposition_judgments.json` (330 existing judgments).

## II. Input Specification
* **Expected Input:** `tests/fixtures/parity/matrix.dispositioned.json`, `tests/fixtures/parity/disposition_judgments.json`, existing `tests/testthat/*.R`.
* **Format:** JSON, R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **No fabrication** | Per spec §4.4, only mark green with a real passing differential test backing it. |
| **Dependency** | Depends on catboost-8z4.73 — execute .73 first; use its chosen closure-recording mechanism. |
| **Scope** | Only these 12 rows. Do not touch other rows' state/test_id. |
| **Reuse first** | `get_feature_importance` is core SHAP/importance surface — check `tests/testthat/` thoroughly before assuming zero coverage. |

## IV. Step-by-Step Logic
1. Read all 12 rows' current state.
2. Add output-type judgments to `tests/fixtures/parity/disposition_judgments.json` (per method, shared across classes where consistent).
3. For each method, check existing test coverage across all 4 classes; sync `test_id`s if adequate, else write differential tests.
4. Flip each row's `state` to `"green"`, set `test_id`, using catboost-8z4.73's chosen closure mechanism (or leave red with documented reason).
5. Run full R suite + `tools/parity` Python suite; confirm green.

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  rows_closed: int
  rows_still_red: int
  red_reasons: [string]
error_log: null | msg
```

## VI. Definition of Done
- [ ] All 12 rows are green with real passing differential tests, or documented red reasons.
- [ ] No other rows touched.
- [ ] Full suite green.

### catboost-8z4.81

# catboost-8z4.NN: Close the prediction-family method parity matrix rows (predict, staged_predict, virtual_ensembles_predict, shrink)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** catboost-8z4.66's pipeline fix surfaced 45 previously-dropped rows in `tests/fixtures/parity/matrix.dispositioned.json` (725 -> 770). Sixteen of them are `predict`, `staged_predict`, `virtual_ensembles_predict`, `shrink` on each of `CatBoost`/`CatBoostClassifier`/`CatBoostRegressor`/`CatBoostRanker` — a coherent prediction-pipeline theme. `predict` almost certainly already has extensive coverage from every prior phase; `virtual_ensembles_predict` was already differentially tested against the Python oracle in Phase 5 (catboost-8z4.61) — check whether these rows just need `test_id` syncing to that existing test rather than new tests.
* **Why:** Split from the originally-bundled catboost-8z4.72 per spec §4.5's rule — found during this batch's plan-review-gate (Scope & Alignment reviewer).
* **Reference Data:** `tests/fixtures/parity/matrix.dispositioned.json` (770 rows; find the 16 rows: 4 methods x 4 classes, by `inventory_row_id`). `tests/fixtures/parity/disposition_judgments.json` (330 existing judgments). Phase 5's `catboost-8z4.61` differential test for `virtual_ensembles_predict` (find via `bd show catboost-8z4.61` and its commit/test file) — this is very likely the existing coverage these 4 rows just need synced to.

## II. Input Specification
* **Expected Input:** `tests/fixtures/parity/matrix.dispositioned.json`, `tests/fixtures/parity/disposition_judgments.json`, existing `tests/testthat/*.R`.
* **Format:** JSON, R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **No fabrication** | Per spec §4.4, only mark green with a real passing differential test backing it. |
| **Dependency** | Depends on catboost-8z4.73 — execute .73 first; use its chosen closure-recording mechanism. |
| **Scope** | Only these 16 rows. Do not touch other rows' state/test_id. |
| **Reuse first** | `predict` is core surface, `virtual_ensembles_predict` likely already covered by catboost-8z4.61 — check `tests/testthat/` thoroughly before assuming zero coverage. |

## IV. Step-by-Step Logic
1. Read all 16 rows' current state.
2. Add output-type judgments to `tests/fixtures/parity/disposition_judgments.json` (per method, shared across classes where consistent).
3. For each method, check existing test coverage across all 4 classes (especially catboost-8z4.61's virtual_ensembles_predict test); sync `test_id`s if adequate, else write differential tests.
4. Flip each row's `state` to `"green"`, set `test_id`, using catboost-8z4.73's chosen closure mechanism (or leave red with documented reason).
5. Run full R suite + `tools/parity` Python suite; confirm green.

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  rows_closed: int
  rows_still_red: int
  red_reasons: [string]
error_log: null | msg
```

## VI. Definition of Done
- [ ] All 16 rows are green with real passing differential tests, or documented red reasons.
- [ ] No other rows touched.
- [ ] Full suite green.

### catboost-8z4.82

# catboost-8z4.NN: Close the model-persistence method parity matrix rows (load_model, save_model)
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** catboost-8z4.66's pipeline fix surfaced 45 previously-dropped rows in `tests/fixtures/parity/matrix.dispositioned.json` (725 -> 770). Eight of them are `load_model`, `save_model` on each of `CatBoost`/`CatBoostClassifier`/`CatBoostRegressor`/`CatBoostRanker` — a coherent model-persistence theme. `save_model`/`load_model` are core, heavily-used surface — check for existing test coverage before writing net-new tests. Note: catboost-8z4.75 (a sibling ticket in this same batch) fixes a `digits=4` JSON precision defect in `catboost.save_model`'s export-parameters serialization — if catboost-8z4.75 lands first, verify this ticket's closure test doesn't rely on the pre-fix truncated behavior.
* **Why:** Split from the originally-bundled catboost-8z4.72 per spec §4.5's rule — found during this batch's plan-review-gate (Scope & Alignment reviewer).
* **Reference Data:** `tests/fixtures/parity/matrix.dispositioned.json` (770 rows; find the 8 rows: 2 methods x 4 classes, by `inventory_row_id`). `tests/fixtures/parity/disposition_judgments.json` (330 existing judgments).

## II. Input Specification
* **Expected Input:** `tests/fixtures/parity/matrix.dispositioned.json`, `tests/fixtures/parity/disposition_judgments.json`, existing `tests/testthat/*.R`.
* **Format:** JSON, R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **No fabrication** | Per spec §4.4, only mark green with a real passing differential test backing it. |
| **Dependency** | Depends on catboost-8z4.73 — execute .73 first; use its chosen closure-recording mechanism. |
| **Scope** | Only these 8 rows. Do not touch other rows' state/test_id. |
| **Reuse first** | `save_model`/`load_model` are core surface — check `tests/testthat/` thoroughly before assuming zero coverage. |

## IV. Step-by-Step Logic
1. Read all 8 rows' current state.
2. Add output-type judgments to `tests/fixtures/parity/disposition_judgments.json` (per method, shared across classes where consistent).
3. For each method, check existing test coverage across all 4 classes; sync `test_id`s if adequate, else write differential tests.
4. Flip each row's `state` to `"green"`, set `test_id`, using catboost-8z4.73's chosen closure mechanism (or leave red with documented reason).
5. Run full R suite + `tools/parity` Python suite; confirm green.

## V. Output Schema (Strict)
```toon
task_id: [ID]
success: bool
data:
  rows_closed: int
  rows_still_red: int
  red_reasons: [string]
error_log: null | msg
```

## VI. Definition of Done
- [ ] All 8 rows are green with real passing differential tests, or documented red reasons.
- [ ] No other rows touched.
- [ ] Full suite green.

