# Phase 4: Analysis Parity — Implementation Plan

Epic: catboost-8z4. Spec: `docs/superpowers/specs/2026-07-30-catboostr-design.md`,
gate at line 638: "Analysis parity: calc_feature_statistics, object-importance
MultiClass fix (#869), plot_tree, catboost.compare, ShapInteractionValues/
PredictionDiff argument values (§4.2 — not new functions). Includes CLI's
eval-feature, roc, model-based-eval modes." Gate condition: "Differential
tests green; structural method per §4.3."

Approved by plan-review-gate 2026-08-03 (3 iterations; Feasibility PASS all
3, Completeness PASS all 3, Scope & Alignment FAIL/FAIL/PASS — two rounds of
scope-creep corrections applied and re-verified). Each task below is the
authoritative source for its own beads ticket (`bd show <id>`); the ticket
and this plan file must stay in sync — the ticket is canonical.

## Global Constraints

- **Read-only vendor pin.** Zero writes under `vendor/catboost/` in any task.
  Verify with `git -C vendor/catboost status --porcelain` before and after —
  must be empty both times.
- **No core patches.** Upstream C++ source is never patched locally. Anything
  needing a core change is out of scope / filed upstream, never worked around
  by editing `vendor/catboost/`.
- **Append-only R API.** Existing exported function signatures never change
  in an incompatible way — new arguments get defaults that preserve existing
  call sites (matches Phase 3 convention).
- **§4.2 mapping rule.** Enum members become accepted *values* of an existing
  argument, never new exported functions. CLI modes and Python methods become
  a single `catboost.*` function (snake_case `verb_object`, never a dotted
  S3-colliding name like `model.compare`).
- **§4.3 verification.** Every capability is proven by a differential test
  against the pinned oracle (Python: relative tolerance `1e-12` default; CLI:
  relative tolerance `1e-9` default, never bit-exact, any override needs a
  first-principles justification recorded in the matrix). Structural outputs
  (`plot_tree`, `calc_feature_statistics`) compare canonical-JSON field-by-
  field per the Phase 2 serializer convention in `tools/parity/`; rendering
  gets only a smoke test for return-value class, never a pixel comparison.
  Error paths assert on error class and message.
- **Parity matrix.** `tests/fixtures/parity/matrix.dispositioned.json` (725
  rows) is updated to `state: green` with a `test_id` for every row a task
  closes. Do not flip a row green without a passing differential test backing
  it.
- **Test convention.** New tests go in `tests/testthat/test_<capability>.R`,
  following the differential-test-against-oracle pattern already used by
  every closed Phase 1-3 ticket.
- **Build/verify command.** `R CMD INSTALL --preclean .` then run the new
  test file; quote exact output in the task report.
- **Isolation.** This plan executes entirely inside the
  `.worktrees/phase-4-analysis-parity` worktree (branch
  `phase-4-analysis-parity`, based on `phase-0`). Each task commits its own
  work on this branch.
- **No scope creep (gate-enforced).** Each task's own "Boundary" guard lists
  what it must NOT do — these were tightened once already by the plan-review
  gate (P4.5 dropped MultiRMSE/MultiLogloss; P4.6 dropped
  get_fnr_curve/get_fpr_curve/select_threshold). Do not re-expand scope
  during implementation without stopping to ask.

## Task 1: P4.1: calc_feature_statistics parity (catboost-8z4.50)

_Beads ticket: `bd show catboost-8z4.50`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._


**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add `catboost.calc_feature_statistics` (dispatching over CatBoost/CatBoostClassifier/CatBoostRegressor/CatBoostRanker equivalents), verified by structural differential test against the pinned Python oracle.
* **Why:** Phase 4 gate (spec line 638, "Analysis parity"). Matrix rows `CatBoost.calc_feature_statistics`, `CatBoostClassifier.calc_feature_statistics`, `CatBoostRegressor.calc_feature_statistics`, `CatBoostRanker.calc_feature_statistics` — all `method_mode_shaped`, state `red`. No R equivalent exists today (grep `R/catboost.R` for `calc_feature_statistics` returns zero matches, checked 2026-08-03).
* **Reference Data:** §4.3's structural-output verification method applies: both sides serialize to a canonical JSON form with a field mapping declared in the parity matrix (`tests/fixtures/parity/matrix.dispositioned.json`); comparison is field-by-field, numeric leaves within tolerance. Do not attempt bit-exact comparison on plot/rendering fields — only the underlying statistics tensor. `catboost.utils.eval_metric` (R/catboost.R:3387 area, `catboost.eval_metrics`) is the closest existing precedent for a Pool+model bound analysis function — read it first for the established R calling convention (model, pool, ... args).
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Python oracle's `CatBoost.calc_feature_statistics` docstring/signature (via `uv run --frozen --project tools/oracle python -c "help(...)"`), `R/catboost.R`, `src/catboostr.cpp`, `tests/fixtures/parity/matrix.dispositioned.json`.
* **Format:** Python introspection output, C++/R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under `vendor/catboost/`. Verify `git -C vendor/catboost status --porcelain` empty before and after. |
| **Logic** | One R function, S3-dispatch-free (matches upstream: bound method on all four Python classes maps to a single `catboost.calc_feature_statistics(model, pool, ...)` per §4.2's mapping table), covering all 4 matrix rows. |
| **Serializer** | Structural output must use the Phase 2 canonical-JSON serializer/mapping convention (`tools/parity/`) — do not invent a second serialization scheme. |
| **Boundary** | Do not implement `plot_tree`, `catboost.compare`, or any other Phase 4 row — those are separate tickets (P4.2, P4.3). |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Introspect Python's `calc_feature_statistics` signature and return shape via the pinned oracle; quote exact output.
2. Design the R function signature per §4.2's mapping rule; check whether new C++ glue is needed or whether existing entry points (grep `src/catboostr.cpp`) already expose the required per-feature binning/statistics data.
3. Implement `catboost.calc_feature_statistics` in `R/catboost.R` (plus new `.Call` entry point in `src/catboostr.cpp`/cpp11 if required).
4. Write a structural differential test in `tests/testthat/test_calc_feature_statistics.R` comparing canonical-JSON serialization against the Python oracle fixture, field-by-field, numeric leaves within the declared tolerance; add a rendering smoke test asserting the return value is of the expected class.
5. Update `tests/fixtures/parity/matrix.dispositioned.json` state for the 4 rows to `green` with the test id.
6. Run `R CMD INSTALL --preclean .` then the new test file. Quote exact output.
7. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:

```toon
task_id: catboost-8z4.P4.1
success: bool
data:
  function_implemented: bool
  matrix_rows_green: int
  matrix_rows_total: 4
  new_native_entry_point: bool
report_path: docs/phase-4/P4.1-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] `catboost.calc_feature_statistics` exists, dispatches over all 4 model classes.
- [ ] Structural differential test passes against Python oracle; matrix rows flipped to green.
- [ ] `git -C vendor/catboost status --porcelain` empty.
- [ ] Report written with exact command output.

---

## Task 2: P4.2: catboost.compare parity (catboost-8z4.51)

_Beads ticket: `bd show catboost-8z4.51`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._


**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add `catboost.compare(model, other_model, ...)`, verified by structural/rendering differential test against the pinned Python oracle.
* **Why:** Phase 4 gate (spec line 638). Matrix rows `CatBoost.compare`, `CatBoostClassifier.compare`, `CatBoostRegressor.compare`, `CatBoostRanker.compare` — all `method_mode_shaped`, state `red`.
* **Reference Data:** §4.2's mapping table is explicit: `catboost.compare(model, other, ...)`, **not** `model.compare` — dotted names collide with R's S3 dispatch already used by `predict.catboost.Model`. Python's `CatBoost.compare` opens an interactive metrics-comparison widget (Jupyter-only visualization); per §4.3's "Structural" row, the comparable, testable surface is the underlying metrics-diff data structure, not the widget rendering. Rendering (if any headless R output exists) gets only a smoke test asserting it runs and returns an object of the expected class — never a pixel/visual comparison.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Python oracle's `CatBoost.compare` signature/return introspection, `R/catboost.R`, `tests/fixtures/parity/matrix.dispositioned.json`.
* **Format:** Python introspection output, R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under `vendor/catboost/`. Verify before/after with `git -C vendor/catboost status --porcelain`. |
| **Logic** | One `catboost.compare` function (never `model.compare`), one row-family, all 4 matrix rows. |
| **Boundary** | Do not attempt to replicate the interactive Jupyter widget — only the underlying comparable metrics data structure, per §4.3. Do not implement P4.1/P4.3 rows. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Introspect Python's `CatBoost.compare` via pinned oracle; determine what data (not widget rendering) it actually computes and returns/exposes; quote exact output.
2. Implement `catboost.compare(model, other, ...)` in `R/catboost.R`, returning the comparable metrics-diff structure.
3. Write a structural differential test in `tests/testthat/test_compare.R`, canonical-JSON field comparison against Python oracle fixture within declared tolerance.
4. Update matrix disposition to `green` for the 4 rows.
5. Run `R CMD INSTALL --preclean .` then the new test file. Quote exact output.
6. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:

```toon
task_id: catboost-8z4.P4.2
success: bool
data:
  function_implemented: bool
  matrix_rows_green: int
  matrix_rows_total: 4
report_path: docs/phase-4/P4.2-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] `catboost.compare` exists as a plain function (not S3 method).
- [ ] Structural differential test passes; matrix rows flipped to green.
- [ ] `git -C vendor/catboost status --porcelain` empty.
- [ ] Report written with exact command output.

---

## Task 3: P4.3: plot_tree parity (catboost-8z4.52)

_Beads ticket: `bd show catboost-8z4.52`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._


**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add `catboost.plot_tree(model, tree_idx, pool = NULL)`, verified by structural differential test against the pinned Python oracle plus a rendering smoke test.
* **Why:** Phase 4 gate (spec line 638). Matrix rows `CatBoost.plot_tree`, `CatBoostClassifier.plot_tree`, `CatBoostRegressor.plot_tree`, `CatBoostRanker.plot_tree` — all `method_mode_shaped`, state `red`.
* **Reference Data:** Python's `plot_tree` returns a `graphviz.Digraph` (nodes/edges/attrs, no rendering performed unless `.render()`/`.view()` called). §4.3 Structural row: both sides serialize to canonical JSON with a declared field mapping (node id, split condition, leaf value, edge labels) — that JSON is the differential-test target, not the rendered image. R side: check installed dependencies first (ponytail ladder rung 4/5) — no `DiagrammeR`/graphviz R binding currently used anywhere in `R/` (confirmed 2026-08-03, zero matches). Only add a new dependency if the DOT-text/graph-object construction cannot be done with base R string building; the JSON differential test does not require any rendering library at all, so default to no new dependency and construct the DOT/graph-object with base R, adding a library only if the "object of the expected class" DoD line genuinely requires one.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Python oracle's `plot_tree` output (`graphviz.Digraph.source` DOT text, node/edge structure), `R/catboost.R`, `DESCRIPTION` (current `Imports`/`Suggests`).
* **Format:** Python introspection output, DOT text, R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under `vendor/catboost/`. |
| **Logic** | One `catboost.plot_tree` function, all 4 matrix rows. |
| **Dependency** | No new R package dependency unless base R cannot construct the returned graph object — justify in the report if one is added. |
| **Boundary** | Do not implement P4.1/P4.2 rows. No pixel/visual rendering comparison — structural JSON only, per §4.3. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Introspect Python's `plot_tree` output structure (node/edge/attr shape) via pinned oracle for a small fitted model; quote exact DOT text.
2. Check `DESCRIPTION` and installed R packages for an existing graph-object capability before adding any dependency.
3. Implement `catboost.plot_tree` in `R/catboost.R`, extracting split/leaf structure from the model (reuse any existing tree-structure introspection glue in `src/catboostr.cpp` — grep first) and building the graph object.
4. Write a structural differential test in `tests/testthat/test_plot_tree.R`: canonical-JSON field comparison (node/edge/split/leaf data) against Python oracle fixture, plus a smoke test asserting the return value is of the expected class.
5. Update matrix disposition to `green` for the 4 rows.
6. Run `R CMD INSTALL --preclean .` then the new test file. Quote exact output.
7. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:

```toon
task_id: catboost-8z4.P4.3
success: bool
data:
  function_implemented: bool
  new_dependency_added: bool
  matrix_rows_green: int
  matrix_rows_total: 4
report_path: docs/phase-4/P4.3-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] `catboost.plot_tree` exists, dispatches over all 4 model classes.
- [ ] Structural differential test passes; rendering smoke test passes; matrix rows flipped to green.
- [ ] `git -C vendor/catboost status --porcelain` empty.
- [ ] Report written with exact command output.

---

## Task 4: P4.4: ShapInteractionValues / PredictionDiff argument-value parity on catboost.get_feature_importance (catboost-8z4.53)

_Beads ticket: `bd show catboost-8z4.53`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._


**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Confirm/extend `catboost.get_feature_importance`'s existing `type`/`fstr_type` argument (R/catboost.R:3130) to accept `"ShapInteractionValues"` and `"PredictionDiff"` as values, each verified by differential test.
* **Why:** Phase 4 gate (spec line 638, "argument values, §4.2 — not new functions"). Matrix rows `catboost.EFstrType.ShapInteractionValues`, `catboost.EFstrType.PredictionDiff` — `enum_member_attachment`, state `red`. Per §4.2's mapping table, an enum member becomes "an accepted value of an existing argument", explicitly **not** a new export — this ticket must not add new top-level functions.
* **Reference Data:** `catboost.get_feature_importance` already exists (R/catboost.R:3130) with a `type`/`fstr_type` argument. Read it fully first to establish which `EFstrType` values it currently accepts/rejects, and whether the existing C++ glue (`src/catboostr.cpp`) already supports `ShapInteractionValues`/`PredictionDiff` at the native level (these are known upstream fstr types — check `CalcFstr`/equivalent native entry point signatures) or whether new native plumbing is needed. `PredictionDiff` requires a second reference pool/object argument (a diff between two predictions) — confirm this from Python's docstring before assuming the existing single-pool call shape suffices.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** `R/catboost.R:3130` (`catboost.get_feature_importance`), `src/catboostr.cpp` (fstr native entry points), Python oracle's `get_feature_importance` docstring for both enum values.
* **Format:** R/C++ source, Python introspection output, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under `vendor/catboost/`. |
| **Logic** | No new exported function/argument name — only extend accepted values of the existing `type` argument, per §4.2. |
| **Boundary** | Do not touch P4.5 (`get_object_importance`) or any other Phase 4 ticket's function. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Read `R/catboost.R:3130` fully; enumerate currently accepted `type` values; quote them.
2. Introspect Python's `get_feature_importance(type="ShapInteractionValues")` and `type="PredictionDiff"` call shapes and return shapes via pinned oracle; quote exact output, including any extra required argument for `PredictionDiff`.
3. Wire native support if missing (grep `src/catboostr.cpp` for existing fstr entry points first — reuse before adding).
4. Extend `catboost.get_feature_importance`'s argument validation/dispatch to accept both values, adding any required extra argument for `PredictionDiff` with a default that preserves existing call signatures for other `type` values (append-only, per prior phase-3 convention).
5. Write differential tests in `tests/testthat/test_fstr_shap_prediction_diff.R` against Python oracle, numeric tolerance per §4.3 defaults.
6. Update matrix disposition to `green` for both rows.
7. Run `R CMD INSTALL --preclean .` then the new test file. Quote exact output.
8. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:

```toon
task_id: catboost-8z4.P4.4
success: bool
data:
  shap_interaction_values_supported: bool
  prediction_diff_supported: bool
  new_native_entry_point: bool
  matrix_rows_green: int
  matrix_rows_total: 2
report_path: docs/phase-4/P4.4-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] `catboost.get_feature_importance(type = "ShapInteractionValues")` and `type = "PredictionDiff"` both work, no new exported function.
- [ ] Differential tests pass; matrix rows flipped to green.
- [ ] `git -C vendor/catboost status --porcelain` empty.
- [ ] Report written with exact command output.

---

## Task 5: P4.5: get_object_importance MultiClass error-path parity (#869) (catboost-8z4.54)

_Beads ticket: `bd show catboost-8z4.54`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._


**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Make `catboost.get_object_importance` (R/catboost.R:3219) fail identically to Python on `MultiClass` models — same error class, equivalent message — verified by error-path differential test. This is upstream issue #869, NOT fixable at the R layer.
* **Why:** Phase 4 gate (spec line 638, "object-importance MultiClass fix (#869)" — named item is MultiClass only). P2.4 (`catboost-8z4.37`, closed, `docs/phase-2/P2.4-report.md`) Finding 1 root-caused this: the core `ostr`/object-importance C++ machinery (`catboost/private/libs/documents_importance/ders_helpers.cpp:95`, `GetEvaluateDerivativesFunc`) is hardcoded to a single scalar target/approx per document and has no `MultiClass` case — `CB_ENSURE(false, ...)` fires. This is a core-source limitation; per the epic's fixed decision "Upstream core source never patched locally. Anything needing core changes filed upstream," this ticket does **not** patch `vendor/catboost/`. "Parity" here means: R raises the same class of error as Python for MultiClass — not that the capability starts working. **Out of scope:** P2.4's Finding 2 (`MultiRMSE`/`MultiLogloss` multi-target losses) is a related but textually distinct item — spec line 638 names only "MultiClass fix (#869)", not multi-target losses generally, and Finding 2's failure point (`target.h:316`'s `GetOneDimensionalTarget`) is a different call site than #869's `ders_helpers.cpp:95` switch (P2.4-report.md is explicit that Finding 1, not Finding 2, "matches upstream issue #869's description exactly"). Do not implement Finding 2 in this ticket; if the user wants multi-target-loss error-path parity too, that is a separate ticket to be filed against a new gate item, not read into this one.
* **Reference Data:** Confidence 95 (MultiClass, Finding 1) per P2.4's findings — read the report before starting, it has a quoted repro command and exact traceback line.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** `docs/phase-2/P2.4-report.md` (Finding 1 section only), `R/catboost.R:3219` (`catboost.get_object_importance`), `src/catboostr.cpp` (native entry point + `R_API_BEGIN`/`R_API_END` error routing).
* **Format:** Markdown report, R/C++ source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **No core patch** | Zero writes under `vendor/catboost/`. If the R-level error message/class does not already match Python's on a MultiClass model, fix only the R-side error surfacing (e.g., wrap/translate the native `CB_ENSURE` message), never the C++ `switch`/check. |
| **Logic** | Verify current R behavior first — do not assume it currently crashes/segfaults/silently-wrongs; it may already surface a comparable R error via the existing `R_API_BEGIN`/`R_API_END` try/catch. |
| **Boundary** | MultiClass only. Do not implement Finding 2 (`MultiRMSE`/`MultiLogloss`) — not named in spec line 638, file as a separate ticket if wanted. Do not touch P4.4 (`get_feature_importance`) or any Pool/data-layer code. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Reproduce P2.4 Finding 1's R-side equivalent: fit a `MultiClass` model in R, call `catboost.get_object_importance`, capture the exact current error/crash behavior. Quote it.
2. Compare against Python's exact error class/message (from P2.4-report.md Finding 1) — determine the gap (e.g., R may currently give a generic/opaque error, or crash instead of erroring cleanly).
3. If R's `R_API_BEGIN`/`R_API_END` already surfaces the native `CB_ENSURE` message as a clean R error, add only a differential error-path test — no code change beyond message-text normalization if genuinely mismatched, and only touching the R wrapper.
4. Write an error-path differential test in `tests/testthat/test_object_importance_multiclass.R` per §4.3's "Error paths" row: assert on error class and message equivalence for MultiClass.
5. Update matrix disposition — determine whether a `get_object_importance` MultiClass row already exists in the matrix or needs adding (it was investigation-only in P2.4, not in the original 725-row inventory as a separate row; confirm before creating a new row).
6. Run `R CMD INSTALL --preclean .` then the new test file. Quote exact output.
7. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:

```toon
task_id: catboost-8z4.P4.5
success: bool
data:
  r_error_matched_python_before_change: bool
  code_changed: bool
  error_path_test_passing: bool
report_path: docs/phase-4/P4.5-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] R raises an equivalent error class/message to Python for `MultiClass` on `get_object_importance`.
- [ ] Error-path differential test passes.
- [ ] `git -C vendor/catboost status --porcelain` empty (no core patch).
- [ ] Report written with exact command output, referencing P2.4 Finding 1's confidence-scored finding.

---

## Task 6: P4.6: CLI roc mode parity (catboost.get_roc_curve) (catboost-8z4.55)

_Beads ticket: `bd show catboost-8z4.55`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._


**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add `catboost.get_roc_curve` — the R function wrapping the CLI's `roc` mode, per §4.2's mapping rule ("Python method / CLI mode -> A `catboost.*` function") — verified by differential test against both the Python oracle's `get_roc_curve` and the CLI's `roc` mode oracle.
* **Why:** Phase 4 gate (spec line 638, "Includes CLI's ... roc ... mode" — a single named CLI mode, not the wider `catboost.utils` ROC/threshold family). Matrix row `mode:roc` (`method_mode_shaped`, `red`, oracle `cli`) is the literal gate item. Matrix row `catboost.utils.get_roc_curve` (`method_mode_shaped`, `red`, oracle `python`) computes the identical ROC-curve capability under a different oracle, so implementing one R function naturally satisfies both rows — do not read this as license to implement the two Python-only rows that are a different, unnamed capability from `roc`, `catboost.utils.get_fnr_curve` and `catboost.utils.select_threshold`. Threshold-selection (`select_threshold`) is a materially different computation (a decision-rule pick, not a curve) and is out of scope here.
* **Reference Data:** `catboost.eval_metrics` (R/catboost.R:3387) is the closest existing precedent for prediction/label-driven metric computation — read it first for the established calling convention.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** Python oracle's `catboost.utils.get_roc_curve` signature, CLI `--help` output for `roc` mode (`tools/oracle/cli` or pinned CLI binary), `R/catboost.R:3387` (`catboost.eval_metrics`).
* **Format:** Python introspection output, CLI help text, R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under `vendor/catboost/`. |
| **Logic** | One new R function, `catboost.get_roc_curve`, differential-tested against both oracles. |
| **CLI mode** | `mode:roc` row is satisfied by demonstrating the same numeric curve data via the CLI oracle for at least one fixture, per §4.3's CLI tolerance (`1e-9`, relative) — R computes natively, the CLI run is only the oracle comparison. |
| **Boundary** | Do not implement `catboost.utils.get_fnr_curve`, `catboost.utils.get_fpr_curve`, or `catboost.utils.select_threshold`, or any other `catboost.utils.*` row — none are named in spec line 638's gate text; a separate ticket must be filed if wanted. Do not implement `text_processing`/`embedding-processing`/`sample_gaussian_process` rows (unrelated capabilities matched by the same keyword search). |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Introspect Python's `get_roc_curve` signature/return shape via pinned oracle; quote exact output.
2. Get CLI `roc` mode `--help` output via pinned CLI oracle; quote exact output.
3. Implement `catboost.get_roc_curve` in `R/catboost.R`, computing from model predictions + Pool labels (reuse existing prediction/label accessors, grep first — no new native glue expected unless curve computation genuinely needs it).
4. Write differential tests in `tests/testthat/test_roc_curve.R`: numeric tolerance `1e-12` relative against the Python oracle (default per §4.3), plus one fixture compared against the CLI `roc` mode oracle at `1e-9` relative tolerance.
5. Update matrix disposition to `green` for both rows (`catboost.utils.get_roc_curve`, `mode:roc`).
6. Run `R CMD INSTALL --preclean .` then the new test file. Quote exact output.
7. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:

```toon
task_id: catboost-8z4.P4.6
success: bool
data:
  function_implemented: bool
  cli_roc_mode_verified: bool
  matrix_rows_green: int
  matrix_rows_total: 2
report_path: docs/phase-4/P4.6-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] `catboost.get_roc_curve` exists with passing differential tests against both oracles.
- [ ] CLI `roc` mode row verified against CLI oracle at declared tolerance.
- [ ] `git -C vendor/catboost status --porcelain` empty.
- [ ] Report written with exact command output.

---

## Task 7: P4.7: CLI eval-feature mode parity (catboost-8z4.56)

_Beads ticket: `bd show catboost-8z4.56`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._


**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add an R-reachable equivalent of the CLI's `eval-feature` mode (feature-elimination/feature-evaluation via repeated cross-validated training), verified by differential test against the pinned CLI oracle.
* **Why:** Phase 4 gate (spec line 638, "Includes CLI's eval-feature ... mode"). Matrix row `mode:eval-feature` — `method_mode_shaped`, `red`, oracle `cli`.
* **Reference Data:** No Python-side method wraps this mode identically — confirm this first by searching the Python oracle for an `eval_feature`/`select_features`-adjacent method (do not assume; `select_features` is explicitly Phase 5 scope per spec line 639, not this ticket) before concluding the CLI is the only oracle. CLI's `--help` output for `eval-feature` (via pinned CLI oracle) is the authoritative signature source. Per §4.3, CLI-only capabilities get relative `1e-9` tolerance, never bit-exact comparison, and no `--precision` flag exists on CLI text output — extraction path must go through the binary model or another mode if tighter comparison is ever needed (not this ticket's problem to solve, just don't violate the tolerance rule).
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** CLI `eval-feature --help` output (pinned CLI oracle), Python oracle search results for adjacent methods, existing `catboost.train`/`catboost.cv` R functions as the calling-convention precedent.
* **Format:** CLI help text, Python introspection output (negative result acceptable if none exists), R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under `vendor/catboost/`. |
| **Scope** | This ticket covers only `eval-feature`. Do not implement `select_features` (Phase 5) or `metadata`/`normalize-model` (Phase 5). |
| **Tolerance** | CLI-only capability: relative `1e-9`, never bit-exact; record the tolerance justification in the matrix per §4.3's governance rule (no ad hoc widening). |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Get CLI `eval-feature --help` output via pinned CLI oracle; quote exact output including every flag.
2. Search the Python oracle for any equivalent method; quote result (including if none found).
3. Design and implement an R function (name per §4.2's `verb_object` snake_case convention, e.g. `catboost.eval_feature`) wrapping the equivalent native training-with-feature-elimination logic, or, if the CLI mode has no in-process equivalent and truly requires the standalone binary's control flow, document that finding and the resulting design (do not silently skip — halt and report per epic's "any deviation halts and reports" guard if no in-process path exists).
4. Write a differential test in `tests/testthat/test_eval_feature.R` comparing R output against the CLI oracle's `eval-feature` output, relative `1e-9` tolerance, with justification recorded.
5. Update matrix disposition to `green` with test id.
6. Run `R CMD INSTALL --preclean .` then the new test file. Quote exact output.
7. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:

```toon
task_id: catboost-8z4.P4.7
success: bool
data:
  in_process_equivalent_found: bool
  function_implemented: bool
  differential_test_passing: bool
report_path: docs/phase-4/P4.7-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] CLI `eval-feature --help` output quoted in report.
- [ ] R equivalent implemented (or blocking design gap reported, not silently skipped).
- [ ] Differential test against CLI oracle passes at declared tolerance; matrix row flipped to green.
- [ ] `git -C vendor/catboost status --porcelain` empty.
- [ ] Report written with exact command output.

---

## Task 8: P4.8: CLI model-based-eval mode parity (catboost-8z4.57)

_Beads ticket: `bd show catboost-8z4.57`. Full body below is the ticket's hermetic 6-section spec, reproduced verbatim._


**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** Add an R-reachable equivalent of the CLI's `model-based-eval` mode, verified by differential test against the pinned CLI oracle.
* **Why:** Phase 4 gate (spec line 638, "Includes CLI's ... model-based-eval mode"). Matrix row `mode:model-based-eval` — `method_mode_shaped`, `red`, oracle `cli`.
* **Reference Data:** Search the Python oracle first for any adjacent method before assuming CLI-only (do not assume; confirm). §4.3's CLI tolerance rule applies: relative `1e-9`, never bit-exact, justification recorded, no ad hoc widening.
* **Philosophy:** Sub-agent = Goldfish Memory. All info here.

## II. Input Specification

* **Expected Input:** CLI `model-based-eval --help` output (pinned CLI oracle), Python oracle search results for adjacent methods.
* **Format:** CLI help text, Python introspection output (negative result acceptable), R source, testthat R test files.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under `vendor/catboost/`. |
| **Scope** | This ticket covers only `model-based-eval`. Do not implement `eval-feature` (P4.7) or any Phase 5 row. |
| **Tolerance** | CLI-only capability: relative `1e-9`, never bit-exact; record justification in the matrix. |
| **Isolation** | Dedicated git worktree; implementer commits its own work. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Get CLI `model-based-eval --help` output via pinned CLI oracle; quote exact output including every flag.
2. Search the Python oracle for any equivalent method; quote result (including if none found).
3. Design and implement an R function (name per §4.2's `verb_object` snake_case convention) wrapping the equivalent native logic, or document a blocking design gap if no in-process equivalent exists (halt and report per epic guard, do not silently skip).
4. Write a differential test in `tests/testthat/test_model_based_eval.R` comparing R output against the CLI oracle's output, relative `1e-9` tolerance, with justification recorded.
5. Update matrix disposition to `green` with test id.
6. Run `R CMD INSTALL --preclean .` then the new test file. Quote exact output.
7. Re-check Section III guards.

## V. Output Schema (Strict)

Sub-agent MUST return:

```toon
task_id: catboost-8z4.P4.8
success: bool
data:
  in_process_equivalent_found: bool
  function_implemented: bool
  differential_test_passing: bool
report_path: docs/phase-4/P4.8-report.md
vendor_clean: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] CLI `model-based-eval --help` output quoted in report.
- [ ] R equivalent implemented (or blocking design gap reported, not silently skipped).
- [ ] Differential test against CLI oracle passes at declared tolerance; matrix row flipped to green.
- [ ] `git -C vendor/catboost status --porcelain` empty.
- [ ] Report written with exact command output.

---

