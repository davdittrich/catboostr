# Phase 6: Parity debt cleanup — implementation plan (approved)

Epic: catboost-8z4. Base branch: phase-0 (591a099 merge + 6135bf6).

## Global Constraints

- **Mechanism:** direct-fix tickets (no new R public API surface, no new native entry points) — small, independent, mostly single-file fixes. subagent-driven-development if executed via fresh implementer subagents; otherwise controller-direct if user selects that execution mode.
- **Forbidden:** no bundling unrelated findings into one commit/ticket; no fabricated matrix rows (catboost-8z4.66 §4.4); no hand-curated whitelist entries (catboost-8z4.67, catboost-8z4.66 §4.5 discipline); no touching vendor/catboost/ (read-only). Note: catboost-8z4.68's full ~44-symbol .exports sync is NOT bundling — it is one mechanical action closing one class of drift (missing .exports registrations), analogous to the project's accepted family-level bulk-disposition pattern, not multiple distinct hand-curated findings.
- **Audit:** each task's own ticket Guards section is authoritative; task reviewer verifies PROTECT/UNPROTECT balance for catboost-8z4.71, byte-identical numeric output for catboost-8z4.65/.68/.70, no-fabrication + correct verification method for catboost-8z4.66 (Python-level checks, not a vacuous R rebuild — matrix.dispositioned.json is never read at R runtime, confirmed by grep), and full R-suite-green for catboost-8z4.67/.69/.70 (both genuinely touch R-affecting generated content or R source).
- Verification method is task-specific, stated explicitly in each task's own Step-by-Step Logic and Definition of Done — not silently inherited from a one-size-fits-all global line. catboost-8z4.65/.67/.68/.69/.70/.71 (all touch R and/or C++ source, or R-generated content) require `R CMD INSTALL --preclean .` clean + full testthat suite green (0 new FAIL/Error, same 2 pre-existing documented skips). catboost-8z4.66 (Python tooling + JSON fixture only, matrix never read at R runtime) instead requires tools/parity's own Python unit test suite green + JSON schema/row-count validation. vendor/catboost/ untouched in all 7 tasks.

## Execution Order

No cross-task file conflicts requiring bd dependency edges. catboost-8z4.68 and catboost-8z4.71 both concern CatBoostGetBinarizedStatistics_R but touch different files (.exports vs .cpp) with no line-level overlap. catboost-8z4.69 and catboost-8z4.70 share catboost.get_roc_curve and ARE sequenced adjacently since .69's added guard shifts .70's cited line numbers; each ticket's Step 1 mandates a fresh read before editing regardless. Suggested serial order:

1. catboost-8z4.65 — jsonlite params digits=10 truncation (R/catboost.R, catboost.train/cv)
2. catboost-8z4.67 — whitelist generator regex gap, CopyOptionWithNewKey (tools/parity/gen_param_reference.py, writes into R/catboost.R)
3. catboost-8z4.66 — Phase 2 inventory pipeline gap audit + fix (tools/parity/*.py, matrix regen; Python-level verification only)
4. catboost-8z4.68 — full .exports-vs-init.c mechanical sync (src/catboostr.exports), plus filing a follow-up for the confirmed opposite-direction dangling-registration gap (CatBoostPoolNumTrees_R)
5. catboost-8z4.69 — get_roc_curve label validation (R/catboost.R)
6. catboost-8z4.70 — get_roc_curve O(n^2) ponytail note (R/catboost.R, same function as .69)
7. catboost-8z4.71 — PROTECT-stack growth fix (src/catboostr.cpp)

## Task Tickets (verbatim)

### catboost-8z4.65

# Parity debt: params JSON precision truncation

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** `prepare_train_export_parameters` (R/catboost.R:2680 area) serializes the `params` list via `jsonlite::toJSON(..., digits = 10)` before passing to the native training call. Any hyperparameter value with more than 10 significant digits (e.g. a long-decimal `learning_rate`, `l2_leaf_reg`) reaches the native layer truncated, while Python passes full-precision doubles. Found as a named-risk check during P5.3's task review (catboost-8z4.60), not yet fixed — filed here as its own ticket per no-bundling rule.
* **Why:** Same defect class as the bug P5.2 (catboost-8z4.59) and P5.3 (catboost-8z4.60) independently hit and fixed in their own native JSON writer paths (`NJson::TJsonValue::ToString()`'s `DefaultDoubleNDigits = 10`) — but this instance is upstream, in R's own `jsonlite::toJSON` call, shared by every `catboost.train`/`catboost.cv` call site, not local to one new function. It is pre-existing (not introduced by any Phase 4/5 ticket) but will silently break a future differential test the moment a fixture uses a long-decimal parameter value at tight tolerance.
* **Reference Data:** `R/catboost.R`'s `prepare_train_export_parameters` (verify exact current line, has moved before). `jsonlite::toJSON` accepts `digits = I(...)` or `digits = NA` for full precision — evaluate whether raising/removing the digit cap changes any existing test's expected output (some other test may rely on the truncation coincidentally).

## II. Input Specification

* **Expected Input:** `R/catboost.R`'s `prepare_train_export_parameters`/`catboost.train`/`catboost.cv`, `jsonlite::toJSON` documentation.
* **Format:** R source.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under vendor/catboost/. |
| **Scope** | This serialization site only. Do not touch the native `NJson::TJsonValue::ToString()` writer-config sites already fixed in P5.2/P5.3 — those are done. |
| **Regression** | Full existing test suite must stay green after the precision change — a truncation-dependent test would be a pre-existing latent bug of its own, surface it if found. |
| **Tone** | Technical, terse. Confidence score (0-100) on every claim. |

## IV. Step-by-Step Logic

1. Locate `prepare_train_export_parameters` (or wherever the `jsonlite::toJSON` call on `params` currently lives); quote exact current code.
2. Change the digits argument to avoid truncation (e.g. `digits = NA` for full round-trip precision, or a value high enough that no representable double loses precision — R's `deparse`-level precision is ~17 significant digits).
3. Add a regression test: a hyperparameter value with >10 significant digits round-trips through `catboost.train`'s params without truncation (compare against what reaches the native layer, or against a Python-oracle run with the same high-precision value).
4. Run full test suite; confirm no existing test depended on the truncation.
5. Run `R CMD INSTALL --preclean .` (redirect to log, check $? directly) then the new test.

## V. Output Schema (Strict)

```toon
task_id: catboost-jpp2
success: bool
data:
  truncation_fixed: bool
  regression_test_added: bool
  existing_suite_still_green: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] `jsonlite::toJSON` call for params no longer truncates to 10 significant digits.
- [ ] Regression test proves a high-precision hyperparameter round-trips intact.
- [ ] `R CMD INSTALL --preclean .` succeeds.
- [ ] Full suite green, no test relied on the old truncation.

### catboost-8z4.67

# Parity debt: whitelist generator regex gap on CopyOptionWithNewKey

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** `tools/parity/gen_param_reference.py`'s `parse_native_copyoption_names()` (added in Phase 5's whole-branch fix wave, commit a5272a0) extracts accepted native parameter names via the regex `CopyOption\(plainOptions,\s*"([^"]+)"`. This misses 8 names reachable only via `CopyOptionWithNewKey(plainOptions, "name", ...)`: `od_pval`, `od_wait`, `od_type`, `bootstrap_type`, `ctr_target_border_count`, `feature_border_type`, `device_config`, `pinned_memory_size` (found in `vendor/catboost/catboost/private/libs/options/plain_options_helper.cpp`). Found during the Phase 5 whole-branch re-review, not blocking (all 8 already present in the Python-surface inventory today, so no current gap), but a future vendor pin bump could add a `CopyOptionWithNewKey`-only name that isn't Python-surfaced, silently reproducing the exact backward-compat regression the fix wave just closed.
* **Why:** The whitelist generator's whole purpose is to derive-not-hand-curate the accepted `params` key set (per §4.5's "machine-generated, never hand-curated" discipline). A regex with a known false-negative class defeats that purpose quietly.
* **Reference Data:** `vendor/catboost/catboost/private/libs/options/plain_options_helper.cpp` — both `CopyOption(plainOptions, "name", ...)` and `CopyOptionWithNewKey(plainOptions, "name", ...)` calls exist; the fix is widening the regex, e.g. `\w*CopyOption\w*\(plainOptions,\s*"([^"]+)"` or two explicit patterns.

## II. Input Specification

* **Expected Input:** `tools/parity/gen_param_reference.py`, `vendor/catboost/catboost/private/libs/options/plain_options_helper.cpp`.
* **Format:** Python source, C++ source.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **Read-only pin** | Zero writes under vendor/catboost/. |
| **Scope** | Widen the extraction regex/logic only. Do not change the whitelist's semantics otherwise. |
| **Regression** | Regenerating `.catboostr_known_params` after the fix must be a strict superset of (or identical to) the current set — confirm via symmetric diff before/after, since the 8 currently-covered-by-inventory names should not disappear or change. |

## IV. Step-by-Step Logic

1. Confirm the 8 named `CopyOptionWithNewKey` sites still exist at their claimed locations (vendor pin may have moved since Phase 5).
2. Widen `parse_native_copyoption_names()` to also match `CopyOptionWithNewKey(plainOptions, "name", ...)` calls.
3. Search for any other option-acceptance helper function beyond these two (grep `plain_options_helper.cpp` for other `plainOptions.Has`/assignment patterns) to confirm no third pattern is being missed.
4. Regenerate `.catboostr_known_params`; diff against the current committed set — confirm it's a superset (ideally identical, since all 8 known names are already covered via the Python inventory today).
5. Add a test or assertion (in the generator itself, or a differential test) that would catch a future name appearing only via `CopyOptionWithNewKey` and not the Python inventory.
6. Regenerating `.catboostr_known_params` rewrites a generated block inside `R/catboost.R` (the `.catboostr_known_params` set and its accompanying `\describe` doc block) — since this is R-source-affecting, run `R CMD INSTALL --preclean .` (env `CATBOOSTR_VENDOR_SRC`, `CATBOOSTR_THIRDPARTY_SRC`; redirect log, check `$?` and log tail directly) and the full `testthat` suite; confirm green (0 new FAIL/Error, same pre-existing skip count), with particular attention to `test_params_validation.R` and `test_param_family_coverage.R`.

## V. Output Schema (Strict)

```toon
task_id: catboost-jpp4
success: bool
data:
  regex_widened: bool
  new_names_found: int
  whitelist_regenerated: bool
  install_clean: bool
  suite_green: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] Extraction logic covers both `CopyOption` and `CopyOptionWithNewKey`.
- [ ] Regenerated whitelist confirmed superset/identical to current, no regression.
- [ ] A test or check exists that would catch this class of gap on a future vendor pin bump.
- [ ] Step 3's third-pattern audit result is explicitly reported (found/not found, with evidence), not merely performed and left unreported.
- [ ] `R CMD INSTALL --preclean .` succeeds and the full test suite is green (0 new FAIL/Error, same pre-existing skip count).

### catboost-8z4.66

# Phase 2 inventory gap: virtual_ensembles_predict missing from matrix

**Status:** READY_FOR_EXECUTION

## I. Context & Objective

* **Objective:** `catboost.virtual_ensembles_predict` is matched in `capability_diff.json` (4 owner-class entries: CatBoost/CatBoostClassifier/CatBoostRegressor/CatBoostRanker) and `r_surface.json` lists `catboost.virtual_ensembles_predict` as an exported R symbol — but zero rows for it exist in `tests/fixtures/parity/matrix.dispositioned.json`. Found and confirmed during P5.4's task review (catboost-8z4.61): the capability was matched by Phase 2's matcher but never reached the final dispositioned matrix. Audit whether this is an isolated miss or a symptom of a broader Phase 2 pipeline gap, and fix the pipeline (not just this one row).
* **Why:** Per spec §4.5, the parity matrix is the "only complete input list" and "no capability may be silently absent." A capability that's matched but drops out before disposition defeats that guarantee silently. This is the 3rd bookkeeping gap surfaced during Phase 5 execution (after two matrix-row-not-flipped misses in P5.2/P5.3's own review passes) — worth determining if there's a systemic cause (e.g. a specific matcher stage or diposition-rule category that drops rows) rather than treating each as an isolated one-off.
* **Reference Data:** `tools/parity/` pipeline scripts (`compute_diff.py`, the disposition/bulk-rule application script from Phase 2, catboost-8z4.34-36). `capability_diff.json`, `r_surface.json`, `tests/fixtures/parity/matrix.dispositioned.json` — compare the full capability_diff.json against the final matrix to find every other capability with the same drop-out pattern (matched in diff, absent from matrix), not just this one.

## II. Input Specification

* **Expected Input:** `tools/parity/*.py`, `capability_diff.json`, `r_surface.json`, `tests/fixtures/parity/matrix.dispositioned.json`.
* **Format:** Python source, JSON.

## III. Constraints & Guards

| Type | Guard |
| :--- | :--- |
| **No fabrication** | Do not hand-add matrix rows. Any row added must come from re-running (or correctly fixing and re-running) the actual machine-generated pipeline, per §4.5's "machine-generated, never hand-curated" rule. |
| **Scope** | Audit + pipeline fix + regenerate the matrix. Do not use this ticket to also close/disposition the resulting new rows with differential tests — that's separate follow-up work once the rows exist (link/create tasks for whichever ones surface, don't bundle). |
| **Regression** | Re-running the pipeline must not silently change or drop any of the 725 already-dispositioned rows' existing state — diff old vs regenerated matrix and account for every change. |

## IV. Step-by-Step Logic

1. Diff `capability_diff.json` (or its current equivalent, re-verify it still exists / is current) against `matrix.dispositioned.json`'s `inventory_row_id`/`members` to find every capability present in the diff but absent from the matrix. Quote the full list — do not assume `virtual_ensembles_predict` is the only one.
2. Read `tools/parity/compute_diff.py` and whatever disposition-application script runs after it; trace exactly which stage drops rows, and why (e.g. a matching-key mismatch, a silent exception, a filter that's too aggressive).
3. Fix the root cause in the pipeline script(s).
4. Re-run the full pipeline; diff the regenerated matrix against the current one; confirm all 725 previously-dispositioned rows are unchanged in state/test_id, and confirm every gap found in step 1 now has a row.
5. Verify the fix with checks that actually exercise the changed code (a full `R CMD INSTALL --preclean .` rebuild is NOT required and is not a meaningful check here — this ticket touches only `tools/parity/*.py` and a JSON fixture; `tests/fixtures/parity/matrix.dispositioned.json` is referenced only in R/testthat source comments, never read at runtime by `R/catboost.R` or by `testthat`, confirmed by grep before relying on any R-level check):
   - Run `tools/parity`'s own Python unit test suite (`python3 -m unittest discover tools/parity` or equivalent covering `test_apply_disposition.py`, `test_build_matrix.py`), confirm it passes both before and after the pipeline fix.
   - Validate the regenerated `matrix.dispositioned.json` is well-formed JSON with the expected row count (725 pre-existing + N newly-surfaced, N from step 1's list) and that every row has the schema's required fields populated (no `null`/missing `inventory_row_id`, `state`, etc.).
6. Commit the regenerated matrix and the pipeline fix together.
7. For each newly-surfaced row, file a separate follow-up ticket (or one combined ticket if they're clearly one family) to close it with a differential test — do not close them in this ticket.

## V. Output Schema (Strict)

```toon
task_id: catboost-jpp3
success: bool
data:
  other_gaps_found: int
  root_cause: string
  pipeline_fixed: bool
  matrix_regenerated: bool
  existing_725_rows_unchanged: bool
  python_tests_green: bool
  matrix_schema_valid: bool
error_log: null | msg
```

## VI. Definition of Done

- [ ] Full list of capability_diff-matched-but-matrix-absent rows quoted (not just virtual_ensembles_predict).
- [ ] Root cause identified in the pipeline script, with file:line citation.
- [ ] Pipeline fixed and re-run; regenerated matrix diffed against current, all 725 existing rows' state/test_id unchanged.
- [ ] Newly-surfaced rows have follow-up tickets filed (not closed in this ticket).
- [ ] `tools/parity`'s own Python unit test suite passes; regenerated matrix confirmed well-formed JSON with expected row count and no missing required fields. (No R rebuild/testthat run required — this ticket touches no R/C++ source and the matrix is not read at R runtime.)

### catboost-8z4.68

# catboost-jppN: Register all catboostr entry points in src/catboostr.exports
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** `src/catboostr.exports` (used by `R CMD SHLIB`/`useDynLib`'s exports mechanism check, one symbol per line prefixed `C `) is missing entries for native entry points that exist in `src/init.c`'s `R_CallMethodDef` table. The finding was originally scoped (in the Phase 4 whole-branch review that discovered it) to 5 Phase-4-analysis-parity symbols (`CatBoostEvaluateFeatures_R`, `CatBoostGetBinarizedStatistics_R`, `CatBoostGetFeatureTypeAndInternalIndex_R`, `CatBoostCalcCatFeaturePerfectHash_R`, `CatBoostGetCatFeatureValues_R`) — **but a direct diff at plan-review time found the true gap is far larger, ~44 symbols**, spanning multiple older, unrelated subsystems (Pool label/quantize accessors, text dictionary/tokenizer entries, dataset statistics, etc.), not just Phase 4's. This ticket's scope is corrected accordingly: it is one mechanical, single-action bulk sync of `.exports` against `init.c`'s authoritative table — add every symbol present in `init.c` but absent from `.exports`, regardless of which historical branch introduced it. This is not new scope creep relative to a "5-symbol Phase-4 fix" — it is what the ticket's own Step 3 (below) already required ("do not assume it is exactly these 5") and is analogous to this project's accepted family-level bulk-disposition pattern (§4.5): one generated, mechanical action covering an entire class of drift, not N separate hand-curated tickets for each missing symbol.
* **Why:** Found in the Phase 4 whole-branch review (opus, base `992acd6..1c08c21`), filed as part of bundled follow-up ticket `catboost-jpp`; split out per the no-bundling rule; rescoped from "5 symbols" to "full mechanical sync" during this plan's review gate (Feasibility reviewer, iteration 2) after a direct diff showed the true gap. Currently harmless — R dispatch for `.Call()` goes through `src/init.c`'s `R_CallMethodDef` table, not this file — but the file's purpose is a complete registration list and it has been silently incomplete for a long time, not just since Phase 4.
* **Reference Data:** `src/catboostr.exports` (current full list, verify by reading — do not assume any prior read cached in this ticket's authoring context is still current). `src/init.c`'s `R_CallMethodDef` table is the authoritative source of every native entry point that must appear. Also noted at plan-review time: `CatBoostPoolNumTrees_R` appears in `.exports` (`src/catboostr.exports:24`) and is implemented in `src/catboostr.cpp:587`, but is **absent from `init.c`'s `R_CallMethodDef` table** — a distinct dangling-registration bug in the opposite direction from this ticket's scope (an `.exports` entry with no `init.c` registration, not a missing `.exports` entry). Do not fix it in this ticket (this ticket only adds missing lines per its Guard, never removes or reconciles the other direction) — surface it as a separate follow-up per Step 6 below.

## II. Input Specification
* **Expected Input:** `src/catboostr.exports`, `src/init.c`.
* **Format:** Plain text, one `C <SymbolName>` per line.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Diff `init.c`'s `R_CallMethodDef` symbol list against `catboostr.exports`'s `C <name>` lines. Add every symbol present in `init.c` but absent from `.exports`. Do not remove any existing line. |
| **Scope** | Only `src/catboostr.exports`. Do not touch `init.c`, `catboostr.cpp`, or any R source. |
| **Format** | Preserve the file's existing sort order and `C ` prefix convention exactly; new entries inserted in alphabetical position, not appended at the end. |
| **Boundary** | Do not add symbols that are not `.Call`-registered entry points (e.g. do not add `R_init_libcatboostr` if it is not already following the `C ` convention used by the rest of the file — verify by reading the existing line for it first). |

## IV. Step-by-Step Logic
1. Read `src/init.c`, extract every symbol name registered in its `R_CallMethodDef` table.
2. Read `src/catboostr.exports`, extract every symbol name currently listed.
3. Compute the set difference (in init.c, not in .exports). Confirm it includes at least the 5 named above; quote the full diff, do not assume it is exactly these 5 (more may have been added by Phase 5 tickets and missed the same way).
4. Separately (informational only, do not act on it beyond filing), check whether any symbol present in `.exports` is absent from `init.c`'s table (the opposite-direction check) — confirm whether `CatBoostPoolNumTrees_R` is such a case, and whether it's the only one. File one follow-up ticket for whatever is found (even if it is exactly this one symbol) — do not fix it here.
5. Add each missing symbol (found in step 3) as a new `C <SymbolName>` line, in alphabetical position.
6. Run `R CMD INSTALL --preclean .` (env `CATBOOSTR_VENDOR_SRC`, `CATBOOSTR_THIRDPARTY_SRC` set per project convention; redirect log to a file, check `$?` and log tail directly, do not pipe through `tail -f` blocking).
7. Run the full `testthat` suite; confirm no regressions (0 new FAIL/Error, same pre-existing skip count as before this change).

## V. Output Schema (Strict)
```toon
task_id: catboost-jppN
success: bool
data:
  symbols_added: int
  symbol_names: [string]
  opposite_direction_gap_found: bool
  follow_up_ticket_filed: string | null
  install_clean: bool
  suite_green: bool
error_log: null | msg
```

## VI. Definition of Done
- [ ] Full init.c-vs-.exports diff quoted, not assumed to be exactly the 5 originally-named symbols (expect ~44, confirm actual count at execution time).
- [ ] Every symbol present in init.c's `R_CallMethodDef` table is present in `.exports`.
- [ ] Opposite-direction gap (`.exports` entries absent from `init.c`, e.g. `CatBoostPoolNumTrees_R`) checked and filed as a separate follow-up ticket, not fixed in this ticket.
- [ ] `R CMD INSTALL --preclean .` succeeds (`* DONE (catboostr)`, no errors).
- [ ] Full test suite green, no new failures, same pre-existing skip count.

### catboost-8z4.69

# catboost-jppN: catboost.get_roc_curve must reject non-{0,1} labels instead of silently miscounting
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** `catboost.get_roc_curve` (`R/catboost.R:4726-4804`) rounds pool labels to integers (`R/catboost.R:4746`, `as.integer(label + 0.5)`) then classifies each observation into class 0 or class 1 by `if (target[i] == 1L) countTarget1++ else countTarget0++` (`R/catboost.R:4793`). A label rounding to any value other than `0L`/`1L` (e.g. `2`) is silently counted as class 0 in that accumulator, but `count0 <- sum(target == 0L)` (`R/catboost.R:4750`) excludes it — so FPR (`newFpr <- countTarget0 / count0`, line 4797) can exceed 1. Add an explicit guard: `stop()` if any rounded label is outside `{0L, 1L}`.
* **Why:** Found in the Phase 4 whole-branch review (opus, base `992acd6..1c08c21`), filed as part of bundled follow-up ticket `catboost-jpp`; split out per the no-bundling rule. The vendor CLI's `TRocCurve::BuildCurve` would crash (out-of-range index) on the same input rather than silently producing a curve with FPR > 1 — this R implementation currently diverges from that fail-fast behavior.
* **Reference Data:** `R/catboost.R:4726-4804` (`catboost.get_roc_curve`, read fresh — do not assume line numbers are unchanged from this ticket's authoring). ROC curve is binary-classification only per both the Python and CLI oracles (§4.2 of the design spec) — a non-binary label set is a genuine user error, not a case to silently handle.

## II. Input Specification
* **Expected Input:** `R/catboost.R`'s `catboost.get_roc_curve` function body.
* **Format:** R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Logic** | Guard must check the rounded `target` vector (post `as.integer(label + 0.5)`, line ~4746), not the raw label — a label of `0.6` rounds to `1L` and is valid; only round-to-non-{0,1} is the error. |
| **Placement** | Add the check immediately after `target` is fully accumulated across all pools (after the `for (p in pools)` loop, before `count1`/`count0` are computed), so it covers the multi-pool case too, not just a single pool. |
| **Message** | Error message must name the actual offending value(s) or at least state clearly that labels must be exactly 0/1 after rounding — do not use a generic "invalid input" message. |
| **Scope** | Only this function. Do not touch other `catboost.get_*` functions even if they share a similar pattern — audit them in a separate ticket if a similar gap is found (do not silently expand scope). |
| **No fabrication** | Do not weaken or remove the existing `count0 == 0 || count1 == 0` guard (line 4751-4752) — this is a new, additional guard, not a replacement. |

## IV. Step-by-Step Logic
1. Read `R/catboost.R`'s current `catboost.get_roc_curve`, confirm line numbers for the `target` accumulation and the `count1`/`count0` guard (they may have shifted since this ticket was authored).
2. Add `if (any(!target %in% c(0L, 1L))) stop(...)` (or equivalent) immediately after the pools loop, before `count1 <- sum(target == 1L)`.
3. Write a regression test: a pool with labels including a value that rounds to something other than `0`/`1` (e.g. label `2`) passed to `catboost.get_roc_curve` must error, not silently return a curve.
4. Run the full `testthat` suite; confirm no regression against existing ROC-curve tests (they presumably all use well-formed binary labels and must still pass unchanged).
5. Run `R CMD INSTALL --preclean .` (redirect log, check `$?` and log tail directly).

## V. Output Schema (Strict)
```toon
task_id: catboost-jppN
success: bool
data:
  guard_added: bool
  regression_test_added: bool
  existing_roc_tests_still_pass: bool
error_log: null | msg
```

## VI. Definition of Done
- [ ] `catboost.get_roc_curve` errors on any rounded label outside `{0L, 1L}`, across single- and multi-pool input.
- [ ] Regression test proves the new guard triggers and reports a clear message.
- [ ] All pre-existing ROC-curve tests still pass unchanged.
- [ ] `R CMD INSTALL --preclean .` succeeds, full suite green.

### catboost-8z4.70

# catboost-jppN: Mark catboost.get_roc_curve's O(n^2) point-accumulation with a ponytail ceiling note
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** `catboost.get_roc_curve`'s inner closure `add_point` (`R/catboost.R:4763-4787`) grows the `boundary`/`fnr`/`fpr` vectors one element at a time via `<<-` (lines 4782, 4786). R vector growth-by-append reallocates and copies on every call, making the whole function O(n^2) in the number of distinct probability boundaries, versus the vendor's single O(n) C++ sweep (`TRocCurve::BuildCurve`). This is a performance ceiling, not a correctness bug — add a `# ponytail:` comment naming the ceiling and the upgrade path (pre-allocate to `n` and truncate, since the number of `add_point` calls is bounded by the number of observations).
* **Why:** Found in the Phase 4 whole-branch review (opus, base `992acd6..1c08c21`), filed as part of bundled follow-up ticket `catboost-jpp`; split out per the no-bundling rule. Fine at the project's current fixture/test scale; would be painful at ~10^6-row pools. Per the project's ponytail convention, a deliberate simplification with a known ceiling gets a marker comment naming the ceiling and upgrade path, not a mandatory rewrite in this ticket.
* **Reference Data:** `R/catboost.R:4763-4787` (the `add_point` closure and its two call-site patterns, read fresh — line numbers may have shifted). Project convention: `# ponytail: <ceiling>, <upgrade path if Y>` per the ponytail skill's documented comment format.

## II. Input Specification
* **Expected Input:** `R/catboost.R`'s `catboost.get_roc_curve` function body, specifically the `add_point` closure.
* **Format:** R source.

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Scope** | Comment-only change. Do NOT rewrite `add_point` to pre-allocate or vectorize in this ticket — that is the noted upgrade path for a future ticket if profiling shows it's load-bearing, not this ticket's deliverable. |
| **Logic** | The comment must correctly name the actual ceiling (O(n^2) vector growth via repeated `<<-` append) and a concrete upgrade path (e.g. pre-allocate `numeric(n)` sized vectors and truncate at the end, since the number of `add_point` invocations is bounded by `n`), not a vague "could be faster" note. |
| **No behavior change** | Diff must be comment-only; the function's numeric output must be byte-identical before and after (verify via the existing ROC-curve test(s) passing unchanged). |

## IV. Step-by-Step Logic
1. Read `R/catboost.R`'s current `catboost.get_roc_curve` and `add_point`, confirm line numbers.
2. Add a `# ponytail: O(n^2) vector growth via <<- append per boundary point; pre-allocate numeric(n) for boundary/fnr/fpr and truncate if profiling shows this is load-bearing at scale` comment immediately above the `add_point <- function(...)` definition.
3. Run the existing ROC-curve test(s) to confirm zero behavior change (comment-only diff).
4. Run `R CMD INSTALL --preclean .` (env `CATBOOSTR_VENDOR_SRC`, `CATBOOSTR_THIRDPARTY_SRC`; redirect log, check `$?` and log tail directly) and the full `testthat` suite; confirm green (0 new FAIL/Error, same pre-existing skip count).

## V. Output Schema (Strict)
```toon
task_id: catboost-jppN
success: bool
data:
  comment_added: bool
  behavior_unchanged: bool
  install_clean: bool
  suite_green: bool
error_log: null | msg
```

## VI. Definition of Done
- [ ] `# ponytail:` comment present above `add_point`, naming the O(n^2) ceiling and a concrete upgrade path.
- [ ] Diff is comment-only; no logic change.
- [ ] Existing ROC-curve test(s) pass unchanged, confirming byte-identical output.
- [ ] `R CMD INSTALL --preclean .` succeeds and the full test suite is green (0 new FAIL/Error, same pre-existing skip count).

### catboost-8z4.71

# catboost-jppN: Fix unbounded R PROTECT-stack growth in CatBoostGetBinarizedStatistics_R
**Status:** `READY_FOR_EXECUTION`

## I. Context & Objective
* **Objective:** `CatBoostGetBinarizedStatistics_R` (`src/catboostr.cpp:2720-2829`) PROTECTs 9 SEXPs per feature inside its `for (size_t s = 0; s < statistics.size(); ++s)` loop (`src/catboostr.cpp:2765` `statList`, `2767` `statNames`, `2770` `borders`, `2777` `binarizedFeature`, `2784` `meanTarget`, `2791` `meanWeightedTarget`, `2798` `meanPrediction`, `2805` `objectsPerBin`, `2812` `predictionsOnVaryingFeature` — verify exact lines by reading fresh, they are approximate) but never calls `UNPROTECT` inside the loop — the single `UNPROTECT(protectedCount)` (`src/catboostr.cpp:2827`) fires once, after the loop, so `protectedCount` accumulates `9 * statistics.size()` entries on R's PROTECT stack even though each `statList` becomes reachable (and therefore already protected transitively) the moment it is placed via `SET_VECTOR_ELT(result, s, statList)`. Add a per-iteration `UNPROTECT` for the 9 per-feature SEXPs once they're no longer needed directly (i.e. once `statList` itself is attached to `result`), leaving only `result` (and, if applicable, `statList`/`statNames` for exactly as long as needed to build them) protected across iterations.
* **Why:** Found in the Phase 4 whole-branch review (opus, base `992acd6..1c08c21`), filed as part of bundled follow-up ticket `catboost-jpp`; split out per the no-bundling rule. `catboost.calc_feature_statistics(feature = NULL)` on a pool with more than ~1100 features (R's PROTECT stack is 10000 deep by default) would exhaust the stack and crash/error inside R's C API rather than the vendor's C++ layer, which has no such structural limit.
* **Reference Data:** `src/catboostr.cpp:2720-2829` (`CatBoostGetBinarizedStatistics_R`, read fresh for exact current line numbers — the compressed read used during ticket authoring garbled some lines; treat the line numbers here as approximate, not authoritative). R's PROTECT/UNPROTECT discipline: a SEXP made reachable from an already-PROTECTed parent (via `SET_VECTOR_ELT` onto a PROTECTed list) does not itself need to stay PROTECTed once attached — this is the standard "protect a child only until it's linked into a protected parent" pattern already used correctly elsewhere in this file (compare e.g. `CatBoostGetModelInfo_R` or another entry point that builds a list of variable-length children, for a pattern to follow).

## II. Input Specification
* **Expected Input:** `src/catboostr.cpp`'s `CatBoostGetBinarizedStatistics_R` function body.
* **Format:** C++ source (R C API, `PROTECT`/`UNPROTECT`/`SET_VECTOR_ELT`).

## III. Constraints & Guards
| Type | Guard |
| :--- | :--- |
| **Correctness** | Every SEXP must remain protected (directly or transitively via an already-protected parent) at every point a GC could run (any subsequent `allocVector` call) between its creation and its being linked into the final `result`. Do not under-protect — verify by reading the R Extensions manual's PROTECT discipline if uncertain, do not guess. |
| **No fabrication** | Do not simply wrap the whole function body differently to "hide" the count without actually bounding per-iteration growth — the fix must make peak PROTECT-stack depth O(1) per iteration (bounded, not `O(features)`), not just cosmetically reorder. |
| **Scope** | Only `CatBoostGetBinarizedStatistics_R`. Do not touch other `.cpp` entry points even if they share a similar per-iteration PROTECT pattern — audit separately if found, do not silently expand scope. |
| **Regression** | `catFeaturesNums`/`floatFeaturesNums` handling, `EPredictionType` parsing, and the numeric contents of every field (`borders`, `binarized_feature`, `mean_target`, `mean_weighted_target`, `mean_prediction`, `objects_per_bin`, `predictions_on_varying_feature`) must be byte-identical before and after — this is a memory-management-only fix. |

## IV. Step-by-Step Logic
1. Read `src/catboostr.cpp`'s current `CatBoostGetBinarizedStatistics_R` in full (native `Read`, not a compressed context tool, to get exact unambiguous line content) and confirm current line numbers and the exact per-iteration PROTECT sequence.
2. Restructure the loop body so each of the 9 per-feature SEXPs is `UNPROTECT`ed as soon as it is no longer needed unprotected directly (immediately after being linked into `statList` via `SET_VECTOR_ELT`, and once `statList` itself is linked into `result`), so peak protect depth no longer grows with `statistics.size()`.
3. Adjust `protectedCount`/explicit `UNPROTECT` call counts to match the new discipline exactly — an UNPROTECT count that doesn't match the PROTECT count corrupts R's protection stack and must not happen.
4. Write or extend a test exercising `catboost.calc_feature_statistics` with a pool that has enough features to make the old code's peak depth clearly bounded-vs-unbounded observable in principle (a full ~1100+-feature reproduction is likely impractical for a fast unit test — instead, assert correctness of the statistics output for a small multi-feature pool, since the memory-safety property itself is not something a unit test can directly measure without an artificially large fixture; note in the report whether a large-N synthetic test was feasible or whether output-correctness plus code-review of the protect/unprotect balance is the achievable verification).
5. Run the full `testthat` suite; confirm `catboost.calc_feature_statistics`'s existing tests pass unchanged (same numeric output).
6. Run `R CMD INSTALL --preclean .` (redirect log, check `$?` and log tail directly).

## V. Output Schema (Strict)
```toon
task_id: catboost-jppN
success: bool
data:
  protect_stack_bounded: bool
  protect_unprotect_balanced: bool
  existing_tests_still_pass: bool
  large_n_test_feasible: bool
error_log: null | msg
```

## VI. Definition of Done
- [ ] Per-iteration PROTECT count no longer accumulates unboundedly across `statistics.size()` iterations — peak protect depth is O(1) per iteration, not O(features).
- [ ] PROTECT/UNPROTECT calls are exactly balanced (verified by count, not just "compiles").
- [ ] All existing `catboost.calc_feature_statistics` tests pass with byte-identical numeric output.
- [ ] `R CMD INSTALL --preclean .` succeeds, full suite green.

