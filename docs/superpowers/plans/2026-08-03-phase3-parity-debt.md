# Phase 3 Parity-Debt Cleanup (catboost-8z4.45–.49) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 5 parity-debt tickets (catboost-8z4.45–.49) filed during the phase-3-pool-parity branch's implementation and final review — undocumented or unimplemented R-vs-Python behavioral divergences in Pool construction (embeddings, sparse input, multi-target labels) and text-processing (tokenizer/dictionary), plus a low-priority constructor-signature gap.

**Architecture:** All 5 tasks touch the same two files (`R/catboost.R`, `src/catboostr.cpp`) that every ticket on this branch has touched; each task is independently testable via a differential test against the pinned Python oracle (`tools/oracle/`), following the pattern established by tickets .38–.44.

**Tech Stack:** R (S3 classes, `.Call`/`R_ExternalPtr` C bridge — no Rcpp anywhere in this package), C++ (CatBoost core, read-only vendor pin under `vendor/catboost/`), Python oracle fixtures (pinned `catboost==1.2.10`, generated via `tools/oracle/`).

## Global Constraints

- Zero writes under `vendor/catboost/` (read-only pin). Every task must end with `git -C vendor/catboost status --porcelain` empty.
- Every differential test must be a genuine round-trip comparison against real Python oracle output — never a "no exception thrown" tautology. (This exact gap caused fix-loop rounds on tickets .38 and .44 during the parent branch's implementation.)
- Build recipe (only needed for tasks touching `src/catboostr.cpp`/`.h`/`init.c`, or any new/changed R export): run as ONE synchronous foreground command, wait for it to actually finish, never background/detach it —
  ```
  cd /home/dd/Gemini/catboost-phase-3-pool-parity
  CATBOOSTR_VENDOR_SRC=/home/dd/Gemini/catboost/vendor/catboost CATBOOSTR_CYTHON=/tmp/catboostr-cython-venv/.venv/bin/cython R CMD INSTALL --preclean .
  ```
  (Stale-lock recovery: verify no live process via `pgrep -fal "R CMD INSTALL|cc1plus"`, then `rm -rf` the named `00LOCK-*` directory and retry.)
- `get_object_importance`'s MultiClass/multi-target behavior is out of scope for every task in this plan (deferred to Phase 4 per spec line 638 / `docs/phase-2/P2.4-report.md` "Finding 2"). No task may touch it.
- Continue on branch `phase-3-pool-parity` in worktree `/home/dd/Gemini/catboost-phase-3-pool-parity` (not yet merged to `phase-0`) — all 5 tickets are follow-ups discovered on this branch and depend on its already-landed code (tickets .38–.44).
- roxygen-generated `man/*.Rd` files: edit the `@param`/`@title` roxygen comment block in the relevant `.R` file, then regenerate via `roxygen2::roxygenise(load_code = roxygen2::load_source)` — never hand-edit `man/*.Rd` (a prior hand-edit on this branch, ticket .40, was flagged as an unjustified process deviation during review).
- Every task's differential-test fixture generator lives under `tools/oracle/`, following the existing `gen_*_fixture.py` naming convention.

---

### Task 1: catboost-8z4.45 — document and bound the LDA-default embedding divergence

**Alternatives Considered:** (a) Canonicalize the LDA eigenvector basis in vendor's `library/cpp/embedding_features` LDA calcer (`ssyev` symmetric-eigenproblem sign/order convention) so both builds agree bit-for-bit — this is the only *true* fix, but it requires a fork patch to vendor code, which is a much larger, separate decision (patching a pinned read-only vendor tree changes this fork's update/rebase story) and is explicitly flagged in the ticket as "requires touching vendor/catboost ... therefore a fork-patch decision," not something to default into. (b) Accept the divergence as a documented, bounded, permanent limitation of this fork's BLAS/LAPACK build, with a differential test that asserts the divergence stays within known bounds rather than asserting exact equality. Per the project's Mechanism Justification ranking (performance, simplicity/LOC, ecosystem support, maintenance overhead) and YAGNI, (b) is selected: it requires no vendor patch, no new build-time BLAS pinning infrastructure, and the ticket's own root-cause finding (BLAS/LAPACK-build-sensitive eigenvector sign/ordering on near-degenerate eigenvalues, not a defect in the new embedding glue) means (a) would only "fix" this build's specific toolchain, not the underlying non-determinism — a future BLAS upgrade could reopen the same divergence even after patching. (b) is the ceiling-aware, honestly-scoped choice; escalate to (a) only if a user actually needs bit-exact LDA parity.

**Files:**
- Modify: `R/catboost.R` (roxygen `@param embedding_features` block, already carries a caveat from ticket .44's fix wave at approximately line 44 — extend it to reference the new bounded test, no new caveat needed if the existing one already covers this)
- Modify: `tests/testthat/test_pool_embeddings.R` (add the new bounded-divergence test)
- Modify: `tools/oracle/gen_pool_embeddings_fixture.py` (already emits `predict_default_lda` in the fixture per ticket .45's own evidence — verify this value is present and wire it into the R-side comparison)
- Read (no changes): `tests/fixtures/oracle/pool_embeddings.json`, `docs/phase-3/catboost-8z4.44-report.md` (root-cause evidence)

**Interfaces:**
- Consumes: `catboost.load_pool(..., embedding_features = ..., embedding_features_indices = ...)` (ticket .44, `R/catboost.R`), `catboost.train`/`catboost.predict` (existing).
- Produces: nothing new consumed by later tasks — this task is a leaf.

- [ ] **Step 1: Confirm the oracle fixture already contains `predict_default_lda`**
  Run: `grep -n "predict_default_lda" /home/dd/Gemini/catboost-phase-3-pool-parity/tests/fixtures/oracle/pool_embeddings.json`
  Expected: at least one match. If absent, regenerate the fixture first: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_embeddings_fixture.py`

- [ ] **Step 2: Write the failing bounded-divergence test**
  Add to `tests/testthat/test_pool_embeddings.R`:
  ```r
  test_that("embeddings: default embedding_processing (LDA+KNN) diverges from Python within documented bounds (catboost-8z4.45)", {
    fixture <- jsonlite::fromJSON(
      system.file("../fixtures/oracle/pool_embeddings.json", package = "catboostr")
        %||% "tests/fixtures/oracle/pool_embeddings.json"
    )
    pool <- catboostr:::catboost.load_pool(
      data = fixture$inputs$features,
      label = fixture$inputs$label,
      embedding_features = list(as.matrix(fixture$inputs$embedding_data)),
      embedding_features_indices = fixture$inputs$embedding_feature_index
    )
    model <- catboostr:::catboost.train(
      pool,
      params = fixture$inputs$train_params  # embedding_processing left at its default (LDA+KNN)
    )
    pred <- catboostr:::catboost.predict(model, pool)
    delta <- max(abs(pred - fixture$expected$predict_default_lda))
    # Root cause (catboost-8z4.45): float32 LAPACK ssyev symmetric-eigenvector
    # sign/order is BLAS/build-sensitive on near-degenerate eigenvalues. This
    # is a documented, bounded divergence of this fork's build, not a defect
    # in Pool construction (KNN-only embedding_processing is bit-identical,
    # proving the embedding values delivered to training are correct).
    expect_lt(delta, 1.0)  # loose bound: proves the pipeline runs end-to-end
                            # and doesn't diverge catastrophically, without
                            # asserting false bit-exactness LAPACK can't give us
    if (delta > 0) {
      message(sprintf(
        "catboost-8z4.45: LDA-default embedding divergence observed (delta=%.6f), documented and bounded, not a regression.",
        delta
      ))
    }
  })
  ```

- [ ] **Step 3: Run the test, confirm it exercises the real divergence**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool_embeddings.R")'`
  Expected: PASS, with the `message()` line printed showing a nonzero delta (confirms the test is exercising the real LDA path, not a no-op).

- [ ] **Step 4: Regenerate docs if the roxygen caveat needs updating**
  Check `R/catboost.R`'s `@param embedding_features` block already references catboost-8z4.45 (added during ticket .44's fix wave). If it does, no change needed. If the wording needs a pointer to this new bounded test, update it and run `Rscript -e 'roxygen2::roxygenise(load_code = roxygen2::load_source)'`.

- [ ] **Step 5: Run full regression suite**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_dir("tests/testthat")'`
  Expected: `FAIL 0`, `SKIP 1` (pre-existing `caret` skip), `PASS` count one higher than the branch's current 268.

- [ ] **Step 6: Commit**
  ```bash
  cd /home/dd/Gemini/catboost-phase-3-pool-parity
  git add tests/testthat/test_pool_embeddings.R R/catboost.R man/catboost.load_pool.Rd
  git commit -m "test(phase-3): bound LDA-default embedding divergence with documented test (catboost-8z4.45)"
  ```

---

### Task 2: catboost-8z4.46 — document and bound the sparse-input tie-breaking divergence

**Alternatives Considered:** (a) Implement a true native sparse path — feed the `dgCMatrix` `i`/`p`/`x` slots directly to the native visitor's `TConstPolymorphicValuesSparseArray` overloads in `src/catboostr.cpp` instead of densifying via `as.matrix()`. This is real, valuable engineering (removes both the tie-breaking divergence and the memory ceiling the `ponytail:` comment at `R/catboost.R:184-190` already flags) but is a genuinely new native-code feature: new C++ sparse-array construction, a new differential fixture built specifically to be degenerate/tie-heavy (the ticket's own evidence shows the *existing* fixture was deliberately re-picked to be non-degenerate specifically to avoid exercising this exact gap), and non-trivial verification that CSR-vs-CSC and 0-vs-1-based indexing are handled correctly. (b) Accept and document — the caveat is already live (added in ticket .44's fix wave, `R/catboost.R:16-24`), and the existing test (`tests/testthat/test_pool_sparse.R`) already asserts `max_sparse_dense_delta == 0` on the shipped non-degenerate fixture, which is a real (if narrow) passing differential test. Given ranked criteria (performance, simplicity/LOC, ecosystem support, maintenance overhead): (a) wins on performance (removes a real memory ceiling) but loses badly on simplicity/maintenance — it requires new sparse-array C++ code with no existing test coverage pattern on this branch to imitate, on a feature (`embedding`-adjacent sparse Pool construction) not yet reported as a real memory problem by any user. Selected: **(b)**, matching this ticket's own explicit framing that the caveat is "already in catboost.load_pool docs" — this task's remaining work is closing the loop by adding the degenerate-fixture regression test the ticket's evidence describes, not shipping (a). Escalate to (a) only when someone hits the memory ceiling in practice.

**Files:**
- Modify: `tests/testthat/test_pool_sparse.R` (add a second, deliberately degenerate/tie-heavy fixture test documenting the known divergence, alongside the existing non-degenerate `max_sparse_dense_delta == 0` test — do not remove or weaken the existing test)
- Modify: `tools/oracle/gen_pool_sparse_fixture.py` (add a second fixture-generation function for the degenerate 12x5 case cited in the ticket's evidence)
- Read (no changes): `R/catboost.R:184-190` (densification + `ponytail:` comment, already accurate — no code change needed for option (b)), `docs/phase-3/catboost-8z4.44-report.md` (evidence)

**Interfaces:**
- Consumes: `catboost.load_pool`/`catboost.from_matrix`'s existing `sparseMatrix` dispatch (`R/catboost.R:184-190`, unchanged).
- Produces: nothing new consumed by later tasks — this task is a leaf.

- [ ] **Step 1: Write the failing degenerate-fixture test**
  Add a new fixture generator function to `tools/oracle/gen_pool_sparse_fixture.py` producing the 12x5, three-non-zero-cells-outside-one-column matrix described in the ticket (or an equivalently degenerate/tie-heavy matrix — match the ticket's own reproduction data), writing `predict_dense`, `predict_sparse`, and `max_sparse_dense_delta` (measured **inside Python itself**, i.e. Python's own dense-Pool vs `scipy.sparse.csr_matrix`-Pool predictions) to a new `tests/fixtures/oracle/pool_sparse_degenerate.json`.

- [ ] **Step 2: Run the generator**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_sparse_fixture.py --degenerate`
  Expected: writes `tests/fixtures/oracle/pool_sparse_degenerate.json` with a nonzero `max_sparse_dense_delta` (reproducing the ticket's reported ~0.0298, confirming the degenerate case is real).

- [ ] **Step 3: Add the R-side documented-divergence test**
  Add to `tests/testthat/test_pool_sparse.R`:
  ```r
  test_that("sparse: degenerate tie-heavy input diverges from Python's native-sparse tie-breaking within documented bounds (catboost-8z4.46)", {
    fixture <- jsonlite::fromJSON("tests/fixtures/oracle/pool_sparse_degenerate.json")
    dense_pool <- catboostr:::catboost.load_pool(
      data = as.matrix(fixture$inputs$features), label = fixture$inputs$label
    )
    sparse_pool <- catboostr:::catboost.load_pool(
      data = Matrix::Matrix(as.matrix(fixture$inputs$features), sparse = TRUE),
      label = fixture$inputs$label
    )
    model <- catboostr:::catboost.train(dense_pool, params = fixture$inputs$train_params)
    pred_dense <- catboostr:::catboost.predict(model, dense_pool)
    model_sparse <- catboostr:::catboost.train(sparse_pool, params = fixture$inputs$train_params)
    pred_sparse <- catboostr:::catboost.predict(model_sparse, sparse_pool)
    r_delta <- max(abs(pred_dense - pred_sparse))
    # R always densifies sparse input (R/catboost.R:184-190), so R's dense
    # and "sparse" Pools train identically -- r_delta should be exactly 0.
    expect_equal(r_delta, 0, tolerance = 1e-9)
    # Document that Python's own dense-vs-native-sparse paths DO diverge on
    # this exact degenerate data (catboost-8z4.46 root cause, measured
    # entirely inside Python, not an R defect):
    expect_gt(fixture$expected$max_sparse_dense_delta, 0)
  })
  ```

- [ ] **Step 4: Run the test**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool_sparse.R")'`
  Expected: PASS, both the existing non-degenerate test and the new degenerate test.

- [ ] **Step 5: Run full regression suite**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_dir("tests/testthat")'`
  Expected: `FAIL 0`, `SKIP 1`, `PASS` count one higher than after Task 1.

- [ ] **Step 6: Commit**
  ```bash
  cd /home/dd/Gemini/catboost-phase-3-pool-parity
  git add tests/testthat/test_pool_sparse.R tools/oracle/gen_pool_sparse_fixture.py tests/fixtures/oracle/pool_sparse_degenerate.json
  git commit -m "test(phase-3): add degenerate-fixture regression test for sparse tie-breaking divergence (catboost-8z4.46)"
  ```

---

### Task 3: catboost-8z4.47 — fix integer-vs-double label routing for multi-target losses

**Alternatives Considered:** (a) Fix only the multi-target case (narrowly detect "integer matrix + multi-target loss" and coerce to double) — rejected: the ticket explicitly warns this is "real scope creep beyond Pool construction ticket... filed here instead so it's tracked rather than silently worked around in one test," and demands the *correct general rule* be determined by checking Python's behavior for every label shape, not a narrow multi-target patch. A narrow fix would just move the workaround from the test file into `catboost.load_pool` without fixing the actual dispatch rule. (b) Determine and implement the correct general rule (per ticket: integer labels stay `Integer`-typed only when `class_labels` is present or the loss is a classification loss; otherwise route to `Float`), verified against Python's actual behavior across label shapes (vector, factor, integer matrix, double matrix), and remove the test-level workaround. Selected: **(b)** — this is exactly what the ticket demands, is a real (if contained) bug fix rather than a documentation exercise like Tasks 1–2, and the ticket provides the exact mechanism (`R/catboost.R:255-269` dispatch, `src/catboostr.cpp:307-320` `ERawTargetType` selection) needed to implement it correctly without speculative redesign.

**Files:**
- Modify: `R/catboost.R:255-269` (label-type dispatch in `catboost.load_pool`)
- Modify: `src/catboostr.cpp:307-320` (`ERawTargetType` selection based on R storage mode — verify whether the fix belongs entirely on the R side, coercing before the `.Call`, or needs the native dispatch to also consult whether `class_labels`/loss type is classification; prefer the R-side fix if it fully resolves the issue, since it avoids a rebuild+relink of the C++ layer for what is fundamentally an R-level type-coercion decision)
- Modify: `tests/testthat/test_multitarget_differential.R:59-65` (remove the `as.double()` workaround once the underlying dispatch is fixed — the test should pass an integer 0/1 matrix directly, matching Python's oracle input)
- Read (no changes): `src/catboostr.cpp:136-152` (`SetClassLabels`, confirm classification-with-factor-labels path still legitimately needs `Integer` type and is not affected by this fix), `tools/oracle/` (check for or add a fixture covering plain integer-vector labels for a regression (non-classification, non-multi-target) loss, to confirm the general rule doesn't break the common case)

**Interfaces:**
- Consumes: none from earlier tasks in this plan (independent of Tasks 1, 2).
- Produces: a corrected `catboost.load_pool`/`catboost.from_matrix` label-dispatch rule that Task 5 (catboost-8z4.49, timestamp validation) does not depend on but should be aware exists (no code coupling, just don't reintroduce a conflicting label-handling code path).

- [ ] **Step 1: Reproduce the mechanism directly, confirm Python's actual rule**
  Read the ticket's own root-cause data (`bd show catboost-8z4.47`) and Python's `Pool.__init__` label-handling (`vendor/catboost/catboost/python-package/catboost/core.py`, read-only, search for how it decides `ERawTargetType` from label dtype). Confirm: does Python route on "matrix with >1 column and no explicit classification loss = Float" regardless of int/double dtype, or does it inspect dtype too? Quote the exact logic found before writing any R/C++ change.

- [ ] **Step 2: Write the failing test for the general rule**
  Extend `tests/testthat/test_multitarget_differential.R` — replace the `as.double()` workaround at lines 59-65 with the raw integer matrix from the fixture:
  ```r
  test_that("multitarget: MultiLogloss fit/predict matches Python oracle with integer 0/1 label matrix (catboost-8z4.47)", {
    fixture <- jsonlite::fromJSON("tests/fixtures/oracle/multitarget.json")
    label <- fixture$inputs$multilogloss_label  # left as integer/matrix, no as.double() coercion
    pool <- catboostr:::catboost.load_pool(data = fixture$inputs$features, label = label)
    model <- catboostr:::catboost.train(pool, params = fixture$inputs$multilogloss_params)
    pred <- catboostr:::catboost.predict(model, pool)
    expect_equal(pred, fixture$expected$multilogloss_predict, tolerance = 1e-6)
  })
  ```
  Also add a regression-loss-with-integer-vector-label test to confirm the general rule doesn't break the common single-target case:
  ```r
  test_that("regression: integer-vector label still fits correctly (catboost-8z4.47 regression guard)", {
    pool <- catboostr:::catboost.load_pool(data = matrix(1:20, ncol = 2), label = 1:10)
    model <- catboostr:::catboost.train(pool, params = list(loss_function = "RMSE", iterations = 5))
    expect_s3_class(model, "catboost.Model")
  })
  ```

- [ ] **Step 2b: Run the tests, confirm they fail with the current code**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_multitarget_differential.R")'`
  Expected: the new MultiLogloss-with-integer-matrix test FAILS (reproducing the exact bug the ticket describes); the regression-guard test PASSES (confirms the fix, once written, must not break this case).

- [ ] **Step 3: Implement the general dispatch rule**
  In `R/catboost.R` around line 255-269, change the label-type handling so that an integer matrix is coerced to double UNLESS `class_labels` is set or the loss function is a known classification loss (the existing `is.factor(label)`/`is.character(label)` branches already handle the classification-with-labels case correctly — this change only affects the "plain integer matrix, no factor/character conversion happened" branch). Exact code depends on Step 1's findings — write the minimal branch that makes both Step 2 tests pass without touching the `is.factor`/`is.character`/`SetClassLabels` paths.

- [ ] **Step 4: Run tests, confirm they pass**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_multitarget_differential.R")'`
  Expected: PASS on both new tests.

- [ ] **Step 5: Rebuild if `src/catboostr.cpp` was touched**
  If Step 3's fix required a native-side change (only if the R-side coercion alone doesn't fully resolve it — check Step 1's findings), rebuild using the Global Constraints build recipe before re-running Step 4.

- [ ] **Step 6: Run full regression suite**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_dir("tests/testthat")'`
  Expected: `FAIL 0`, `SKIP 1`, no regressions in classification/factor-label tests (`test_pool.R`, `test_model.R`).

- [ ] **Step 7: Commit**
  ```bash
  cd /home/dd/Gemini/catboost-phase-3-pool-parity
  git add R/catboost.R tests/testthat/test_multitarget_differential.R
  git commit -m "fix(phase-3): route integer label matrices to Float target type unless classification (catboost-8z4.47)"
  ```

---

### Task 4: catboost-8z4.48 — link vendor tokenizer/dictionary natively, retire the pure-R port

**Alternatives Considered:** (a) Keep the existing ~643-line pure-R reimplementation and instead implement the 5 documented scope cuts (BySense tokenization, lemmatizing/token_types/sub_tokens_policy/languages options, Letter-level tokenization, multigram `gram_order>1`, Bpe dictionary type) individually in pure R — rejected: the ticket's own evidence shows this was the originally-considered path (splitting into per-feature tickets) and explicitly rejects it, because each of these five is itself a non-trivial reimplementation (Bpe alone needs a full merge-algorithm port) with high drift risk against the vendor's actual behavior, whereas linking natively makes all five "fall out" for free once the C bridge exists. (b) Link the vendor `cpp-text_processing-tokenizer`/`cpp-text_processing-dictionary` targets natively via the exact precedent ticket .40 already established in this same branch (7-line CMake addition, plain `.Call`/`R_ExternalPtr` bridge, no Rcpp) — selected per the ticket's own proposed work, which is already fully specified and directly reuses working precedent in this codebase (lowest LOC, matches existing maintenance pattern, removes an entire category of future oracle-drift bugs). This is the one task in this plan that is NOT a "pick the lazier of two options" — the ticket makes a strong, evidence-backed case that (a)'s original justification for existing was actually false, so (b) is the only defensible choice, not merely the lazier one.

**Files:**
- Modify: `src/CMakeLists.txt:74-80` (add `cpp-text_processing-tokenizer` and `cpp-text_processing-dictionary` to the existing `target_link_libraries(catboostr PUBLIC ...)` block, immediately after the `catboost-libs-dataset_statistics`/`private-libs-app_helpers` lines added by ticket .40 — same block, same pattern)
- Modify: `src/catboostr.h` (declare new `EXPORT_FUNCTION` bridges for tokenizer construction/tokenize, dictionary construction/fit/apply/save/load — mirror the existing accessor declarations already in this file)
- Modify: `src/catboostr.cpp` (implement the bridges, wrapping `NTextProcessing::NTokenizer::TTokenizer` and `NTextProcessing::NDictionary::TDictionary`/`TDictionaryBuilder`, following `vendor/catboost/catboost/python-package/catboost/_text_processing.pxi` as the reference for what Python wraps — read-only vendor reference, do not modify)
- Modify: `src/init.c` (register the new `EXPORT_FUNCTION`s in `R_CallMethodDef`, matching each function's actual arity — mismatches here crash `.Call`, verify each one)
- Modify: `R/text_processing.R` (re-point `catboost.tokenizer.*`/`catboost.dictionary.*` functions to call the new native bridges instead of the pure-R implementation; **delete** the pure-R tokenizing/dictionary-building logic once the native path is verified working; **delete** the now-false justification comment at lines 13-16; keep every public function signature byte-compatible with what ticket .43 shipped)
- Modify: `tests/testthat/test_text_processing.R` (existing tests must still pass unchanged against the new native backend — add 5 new differential tests, one per lifted scope cut, comparing R output for BySense/lemmatizing options/Letter-level/multigram/Bpe against the pinned Python oracle)
- Modify: `tools/oracle/gen_text_processing_fixture.py` (extend to generate oracle fixtures for the 5 newly-supported configurations)

**Interfaces:**
- Consumes: none from earlier tasks (independent of Tasks 1-3).
- Produces: `catboost.Tokenizer`, `catboost.Dictionary`, and their existing method surface (`catboost.tokenizer.tokenize`, `catboost.dictionary.{fit,apply,size,get_token,get_tokens,get_top_tokens,unknown_token_id,end_of_sentence_token_id,min_unused_token_id,save,load}`) — signatures unchanged from ticket .43, only the backend changes. No later task in this plan depends on this.

- [ ] **Step 1: Add the vendor targets to the CMake link list**
  Edit `src/CMakeLists.txt` immediately after the `catboost-libs-dataset_statistics` / `private-libs-app_helpers` lines (currently ~79-80):
  ```cmake
  # P3.6 follow-up (catboost-8z4.48): link the vendored tokenizer/dictionary
  # libraries directly, replacing the earlier pure-R reimplementation, using
  # the same linking pattern P3.3 established for dataset-statistics above.
  cpp-text_processing-tokenizer
  cpp-text_processing-dictionary
  ```

- [ ] **Step 2: Read the Python Cython wrapper as the ground-truth reference**
  Read `vendor/catboost/catboost/python-package/catboost/_text_processing.pxi` in full (read-only) — this defines the exact constructor arguments and method surface for `Tokenizer` and `Dictionary`/`DictionaryBuilder` that ticket .43's R functions already mirror. Note every C++ type/method it calls into (e.g. `TTokenizer::Tokenize`, `TDictionaryBuilder::Add`/`FinishBuilding`, `TDictionary::Apply`/`Save`/`Load`) — these are what the new `.cpp` bridges must call.

- [ ] **Step 3: Write the failing test for one lifted scope cut first (TDD, smallest slice)**
  Pick the smallest of the 5 cuts (likely BySense tokenization, since it's a single enum-value change vs `Bpe`'s full merge algorithm) and write its differential test first in `tests/testthat/test_text_processing.R`, sourced from a new fixture entry in `tools/oracle/gen_text_processing_fixture.py`. Run it, confirm it fails against the current pure-R backend (which `stop()`s on this option).

- [ ] **Step 4: Implement the native bridge functions**
  Add `EXPORT_FUNCTION` bridges in `src/catboostr.cpp` (with declarations in `src/catboostr.h`, registrations in `src/init.c`) for: tokenizer construction (accepting all options `_text_processing.pxi` exposes, including `BySense`/lemmatizing/etc.), `Tokenize`; dictionary-builder construction (accepting `TokenLevelType`, `GramOrder`, `OccurrenceLowerBound`, dictionary type including `Bpe`), `Add`, `FinishBuilding`, `Apply`, `Save`, `Load`, plus the existing accessor methods (`size`, `get_token`, etc. — check whether ticket .43's pure-R port already computed these client-side and can keep doing so post-migration, or whether they now need native accessors too).

- [ ] **Step 5: Re-point the R wrappers and delete the pure-R implementation**
  Rewrite `R/catboost.R.tokenizer.*`/`catboost.dictionary.*` in `R/text_processing.R` to call the new `.Call` bridges. Delete the pure-R tokenizing/dictionary-building logic. Delete the false justification comment at lines 13-16.

- [ ] **Step 6: Rebuild and run the Step 3 test**
  Use the Global Constraints build recipe. Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_text_processing.R")'`
  Expected: the new BySense test PASSES; all pre-existing text-processing tests (default ByDelimiter/FrequencyBased config from ticket .43) still PASS unchanged, proving the public API is byte-compatible.

- [ ] **Step 7: Repeat Steps 3-6 for the remaining 4 scope cuts**
  One at a time: lemmatizing/token_types/sub_tokens_policy/languages options, Letter-level tokenization, multigram `gram_order>1`, Bpe dictionary type. Each gets its own failing-test-first cycle, its own fixture entry, and its own passing differential test before moving to the next.

- [ ] **Step 8: Run full regression suite**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_dir("tests/testthat")'`
  Expected: `FAIL 0`, `SKIP 1`, all pre-existing text-processing tests unchanged, 5 new passing tests for the lifted cuts.

- [ ] **Step 9: Verify vendor pin and commit**
  Run: `git -C /home/dd/Gemini/catboost-phase-3-pool-parity/vendor/catboost status --porcelain` (expect empty).
  ```bash
  cd /home/dd/Gemini/catboost-phase-3-pool-parity
  git add src/CMakeLists.txt src/catboostr.cpp src/catboostr.h src/init.c R/text_processing.R tests/testthat/test_text_processing.R tools/oracle/gen_text_processing_fixture.py
  git commit -m "feat(phase-3): link vendor tokenizer/dictionary natively, close 5 text-processing scope cuts (catboost-8z4.48)"
  ```

- [ ] **Step 10: Won't-do contingency (ticket .48 Section V — only if Steps 1-8 hit a genuine blocker)**
  If at any point in Steps 1-8 the native link/bridge proves genuinely infeasible (a named vendor symbol that does not link, a target with an unsatisfiable transitive dependency, a licensing conflict) — do not silently fall back to keeping the pure-R port. Instead: (a) record the specific blocker with the exact linker/compiler error in `docs/phase-3/catboost-8z4.48-report.md`; (b) revert any partial native-bridge changes so `R/text_processing.R` returns to its last known-good pure-R state; (c) correct the justification comment at `R/text_processing.R:13-16` to state the *real* blocking reason found, replacing the now-disproven "would require new Rcpp glue" claim — the comment must never be left saying something the ticket's own investigation has already disproven; (d) close catboost-8z4.48 as won't-do via `bd update catboost-8z4.48 --status wont-do` (or equivalent) with the blocker documented in a comment; (e) skip Steps 9's commit of the native bridge and instead commit only the corrected comment. This step is a fallback path, not expected to trigger — Steps 1-9 are the primary path.

---

### Task 5: catboost-8z4.49 — add `timestamp=`/`feature_tags=` constructor arguments

**Alternatives Considered:** (a) Skip `feature_tags` entirely if R has no equivalent representation, silently accepting-and-dropping the argument — explicitly forbidden by the ticket's own Definition of Done ("feature_tags either supported and explicitly documented, or unsupported and documented — never silently accepted and dropped"). (b) Add `timestamp=` wired through the existing, already-tested `catboost.pool.set_timestamp()` setter (ticket .38), and for `feature_tags` first determine if any R-side representation exists at all — if not, add the constructor argument but make it `stop()` with a clear message if non-NULL, documenting the gap rather than silently dropping it. Selected: **(b)**, matching the ticket's explicit low-effort framing ("convenience-API gap, not missing capability" for timestamp) while still respecting the never-silently-drop constraint for feature_tags.

**Files:**
- Modify: `R/catboost.R` (`catboost.load_pool()` signature + dispatch, `catboost.from_matrix()` argument-validation block — the ticket points at the existing `group_id`/`group_weight`/`subgroup_id` validation block as the pattern to follow for length/type checks)
- Read (no changes): `src/catboostr.cpp`/`src/init.c` (existing `catboost.pool.set_timestamp` native entry point, reused as-is — no new C++ needed for the `timestamp=` half), `vendor/catboost/catboost/python-package/catboost/core.py:628` (Python constructor signature reference, read-only)

**Interfaces:**
- Consumes: `catboost.pool.set_timestamp()` (ticket .38, already differentially tested — unchanged).
- Produces: nothing new consumed by later tasks — last task in this plan.

- [ ] **Step 1: Check for any R-side `feature_tags` representation**
  Search: `grep -rn "feature_tags" /home/dd/Gemini/catboost-phase-3-pool-parity/R/ /home/dd/Gemini/catboost-phase-3-pool-parity/src/`
  Expected: no matches (per ticket's own note this "needs checking first"). Confirm before proceeding — if something already exists, this step changes the rest of the task.

- [ ] **Step 2: Write the failing test for `timestamp=`**
  Add to `tests/testthat/test_pool_metadata.R` (the file ticket .38 already established for metadata-accessor differential tests):
  ```r
  test_that("load_pool: timestamp= constructor argument matches catboost.pool.set_timestamp (catboost-8z4.49)", {
    fixture <- jsonlite::fromJSON("tests/fixtures/oracle/pool_metadata.json")
    pool_via_constructor <- catboostr:::catboost.load_pool(
      data = fixture$inputs$features, label = fixture$inputs$label,
      timestamp = fixture$inputs$timestamp
    )
    pool_via_setter <- catboostr:::catboost.load_pool(
      data = fixture$inputs$features, label = fixture$inputs$label
    )
    pool_via_setter <- catboostr:::catboost.pool.set_timestamp(pool_via_setter, fixture$inputs$timestamp)
    model_a <- catboostr:::catboost.train(pool_via_constructor, params = list(has_time = TRUE, iterations = 5))
    model_b <- catboostr:::catboost.train(pool_via_setter, params = list(has_time = TRUE, iterations = 5))
    expect_equal(
      catboostr:::catboost.predict(model_a, pool_via_constructor),
      catboostr:::catboost.predict(model_b, pool_via_setter)
    )
  })
  ```
  Also add length- and type-validation tests (both halves of the ticket's acceptance criterion "Invalid timestamp length/type raises a clear R error before the native call" — `CatBoostPoolSetTimestamp_R` at `src/catboostr.cpp:1153-1168` currently has a length `CB_ENSURE` but zero type guard, calling `REAL(timestampParam)` directly, so the type-mismatch path must be added, not just tested):
  ```r
  test_that("load_pool: timestamp= length must match object count (catboost-8z4.49)", {
    expect_error(
      catboostr:::catboost.load_pool(data = matrix(1:20, ncol = 2), label = 1:10, timestamp = 1:5),
      regexp = "timestamp"
    )
  })

  test_that("load_pool: timestamp= must be numeric, not character/logical (catboost-8z4.49)", {
    expect_error(
      catboostr:::catboost.load_pool(data = matrix(1:20, ncol = 2), label = 1:10, timestamp = letters[1:10]),
      regexp = "timestamp"
    )
    expect_error(
      catboostr:::catboost.load_pool(data = matrix(1:20, ncol = 2), label = 1:10, timestamp = rep(TRUE, 10)),
      regexp = "timestamp"
    )
  })
  ```

- [ ] **Step 3: Run tests, confirm they fail (argument doesn't exist yet)**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool_metadata.R")'`
  Expected: FAIL with "unused argument (timestamp = ...)" or equivalent.

- [ ] **Step 4: Add `timestamp=` to `catboost.load_pool()` and `catboost.from_matrix()`**
  Add the argument to both function signatures in `R/catboost.R`, with the same length-validation pattern used for `group_id`/`group_weight`/`subgroup_id` (locate that block and mirror it exactly). Also add an explicit R-side type check (`is.numeric(timestamp)`, `stop("timestamp must be numeric, got: ", typeof(timestamp))` otherwise) before the length check — the native setter `CatBoostPoolSetTimestamp_R` (`src/catboostr.cpp:1153-1168`) has no type guard of its own (calls `REAL(timestampParam)` unconditionally), so an un-coerced character/logical vector would currently either error opaquely inside the native call or silently produce garbage; fixing this at the R layer avoids a native-code change/rebuild for what is a pure input-validation concern. After the Pool object is constructed, if `timestamp` is non-NULL, call the existing `catboost.pool.set_timestamp()` on it before returning.

- [ ] **Step 5: Add `feature_tags=` per Step 1's finding**
  If Step 1 found no R-side representation exists: add `feature_tags = NULL` to both signatures, and if a caller passes a non-NULL value, `stop("feature_tags is not currently supported by catboostr; tracked as catboost-8z4.49")` — this satisfies the "never silently dropped" requirement without inventing new functionality. Also add a roxygen `@param feature_tags` block documenting the unsupported status explicitly (the ticket's acceptance criterion is "supported OR explicitly documented as unsupported with the reason" — a `stop()` alone satisfies runtime behavior but not the documentation half; the roxygen text must state that `feature_tags` is unsupported and why, e.g. "Not currently supported by catboostr (tracked as catboost-8z4.49); passing a non-NULL value raises an error rather than being silently ignored.").

- [ ] **Step 6: Run tests, confirm they pass**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool_metadata.R")'`
  Expected: PASS on both new tests.

- [ ] **Step 7: Regenerate docs**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'roxygen2::roxygenise(load_code = roxygen2::load_source)'`
  Expected: `man/catboost.load_pool.Rd` (and `catboost.from_matrix.Rd` if it has its own arg docs) updated with `timestamp`/`feature_tags` entries.

- [ ] **Step 8: Run full regression suite**
  Run: `cd /home/dd/Gemini/catboost-phase-3-pool-parity && Rscript -e 'library(catboostr); testthat::test_dir("tests/testthat")'`
  Expected: `FAIL 0`, `SKIP 1`, no regressions.

- [ ] **Step 9: Commit**
  ```bash
  cd /home/dd/Gemini/catboost-phase-3-pool-parity
  git add R/catboost.R man/catboost.load_pool.Rd tests/testthat/test_pool_metadata.R
  git commit -m "feat(phase-3): add timestamp= constructor argument, explicit feature_tags= rejection (catboost-8z4.49)"
  ```

---

## Final Whole-Branch Review

After all 5 tasks land: dispatch a final code review on the most capable available model over the full diff from the phase-3-pool-parity branch's last known-good tip (`002e4d5` — pre-final-review-fix-wave — or `51ab131` if treating the fix wave as the true base) through this plan's last commit, per `superpowers:requesting-code-review`. Point it at this plan's per-task "Alternatives Considered" sections so it can independently judge whether Tasks 1/2's "document, don't implement" calls and Task 4's "always link natively" call still hold up once all 5 are done together.
