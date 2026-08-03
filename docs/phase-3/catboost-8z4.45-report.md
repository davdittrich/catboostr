# catboost-8z4.45: bounded-divergence differential test for default embedding_processing (LDA+KNN)

## Summary

Added a new `test_that` block to `tests/testthat/test_pool_embeddings.R` that
exercises R's *default* `embedding_processing` (`("LDA", "KNN")`) end-to-end
(fit + predict) and compares against the pinned Python oracle's
`predict_default_lda` fixture values, which were already present in
`tests/fixtures/oracle/pool_embeddings.json` (recorded for auditability by
`tools/oracle/gen_pool_embeddings_fixture.py` since the P3.7 fixture wave).

The divergence between R's LDA calcer and the pinned Python wheel's LDA
calcer is root-caused, permanent, and out of scope to fix: float32 LAPACK
`ssyev` symmetric-eigenvector sign/ordering is BLAS/build-sensitive on the
near-degenerate eigenvalues this dataset produces. The existing KNN-only
test in the same file already proves embedding values reach training
bit-identically, so this is not a Pool-construction bug.

Rather than requiring bit-exact equality (structurally impossible for LDA)
or skipping the default path (leaving it unverified), the new test asserts:

- no `NA`s in the prediction vector,
- `abs(prediction - oracle_lda) < 1.5` for every row (bound; observed actual
  max was `0.403294` -- see below),
- `sign(prediction) == sign(oracle_lda)` for every row (the LDA
  sign/ordering disagreement rotates the projection but must not flip the
  resulting classification decision),
- `any(abs(prediction - oracle_knn) > 1e-3)` (confirms the default path
  really exercises LDA+KNN, not a silent fallback to the KNN-only pin used
  by the sibling test).

## Fixture check (step 1)

`predict_default_lda` key was already present in
`tests/fixtures/oracle/pool_embeddings.json`:

```
$ grep -n "predict_default_lda" tests/fixtures/oracle/pool_embeddings.json
310:    "predict_default_lda": [
```

No fixture regeneration was needed.

## Observed divergence bound (calibration for the 1.5 tolerance)

Ad hoc script run against the built R package before finalizing the test,
comparing R's default-LDA prediction to the oracle's `predict_default_lda`:

```
max abs diff: 0.403293560869
min abs diff: 0.0116676311463
sign match: TRUE
```

The `1.5` tolerance in the committed test gives roughly 3.7x margin above
the observed `0.403294` max divergence -- tight enough to catch a real
regression (e.g. corrupted embedding values or an exploding LDA output)
while accommodating BLAS/build-sensitive variation across environments.

## Step 3: `test_file` output

```
$ Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool_embeddings.R")'
Welcome back dd!
Working directory is: /home/dd/Gemini/catboost-phase-3-pool-parity
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 20 ]
```

All 5 `test_that` blocks passed (20 expectations total; 5 new expectations
added by the new default-LDA test). The LDA no-op-delta sanity check was not
triggered (the `any(abs(prediction - oracle_knn) > 1e-3)` assertion passed,
confirming the default path is genuinely running LDA, not silently
collapsing to KNN-only).

## Step 4: roxygen doc check

`R/catboost.R`'s `@param embedding_features` roxygen block (lines 50-62)
already documents the LDA divergence and cites `catboost-8z4.45`:

```
57:#' here) defaults to \code{list(default = list("LDA", "KNN"))}. The LDA calcer's output is not
58:#' bit-reproducible against the Python package: LDA solves a float32 symmetric eigenproblem whose
...
62:#' Tracked as catboost-8z4.45.
```

No doc change or `roxygen2::roxygenise()` regeneration was needed.

## Step 5: full regression

```
$ Rscript -e 'library(catboostr); testthat::test_dir("tests/testthat")'
...
[ FAIL 0 | WARN 0 | SKIP 1 | PASS 273 ]
```

`FAIL 0`, `SKIP 1` (pre-existing caret skip), as expected. `PASS 273` is
exactly 5 higher than the pre-change baseline of `PASS 268` (confirmed by
`git stash`-ing this change and re-running `test_dir` before restoring it),
matching the 5 new expectations in the added `test_that` block.

## Step 6: vendor pin check

```
$ git -C vendor/catboost status --porcelain
```

Empty output -- vendor untouched, as required (read-only pin).

## Files changed

- `tests/testthat/test_pool_embeddings.R`: added the new
  `test_that("pool embeddings: default (LDA+KNN) fit/predict diverges from
  the Python oracle only within a bounded, documented tolerance", ...)`
  block; updated the adjacent comment on the pre-existing KNN-pinned test to
  say "not a usable exact-equality probe" (was "not a usable differential
  probe" -- corrected since it is now used differentially, just not for
  exact equality).
- `docs/phase-3/catboost-8z4.45-report.md`: this report (new file).

No rebuild of the compiled R package (`src/catboostr.cpp`/`.h`/`init.c` or
any R export signature) was needed -- this was a test-file and doc-only
change.
