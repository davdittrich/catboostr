# catboost-8z4.44 (P3.7) — Pool construction parity: multi-target labels, embeddings, sparse/CSR

**Status:** COMPLETE
**Branch/worktree:** `phase-3-pool-parity` @ `/home/dd/Gemini/catboost-phase-3-pool-parity`
**Date:** 2026-08-03

## V. Output Schema

```yaml
task_id: catboost-8z4.44
success: true
data:
  multitarget_differential_passing: true
  embeddings_implemented: true
  sparse_implemented: true
  differential_tests_passing: 12   # test_that blocks; 38 passing expectations
  get_object_importance_touched: false
  report_path: docs/phase-3/catboost-8z4.44-report.md
  vendor_clean: true
error_log: null
```

## IV.1 — P2.4 "Finding 2" read, claim reproduced directly

`docs/phase-2/P2.4-report.md` "Finding 2: multi-target losses" root-causes the
multi-target failure to `catboost.get_object_importance` alone
(`vendor/catboost` `ders_helpers.cpp`'s scalar-loss allowlist in
`GetEvaluateDerivativesFunc` reached via `target.h`'s
`GetOneDimensionalTarget()`); `catboost.load_pool` / `catboost.train` /
`catboost.predict` were reported already correct for multi-target losses.

Reproduced directly, **before** any rebuild, against the *installed* package
and the freshly generated oracle fixture (so nothing in this ticket's C++
changes could have influenced the result):

```
dim X 16 3 dim Y 16 2
num_col 3 num_row 16
label dim 16 2
              [,1]          [,2]
[1,] 1.00000000000 5.00000000000
[2,] 1.29999995232 4.30000019073
pred dim 16 2
              [,1]          [,2]
[1,] 1.88779479901 2.92847882009
[2,] 1.95688657106 2.76726470602
expected
              [,1]          [,2]
[1,] 1.88779479901 2.92847882009
[2,] 1.95688657106 2.76726470602
maxdiff 0
```

`maxdiff 0` — R's existing fit/predict path already matches the Python oracle
bit-for-bit for MultiRMSE. Confirmed the exclusion boundary: **no file under
`get_object_importance`'s call path was read for modification and
`catboost.get_object_importance` was not touched** (`git diff` below shows the
only R change to a non-constructor function is documentation).

## IV.2 — Multi-target differential test (part a)

New oracle: `tools/oracle/gen_multitarget_fixture.py` →
`tests/fixtures/oracle/multitarget.json` (16 rows, 3 tie-free float features,
MultiRMSE float 16x2 label matrix and MultiLogloss binary 16x2 label matrix,
`iterations=10, depth=3, learning_rate=0.3, random_seed=42, thread_count=1`).

```
catboost.__version__=1.2.10
multirmse_predict[0]=[1.8877947990093715, 2.9284788200907093]
multilogloss_predict[0]=[-0.4344834740485522, -0.7019706534000839]
OK
```

New test: `tests/testthat/test_multitarget_differential.R` — asserts
`num_row`/`num_col`, the full `get_label()` matrix, and the full
`predict(..., "RawFormulaVal")` matrix for both losses at `tolerance = 1e-6`.
No new implementation was added for this part.

## IV.3 — `embedding_features=` support (part b)

**Contract.** Python holds embedding features *inside* the data frame (a
column whose cells are arrays) and names their positions with
`Pool(data, embedding_features=[2])`. An R matrix cell cannot hold a vector,
so R takes them *beside* the matrix, exactly as it already does for text
features:

```r
catboost.load_pool(m, label, embedding_features = list(emb = emb_matrix))
catboost.from_matrix(m, label, embedding_features_data = list(...),
                     embedding_features_indices = 2L)
```

Each list element is an (objects x dimension) numeric matrix; list names
become feature names; without explicit indices the embeddings occupy the
trailing flat feature indices, which is what makes the R Pool identical to the
oracle's `[f0, f1, emb]` layout.

**Native glue.** `CatBoostCreateFromMatrix_R` previously hardcoded
`TVector<ui32>{}, // TODO(akhropov) support embedding features in R` as the
embedding-feature index vector, so no Pool built by this package could ever
have an embedding feature. It now takes two more `SEXP` arguments
(`embeddingListParam`, `embeddingFeaturesIndicesParam`, arity 15 → 17 in
`src/init.c` and `src/catboostr.h`), forwards the indices to the
`TFeaturesLayout` constructor, and feeds each column-major R matrix into
`visitor->AddEmbeddingFeature(j, MakeTypeCastArraysHolderFromVector<float, float>(...))`.
The stale "always empty in this fork" comment on
`CatBoostPoolGetEmbeddingFeatureIndices_R` was corrected in the same commit.

Oracle: `tools/oracle/gen_pool_embeddings_fixture.py` →
`tests/fixtures/oracle/pool_embeddings.json` (24 rows, 2 float features + one
dimension-4 embedding feature at flat index 2).

```
catboost.__version__=1.2.10
embedding_feature_indices=[2]
feature_names=['f0', 'f1', 'emb']
predict[:3]=[-0.8012606671808449, 0.9709191306602332, -0.7585434678016344]
OK
```

Test: `tests/testthat/test_pool_embeddings.R` — `num_row`, `num_col`,
`get_feature_names`, `get_embedding_feature_indices`, empty cat/text index
sets, `get_label`, a `from_matrix`-with-explicit-indices equivalence check,
input-validation errors, and a full `predict` round-trip at `1e-6`.

### Finding: LDA embedding calcer diverges from the Python wheel (pre-existing, out of scope)

The fixture pins `embedding_processing = {"default": ["KNN"]}` rather than
CatBoost's default `["LDA", "KNN"]`. Reason, established by control
experiment: with `iterations=1, depth=1` and *only* the LDA calcer, R and
Python disagree, while with *only* the KNN calcer they agree exactly.

```
R LDA -0.218181826852 0.184615391951 -0.218181826852 0.184615391951
PY LDA [-0.27272728  0.23076924 -0.27272728  0.23076924]
R KNN -0.171428578241 0.240000009537 -0.171428578241 0.240000009537
PY KNN [-0.17142858  0.24000001 -0.17142858  0.24000001]
```

KNN is a distance-based calcer over the raw embedding vectors, so its exact
agreement is itself proof that the embedding values delivered through the new
matrix path are bit-identical to Python's.

To rule out this ticket's construction path as the cause, the same data was
written to a dsv file with a `NumVector` column and loaded through the
*identical file loader* on both sides:

```
3 [2] ['f0', 'f1', 'emb']
PY file LDA [-0.27272728  0.23076924 -0.27272728  0.23076924]
3 2
R file LDA -0.218181826852 0.184615391951 -0.218181826852 0.184615391951
```

Same file, same loader, same divergence — so it predates and is independent of
Pool construction (this fork's compiled binary vs. the pinned Python wheel).
Also verified not a parameter difference: R's serialized training parameters
are `{"thread_count":1,"logging_level":"Silent","loss_function":"Logloss","iterations":10,"learning_rate":0.3,"random_seed":42,"depth":3}`
and Python's `get_all_params()` resolves the same
`embedding_processing {"default": ["LDA","KNN"]}`; and with the embedding
feature removed both sides produce identical predictions
(`-0.742779260339 0.350840772163 -0.823070940383` on both). Not fixed here —
out of this ticket's scope; the default-processing oracle predictions are
still recorded in the fixture as `predict_default_lda` for a follow-up.

**Filed as catboost-8z4.45** (parity debt). Root cause, confirmed on review:
LDA solves a float32 symmetric eigenproblem (LAPACK `ssyev`) whose eigenvector
signs and near-degenerate eigenvalue ordering are not canonicalized and are
therefore BLAS/build-sensitive — not a defect in the new embedding-matrix glue.
Because `embedding_processing` defaults to `["LDA","KNN"]`, a user who does not
set it explicitly hits the divergent path; the `catboost.load_pool`
`embedding_features` documentation now carries that caveat and recommends
`embedding_processing = list(default = list("KNN"))` for parity-sensitive use.

## IV.4 — Sparse / CSR support (part c)

`catboost.from_matrix` (and therefore `catboost.load_pool`) now accepts any
`Matrix` `sparseMatrix` and densifies it via `as.matrix()` before the native
call. This is exactly the ticket's "converting to whatever format the native
glue expects": CatBoost's sparse column format is a memory optimisation whose
unstored cells are ordinary zeros, so the resulting Pool must equal the dense
one. A `ponytail:` comment in `R/catboost.R` records the ceiling (memory) and
the upgrade path (feed the `dgCMatrix` `i/p/x` slots to the visitor's
`TConstPolymorphicValuesSparseArray` overloads). `Matrix` was **not** already
declared in `DESCRIPTION`; it is added to `Suggests` (test-only — the R code
uses no `Matrix` symbol, `as.matrix()` dispatches on the caller's object).

Oracle: `tools/oracle/gen_pool_sparse_fixture.py` →
`tests/fixtures/oracle/pool_sparse.json` (40x6, 92/240 stored values, Pool
built from `scipy.sparse.csr_matrix`).

```
catboost.__version__=1.2.10
nnz=92 of 240
max_sparse_dense_delta=0.0
predict[:3]=[-1.656087323333636, 1.4379872415878974, 1.0190921327430642]
OK
```

Test: `tests/testthat/test_pool_sparse.R` — asserts against the oracle's
**scipy-sparse** Pool (`num_row`, `num_col`, `get_feature_names`, `get_label`,
the full `get_features()` matrix, and the full `predict` round-trip at `1e-6`),
plus `dgRMatrix` (row-major/CSR) acceptance.

### Finding: sparse-vs-dense training ties on degenerate data

An earlier candidate fixture (12x5 with only three non-zero cells outside one
column) produced `max |dense - sparse| = 0.0298` **inside Python itself** —
CatBoost's tie-breaking among equal-scoring split candidates differs between
its sparse and dense column layouts. Quantization borders and `get_features()`
were identical in that case, and a 1-iteration/1-depth model was identical,
so the divergence is pure tie-breaking. The shipped fixture therefore uses a
non-degenerate ~40 %-filled matrix, on which the oracle's own sparse and dense
paths agree exactly (`max_sparse_dense_delta = 0.0`, asserted in the test).
This is what makes densification in R observationally equivalent here, and the
test records it rather than assuming it.

**Filed as catboost-8z4.46** (parity debt): because R densifies, an R Pool
built from a `sparseMatrix` reproduces Python's *dense* Pool, never its native
sparse one, so it cannot match Python's sparse-layout tie-breaking on
degenerate data. The `catboost.load_pool` `data` documentation now says so; the
upgrade path (feed the `dgCMatrix` `i/p/x` slots to
`TConstPolymorphicValuesSparseArray`) is recorded in the ticket.

### Finding: integer label matrices take the class-label path (filed as catboost-8z4.47)

`tests/testthat/test_multitarget_differential.R:59-65` casts the oracle's
integer 0/1 MultiLogloss label matrix to double. That workaround is required
because `R/catboost.R:237-251` keeps an integer label integer and
`src/catboostr.cpp:307-320` dispatches the target type on the R *storage mode*
(`Rf_isInteger(targetParam)` → `ERawTargetType::Integer` + `SetClassLabels`),
whereas Python decides by loss/target semantics rather than numpy dtype. Not
fixed here — changing that dispatch touches every label path (binary,
multiclass, regression, ranking, factor labels) and is scope creep for a Pool
construction ticket; it is now tracked instead.

## IV.5 — Build and test output (verbatim)

```
cd /home/dd/Gemini/catboost-phase-3-pool-parity
CATBOOSTR_VENDOR_SRC=/home/dd/Gemini/catboost/vendor/catboost \
CATBOOSTR_CYTHON=/tmp/catboostr-cython-venv/.venv/bin/cython \
R CMD INSTALL --preclean .
```

```
[100%] Building CXX object catboostr-fork-build/CMakeFiles/catboostr.dir/catboostr.cpp.o
[100%] Building C object catboostr-fork-build/CMakeFiles/catboostr.dir/__vcs_version__.c.o
[100%] Building C object catboostr-fork-build/CMakeFiles/catboostr.dir/init.c.o
[100%] Linking CXX shared library libcatboostr.so
[100%] Built target catboostr
make: Leaving directory '/tmp/catboostr-build.k9ZT9B/build'
*** installed /tmp/catboostr-build.k9ZT9B/build/catboostr-fork-build/libcatboostr.so -> inst/libs/libcatboostr.so
** libs
make: Nothing to be done for 'all'.
** R
** inst
** byte-compile and prepare package for lazy loading
** help
*** installing help indices
** building package indices
** testing if installed package can be loaded from temporary location
** checking absolute paths in shared objects and dynamic libraries
** testing if installed package can be loaded from final location
** testing if installed package keeps a record of temporary installation path
* DONE (catboostr)
```

Three new test files:

```
== test_multitarget_differential.R
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 9 ]
== test_pool_embeddings.R
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 15 ]
== test_pool_sparse.R
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 14 ]
```

Full suite (`Rscript tests/testthat.R`), no regressions:

```
[ FAIL 0 | WARN 0 | SKIP 1 | PASS 260 ]

══ Skipped tests (1) ═══════════════════════════════════════════════════════════
• {caret} is not installed. (1): 'test_caret_parameter_tuning.R:35:3'
```

(The single skip is pre-existing: `caret` is a `Suggests` package absent from
this machine.)

## IV.6 — Section III guard re-check

| Guard | Result |
|---|---|
| Read-only pin on `vendor/catboost/` | `git -C vendor/catboost status --porcelain` → empty output, exit 0. **vendor_clean: true** |
| Anti-scope-creep: no `get_object_importance` MultiClass fix | `catboost.get_object_importance` and `vendor/catboost`'s `ders_helpers.cpp` / `target.h` untouched. **get_object_importance_touched: false** |
| Logic: each of the 3 scenarios needs a passing differential test | 3 test files, 12 `test_that` blocks, 38 expectations, all asserting against pinned `catboost==1.2.10` output — no "no exception thrown" tautologies; every scenario round-trips real oracle values |
| Dependency on catboost-8z4.42 (P3.5) | Extended, not forked: the `FeaturesData` branch of `catboost.from_matrix`'s `data=` dispatch is untouched; sparse coercion is a new sibling branch ahead of it, embeddings are new trailing arguments |
| Isolation | All work in the dedicated worktree; `/home/dd/Gemini/catboost` untouched |

## Files changed

| File | Change |
|---|---|
| `src/catboostr.cpp` | `CatBoostCreateFromMatrix_R` +2 params, embedding index vector wired into `TFeaturesLayout`, `AddEmbeddingFeature` branch; stale comment on `CatBoostPoolGetEmbeddingFeatureIndices_R` corrected |
| `src/catboostr.h` | declaration updated |
| `src/init.c` | `CatBoostCreateFromMatrix_R` arity 15 → 17 |
| `R/catboost.R` | `catboost.load_pool(embedding_features=)`, `catboost.from_matrix(embedding_features_data=, embedding_features_indices=)`, sparse coercion + validation + docs |
| `man/catboost.load_pool.Rd` | regenerated (`roxygen2::roxygenise(load_code = roxygen2::load_source)`) |
| `DESCRIPTION` | `Matrix` added to `Suggests` |
| `tools/oracle/gen_multitarget_fixture.py`, `gen_pool_embeddings_fixture.py`, `gen_pool_sparse_fixture.py` | new oracle generators |
| `tests/fixtures/oracle/multitarget.json`, `pool_embeddings.json`, `pool_sparse.json` | new pinned fixtures |
| `tests/testthat/test_multitarget_differential.R`, `test_pool_embeddings.R`, `test_pool_sparse.R` | new differential tests |

## Confidence

| Claim | Score |
|---|---|
| Multi-target fit/predict matches the Python oracle exactly | 98 — `maxdiff 0` observed pre-build and asserted post-build |
| Embedding features are constructed correctly (values bit-identical to Python) | 95 — exact KNN-calcer prediction agreement plus matching indices/names/shape |
| LDA divergence is pre-existing and unrelated to this ticket | 92 — identical-file/identical-loader control run reproduces it |
| Sparse densification is observationally equivalent to the oracle's CSR path on this fixture | 95 — full `get_features()` and `predict` equality, plus the oracle's own sparse/dense delta of 0 |
| Sparse densification is equivalent on *all* data | 55 — falsified for degenerate tie-heavy matrices (0.0298 delta measured inside Python itself); flagged, not asserted |
