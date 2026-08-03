STATUS: DONE

COMMITS:
bc35513 feat(phase-3): add R equivalents of Pool.save/slice/train_eval_split

TEST_SUMMARY:
`Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool_structural.R", reporter = "list")'`
7 test blocks, 19 assertions total, 0 failed, 0 errors, 0 skipped:
- pool slice: features/label match Python oracle for a contiguous numeric range (3 pass)
- pool slice: categorical feature present raises the native CB_ENSURE error (1 pass)
- pool train_eval_split: has_time=TRUE (no shuffle) matches Python oracle exactly (4 pass)
- pool train_eval_split: is_classification=TRUE (stratified, shuffled) matches Python oracle exactly (4 pass)
- pool train_eval_split: save_eval_pool=FALSE returns eval=NULL, train matches Python oracle (2 pass)
- pool save: saving an unquantized pool raises the same error as Python's Pool.save() (2 pass)
- pool save: quantized-pool binary output round-trips through the pinned Python oracle (3 pass)

Regression check (unchanged, all pass, 0 failed/0 errors):
test_pool.R, test_pool_introspection.R, test_pool_metadata.R, test_pool_quantization.R

REPORT: /home/dd/Gemini/catboost-phase-3-pool-parity/docs/phase-3/catboost-8z4.41-report.md

CONCERNS:
- Compile blocker (confirmed by controller before this run): unqualified `Shuffle(pool->ObjectsGrouping, 1, &rand)`
  resolved via ADL to util/random/shuffle.h's iterator-range template instead of
  `NCB::Shuffle(TObjectsGroupingPtr, ui32, TRestorableFastRng64*)` (objects_grouping.h:223). Fixed by
  fully qualifying as `NCB::Shuffle(...)` (src/catboostr.cpp:663 in the pre-fix draft).
- Second, independent compile error at src/catboostr.cpp:686 (as the controller flagged as possibly separate):
  `ITypedSequence<T>::ForEach` (polymorphic_type_containers.h:351) invokes its visitor as `f(element)` --
  single argument -- not `f(index, element)` like `ITypedArraySubset<T>::ForEach` (line 40 in the same header).
  The draft's lambda `[&classesVec](ui32 i, float value) {...}` matched the wrong overload's contract.
  Fixed with an explicit running index captured by the closure instead.
- Found and fixed two correctness bugs (not compile errors) via the new differential test against the pinned
  Python oracle, confidence 95 (exact code + failing/passing test evidence):
  1. `catboost.pool.slice` (R/catboost.R) passed `dimnames(pool)[[2]]` -- a bare character vector -- as
     `catboost.from_matrix`'s `feature_names` argument, which requires a list (R/catboost.R:226-227 asserts
     `is.list`). Wrapped in `as.list(...)`.
  2. `CatBoostPoolTrainEvalSplit_R`'s stratified branch (src/catboostr.cpp) built `classesVec` from the pool's
     original, pre-shuffle target order, then passed it to `StratifiedTrainTestSplit` alongside
     `postShuffleGrouping`, which represents post-shuffle row positions -- misaligning class labels with rows.
     Python's own `TrainEvalSplit()` (python-package/catboost/helpers.cpp) re-orders the target array by
     `postShuffleGroupingSubset.GetObjectsIndexing()` via `NCB::GetSubset<float>` before stratifying; the R
     glue now does the same when `shuffle` is true. This was caught because the stratified-split test failed
     with exactly one train/eval row swapped relative to the Python oracle before the fix, and passed exactly
     after it.
- `catboost.save_pool` (existing, R/catboost.R:325) genuinely does not satisfy Python's `Pool.save()`: it writes
  a CD/TSV column-description format read back via `catboost.load_pool`'s `column_description` path, whereas
  Python's `Pool.save()` requires an already-quantized pool and writes CatBoost's own binary quantized-pool
  format (`SaveQuantizedPool`, catboost/private/libs/quantized_pool/serialization.h). New `catboost.pool.save`
  wraps a new `CatBoostPoolSave_R` entry point calling that same core function directly; this is new code, not
  a duplicate of `catboost.save_pool`.
- `catboost.pool.save`'s round-trip test is not byte-identity against Python's own output: R's `catboost.load_pool`
  has no `"quantized://"` scheme support, so the test shells out to `tools/oracle/read_quantized_pool.py` via
  `uv run` at test time to read back R's saved file with the pinned Python oracle and compare shape/label; this
  leg is skipped (not failed) if `uv` is unavailable on PATH.
- `vendor/catboost` verified clean both before and after the full session (`git -C vendor/catboost status --porcelain`
  empty at both checks).
- Boundary respected: no P3.1/P3.2/P3.3/P3.5/P3.6 methods touched; diff limited to slice/train_eval_split/save
  glue plus their tests, docs, and oracle fixture tooling.
