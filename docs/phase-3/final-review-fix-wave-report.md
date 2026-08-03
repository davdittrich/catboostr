# Phase 3 — final review fix wave

Scope: the whole-branch review of the seven already-landed Phase 3 tickets
(catboost-8z4.38 – .44) produced nine follow-up items. This document records what
was fixed, what was filed as debt, and what was verified as a non-issue.

Branch: `phase-3-pool-parity`. Base of this wave: `002e4d5`.

---

## 1. `catboost.pool.slice` silently dropped group_id, baseline and feature names — FIXED

### Bug mechanism

`catboost.pool.slice()` built its result by round-tripping the sliced rows through a
dense numeric matrix:

    rows <- .Call("CatBoostPoolSlice_R", pool, size, offset)   # dense double matrix
    mat  <- matrix(unlist(rows), nrow = size, byrow = TRUE)
    ...
    catboost.from_matrix(features, label = label, weight = weight,
                         feature_names = feature_names)

`CatBoostPoolSlice_R` flattens each row into `[targets..., weight, features...]`. Every
column outside that layout — group id, subgroup id, baseline, pairs, and (except for a
best-effort re-read from the R-side `.Dimnames` attribute) feature names — was simply
absent from the matrix and therefore absent from the reconstructed Pool. The
reconstruction also could not represent categorical, text or embedding features at all,
which is why the native entry point carried a `CB_ENSURE` rejecting such pools.

Python does not have this problem: `Pool.slice()`
(`vendor/catboost/catboost/python-package/catboost/core.py:1135`) calls `_take_slice()`
(`_catboost.pyx:5197`), which builds the new Pool with
`TDataProvider::GetSubset(GetGroupingSubsetFromObjectsSubset(...))` — the core subset
machinery, which carries every column through.

### Fix mechanism

Added a native entry point that does exactly what `_take_slice()` does:

- `src/catboostr.cpp` — `CatBoostPoolSliceSubset_R(pool, size, offset)`: builds a
  contiguous `TRangesSubset` over `[offset, min(objectCount, offset + size))`, feeds it
  through `GetGroupingSubsetFromObjectsSubset(pool->ObjectsGrouping, ..., Ordered)`, and
  returns `pool->GetSubset(...)` as an external pointer with the usual
  `_Finalizer<TPoolHandle>`.
- `src/catboostr.h` — declaration.
- `src/init.c` — `R_CallMethodDef` registration (3 args).
- `R/catboost.R` — `catboost.pool.slice()` is now three lines: call the new entry
  point, copy the Pool attributes, return. This mirrors how the sibling
  `catboost.pool.train_eval_split()` (catboost-8z4.41) already worked.

The subset construction is deliberately the same idiom already proven to compile in
this file (the `TRangesSubset`/`TSubsetBlock` form used by the older
`CatBoostPoolSlice_R`), combined with the `pool->GetSubset(...)` + external-pointer
return already used by `CatBoostPoolTrainEvalSplit_R`.

### Unchanged components

- `CatBoostPoolSlice_R` is untouched and still registered. `head.catboost.Pool()` and
  `tail.catboost.Pool()` continue to use it: they genuinely want a dense numeric
  matrix for printing, not a Pool.
- `catboost.pool.slice()`'s R signature and argument validation are unchanged.

### Consequences

- group_id / subgroup_id, baseline, weights, all target columns, pairs and the feature
  layout (names) now survive a slice.
- Categorical, text and embedding features now slice correctly instead of raising
  `CB_ENSURE`. This removes a restriction relative to Python rather than adding one.
- The only remaining difference from Python is the row selector: Python accepts an
  arbitrary index array (`rindex`), R exposes a contiguous `[offset, offset + size)`
  range. This is stated in the roxygen block.

### Tests

`tests/testthat/test_pool_structural.R`:

- New test *"pool slice: group_id, baseline and feature names survive the slice"* —
  builds a Pool with `group_id` and a `baseline` matrix, slices it, and asserts the
  sliced feature names (via both `catboost.pool.get_feature_names()` and
  `dimnames()`), the sliced baseline rows and the sliced group-id hashes. `baseline`
  is compared with `tolerance = 1e-6` because the Pool stores it as float32.
- The old test *"pool slice: categorical feature present raises the native CB_ENSURE
  error"* is replaced by *"pool slice: categorical features are preserved, matching
  Python's Pool.slice()"*, asserting the sliced pool keeps the same categorical
  feature indices and names. Asserting the old error would now be asserting a defect.
- The existing Python-oracle differential assertions on features/label are unchanged
  and still pass.

## 2. `catboost.pool.slice` roxygen restriction comment — FIXED

The old block claimed only a categorical/text/embedding restriction and said nothing
about the group_id / baseline / feature-name loss. Since item 1 removes both problems,
the block was rewritten to state what is now true: the Pool is built by the same
`GetSubset` machinery Python uses, so all column kinds are preserved, and the sole
divergence from Python is the contiguous-range row selector. `man/catboost.pool.slice.Rd`
regenerated.

## 3. `catboost.pool.set_feature_names` vs. `dimnames()` divergence — FIXED

### Bug mechanism

`catboost.pool.set_feature_names()` mutates the C++-side `TFeaturesLayout` in place
(`CatBoostPoolSetFeatureNames_R` → `SetExternalFeatureIds`). `dimnames.catboost.Pool()`
returned the R-side `.Dimnames` attribute, set once at construction time. After a
`set_feature_names()` call the two disagreed: `catboost.pool.get_feature_names()`
reported the new names, `dimnames(pool)[[2]]` and `colnames(pool)` reported the old
ones (or `NULL`, for pools loaded from file, whose names live only on the C++ side).

Patching the setter cannot fix this: R semantics mean an attribute assignment inside
`catboost.pool.set_feature_names()` would apply to a local copy, not to the caller's
object. That is precisely why the setter mutates C++ state.

### Fix mechanism

`dimnames.catboost.Pool()` now reads column names from the Pool itself
(`CatBoostPoolGetFeatureNames_R`), returning `NULL` when the Pool has no names (empty
vector, or all names empty strings) so the previous "unnamed pool ⇒ `NULL` colnames"
contract is preserved. This is the single point all callers route through —
`colnames()`, `print.catboost.Pool()`, and the feature-importance naming at
`R/catboost.R:3118` all go via `dimnames()`.

The vestigial `.Dimnames` attribute is still written at construction time and still
copied by `catboost.pool.slice()` / `catboost.pool.train_eval_split()`; it is simply no
longer the source of truth. Removing it was left out of scope for this wave.

### Side effect (an improvement)

Pools loaded from file with a column description now report their feature names through
`dimnames()`/`colnames()`; previously they reported `NULL` because `catboost.from_file()`
sets `.Dimnames = list(NULL, NULL)` unconditionally.

## 4. False "R cannot read quantized://" claim — FIXED

### Verification

The claim in `tools/oracle/read_quantized_pool.py:7-8` (and echoed in
`tools/oracle/gen_pool_structural_fixture.py` and the header comment of
`tests/testthat/test_pool_structural.R`) that R's `catboost.load_pool` has no
`quantized://` scheme support is **false**. Verified directly against the freshly built
package: quantizing a 20×2 Pool, saving it with `catboost.pool.save()`, and reloading it
with

    catboost.load_pool(paste0("quantized://", path), column_description = "")

returns a valid `catboost.Pool` of `dim` 20 × 2.

### Is the Python helper still needed?

Yes, and the corrected comments now say why. The point of that test leg is not "can the
bytes be read back at all" — it is "can they be read by an *independently built,
version-pinned* CatBoost (`catboost==1.2.10`)". Re-reading with the same in-tree build
that wrote the file would demonstrate only self-consistency and would not pin the
on-disk format. The helper and its test coverage are kept; only the justification text
was corrected, in all three places that repeated it.

## 5. Round-trip test skipped on genuine corruption — FIXED

### Bug mechanism

The round-trip test treated *any* nonzero exit status from the Python reader as grounds
for `testthat::skip()`:

    status <- attr(result, "status")
    if (!is.null(status) && status != 0) {
      testthat::skip(paste("Python oracle round-trip check failed to run:", ...))
    }

A malformed or corrupted saved pool makes `Pool(data="quantized://...")` raise, the
script exits nonzero, and the test skips — silently swallowing exactly the defect it
exists to catch.

### Fix mechanism

Environment availability is now decided once, up front, by a probe that cannot be
affected by the saved pool:

1. `skip_if(Sys.which("uv") == "")` — uv absent.
2. `run_oracle(repo_root, c("-c", shQuote("import catboost")))` — resolves the pinned
   `--frozen --project tools/oracle` environment and imports catboost, touching no
   pool file. Nonzero ⇒ `skip_if` with the captured output.

After both probes pass, the environment is known good, so the actual reader invocation
is asserted, not skipped: `expect_equal(exit_status(result), 0L, info = <captured
stdout+stderr>)`. A reader that runs but rejects the pool now fails the test loudly and
prints the Python traceback. Two small helpers (`run_oracle`, `exit_status`) remove the
duplication between probe and real call.

`shQuote` is required because `system2()` does not quote its arguments, so
`c("-c", "import catboost")` would reach python as `-c import` plus a stray `catboost`.

Confirmed in the full-suite run: the round-trip test executes and passes (the only
skipped test in the suite is the pre-existing caret one).

## 6. Tokenizer/dictionary parity debt — FILED as **catboost-8z4.48**

`catboost-8z4.43`'s ~643-line pure-R reimplementation of the vendor tokenizer/dictionary
logic is justified at `R/text_processing.R:13-16` by the claim that reusing the vendor
implementation "would require adding new Rcpp glue and wiring it into the build". Two
facts in this branch contradict it: catboost-8z4.40 added exactly that kind of build
wiring in 7 lines of CMake (`src/CMakeLists.txt:74-80`), and this package uses no Rcpp
at all (the bridge is plain `.Call`/`R_ExternalPtr`). CMake targets for both
`library/cpp/text_processing/tokenizer` and `.../dictionary` already exist in the vendor
tree.

Filed as ONE self-contained ticket rather than three, because the five documented scope
cuts (`docs/phase-3/catboost-8z4.43-report.md:299-306` — BySense tokenization;
lemmatizing/token_types/sub_tokens_policy/languages; Letter-level and `gram_order > 1`
dictionaries; Bpe dictionaries; file-path `fit`) are all consequences of not linking the
vendor library. Once the native path is in, they fall out of it rather than needing five
separate implementations.

Ticket: `catboost-8z4.48`, parent `catboost-8z4`, P2, labels
`parity-debt,phase-3,discovered-from:catboost-8z4.43`.

## 7. Constructor-signature parity for `timestamp` / `feature_tags` — FILED as **catboost-8z4.49**

Python's Pool constructor takes `timestamp=` and `feature_tags=`
(`vendor/catboost/catboost/python-package/catboost/core.py:628`); R's
`catboost.load_pool` / `catboost.from_matrix` take neither. For `timestamp` the
capability exists via `catboost.pool.set_timestamp()` (already differentially tested in
catboost-8z4.38), so this is a convenience-API gap. `feature_tags` needs checking — the
ticket requires it to be either supported or explicitly documented, never silently
accepted and dropped.

Ticket: `catboost-8z4.49`, parent `catboost-8z4`, P3, labels
`parity-debt,phase-3,discovered-from:catboost-8z4.38`.

## 8. `man/catboost.caret.Rd` lost `\docType{data}` / `\format{}` / `\keyword{datasets}` — VERIFIED NON-ISSUE

Investigated, no change made.

- The sections were dropped in commit `8741843`, not `8d459e4`.
- The roxygen source block for this object (`R/catboost.caret.R:1-4`) contains only
  `@name`, `@title` and `@export`. It has never contained `@docType`, `@format` or
  `@keywords` — `git log -- R/catboost.caret.R` shows only the phase-1 import
  (`46b979a`) and an unrelated package-name string rename (`fae3617`), and a grep for
  those tags across all of `R/` finds them nowhere except two `@keywords internal` in
  `R/zzz.R`.
- Those three sections were previously **auto-generated** by older roxygen2 for exported
  non-function (data) objects. roxygen2 8.0.0 no longer emits them automatically.

So nothing was accidentally removed from this branch's sources; this is a roxygen2
version behaviour change, outside the branch's control. Restoring the sections would
mean adding roxygen tags that never existed upstream. Left as is, recorded here.

## 9. Stale file:line citations in catboost-8z4.46 / .47 — FIXED

Line numbers had shifted after the roxygen caveats added in `49ba66d` / `002e4d5`, and
shifted again when item 1 shortened `catboost.pool.slice()`. Citations were re-verified
against the current tree and the ticket descriptions updated in place:

| Ticket | Was | Now | Anchor verified |
|---|---|---|---|
| .46 | `R/catboost.R:166-172` | `R/catboost.R:184-190` | the `inherits(..., "sparseMatrix")` densification block and its `ponytail:` comment |
| .46 | `R/catboost.R:17-19` | `R/catboost.R:16-24` | the `@param data` roxygen block carrying the sparse caveat |
| .47 | `R/catboost.R:237-251` | `R/catboost.R:255-269` | `is.character(label)` through the `stop("Unsupported label type"...)` guard |
| .47 | `R/catboost.R:240-246` | `R/catboost.R:258-264` | the `is.factor(label)` branch |

---

## Verification

Build (`R CMD INSTALL --preclean`) with the branch's standard recipe: clean.

Full suite, `testthat::test_dir("tests/testthat")` against the freshly installed
package:

    [ FAIL 0 | WARN 0 | SKIP 1 | PASS 268 ]

The single skip is pre-existing and unrelated: `{caret} is not installed.` at
`test_caret_parameter_tuning.R:35:3`. The Python-oracle quantized round-trip test is
**not** skipped — it runs and passes, confirming the item-5 restructuring.

## Not in scope for this wave

Embedding-only `catboost.pool.save` / `catboost.save_pool` behaviour, and anything under
`vendor/catboost/` (read-only; `git status --porcelain vendor/catboost` is clean).
