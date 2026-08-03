# catboost-8z4.39: Pool feature/shape introspection accessors — report

**Status:** DONE

## I. Summary

Added R equivalents of Python `Pool`'s feature/shape introspection methods:
`get_cat_feature_indices`, `get_embedding_feature_indices`,
`get_text_feature_indices`, `get_feature_names`, `set_feature_names`,
`get_features`, `shape`, `num_col`, `num_row`, `is_empty_`. Verified with a
differential test pinned against a recorded Python 1.2.10 oracle fixture.

This session resumed after a prior implementer session stalled with
substantial uncommitted work (no evidence of a real build having run). That
work was reviewed against the ticket spec, one test bug was found and fixed,
the package was rebuilt and tested for real, and everything was committed.

## II. Methods delivered (10/10)

| Method | R function | Mechanism |
|---|---|---|
| `num_row()` | `catboost.pool.num_row` | reuses existing `CatBoostPoolNumRow_R` (was unwrapped at R level) |
| `num_col()` | `catboost.pool.num_col` | reuses existing `CatBoostPoolNumCol_R` (was unwrapped at R level) |
| `shape` | `catboost.pool.shape` | pure R composition of the two above |
| `is_empty_` | `catboost.pool.is_empty` | pure R composition (`num_row(pool) == 0`) |
| `get_feature_names()` | `catboost.pool.get_feature_names` | new C glue `CatBoostPoolGetFeatureNames_R` |
| `set_feature_names()` | `catboost.pool.set_feature_names` | new C glue `CatBoostPoolSetFeatureNames_R` |
| `get_cat_feature_indices()` | `catboost.pool.get_cat_feature_indices` | new C glue `CatBoostPoolGetCatFeatureIndices_R` |
| `get_text_feature_indices()` | `catboost.pool.get_text_feature_indices` | new C glue `CatBoostPoolGetTextFeatureIndices_R` |
| `get_embedding_feature_indices()` | `catboost.pool.get_embedding_feature_indices` | new C glue `CatBoostPoolGetEmbeddingFeatureIndices_R` (always empty: this package cannot build Pools with embedding features) |
| `get_features()` | `catboost.pool.get_features` | new C glue `CatBoostPoolGetFeatures_R`, numeric-only Pools |

4 of 10 methods reuse existing C/R infrastructure (`num_row`, `num_col`,
`shape`, `is_empty`); 6 required new native glue in `src/catboostr.cpp` /
`src/catboostr.h` / `src/init.c`.

## III. Files changed

- `R/catboost.R` (+157): 10 new exported `catboost.pool.*` wrappers, roxygen docs.
- `src/catboostr.cpp` (+138): 6 new `EXPORT_FUNCTION` C entry points reading
  `pool->MetaInfo.FeaturesLayout` / `pool->ObjectsData`, mirroring
  `_catboost.pyx`'s equivalent Python methods and `CatBoostPoolSlice_R`'s
  existing pattern for reading pool internals.
- `src/catboostr.h` (+17), `src/init.c` (+6): declarations and `.Call`
  registration for the 6 new entry points.
- `NAMESPACE`: 10 new `export(...)` entries (roxygen2-regenerated).
- `man/catboost.pool.{get_cat_feature_indices,get_embedding_feature_indices,get_feature_names,get_features,get_text_feature_indices,is_empty,num_col,num_row,set_feature_names,shape}.Rd`:
  roxygen2-generated man pages.
- `tools/oracle/gen_pool_introspection_fixture.py`: oracle fixture generator,
  pins the Python `catboost==1.2.10` Pool's outputs for a fixed mixed
  numeric/categorical dataset (8 rows) plus separate all-numeric and 0-row
  probes.
- `tests/fixtures/oracle/pool_introspection.json`: the recorded oracle output.
- `tests/testthat/test_pool_introspection.R`: differential test, 8 test
  blocks / 14 expectations, comparing R output to the fixture.
- `DESCRIPTION`: `RoxygenNote: 7.3.2` → `Config/roxygen2/version: 8.0.0`
  (side effect of the installed roxygen2 version regenerating `NAMESPACE`/man
  pages; not itself part of this ticket's scope, left as-is since it was
  already staged by the resumed session and regenerating docs with the older
  field would just flip it back).

## IV. Fix applied this session

`tests/testthat/test_pool_introspection.R` originally built the 0-row probe
pool as `catboost.load_pool(matrix(nrow = 0, ncol = 1))`. Bare `matrix(nrow=,
ncol=)` with no `data=` defaults to a **logical** matrix in R, and
`catboost.from_matrix()` calls `REAL()` on it, so this failed with:

```
Error in `catboost.from_matrix(...)`: REAL() can only be applied to a 'numeric', not a 'logical'
```

Root cause: R's `matrix()` default fill type, not a defect in the new
introspection code. Fixed by constructing an explicitly numeric empty
matrix, matching the Python oracle's `np.empty((0, 1), dtype=np.float32)`:

```r
pool <- catboost.load_pool(matrix(numeric(0), nrow = 0, ncol = 1))
```

## V. Build (exact commands and output)

`R CMD INSTALL --preclean .` requires `cython`/`numpy` at configure time and
no system `cython` is on `PATH`; used a throwaway `uv` venv (consistent with
`docs/phase-3/catboost-8z4.38-report.md` section 6.4), pointed at the
read-only sibling checkout's vendored sources:

```
$ uv venv /tmp/catboostr-fix-venv2/.venv --python 3.12
$ uv pip install --python /tmp/catboostr-fix-venv2/.venv/bin/python cython numpy
$ CATBOOSTR_VENDOR_SRC=/home/dd/Gemini/catboost/vendor/catboost \
  CATBOOSTR_THIRDPARTY_SRC=/home/dd/Gemini/catboost/vendor/thirdparty \
  CATBOOSTR_PYTHON3=/tmp/catboostr-fix-venv2/.venv/bin/python3 \
  CATBOOSTR_CYTHON=/tmp/catboostr-fix-venv2/.venv/bin/cython \
  R CMD INSTALL --preclean .
...
[100%] Linking CXX shared library libcatboostr.so
[100%] Built target catboostr
make: Leaving directory '/tmp/catboostr-build.pC2W95/build'
*** installed /tmp/catboostr-build.pC2W95/build/catboostr-fork-build/libcatboostr.so -> inst/libs/libcatboostr.so
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

Full build ran synchronously in the foreground (no backgrounding), took
~14 minutes wall clock.

## VI. Test (exact command and output)

New differential test file, after the `matrix()` fix:

```
$ Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool_introspection.R")'
No objects info loaded
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 14 ]
```

Full existing test suite (regression check):

```
$ Rscript -e 'library(catboostr); testthat::test_dir("tests/testthat")'
...
[ FAIL 0 | WARN 0 | SKIP 1 | PASS 133 ]
```

(1 pre-existing skip, unrelated to this ticket.)

## VII. Vendor cleanliness check

```
$ ls vendor
ls: cannot access 'vendor': No such file or directory
```

No `vendor/` directory exists in this worktree (gitignored, never created
here).

```
$ git -C /home/dd/Gemini/catboost/vendor/catboost status --porcelain
(empty output)
```

The read-only sibling checkout's `vendor/catboost` (pointed at via
`CATBOOSTR_VENDOR_SRC` for the build) is untouched — empty porcelain status.

## VIII. Output Schema (per ticket Section V)

```yaml
task_id: catboost-8z4.39
success: true
data:
  methods_implemented: 10
  methods_total: 10
  methods_reused_existing_s3: 4  # num_row, num_col, shape, is_empty (reuse pre-existing C glue / pure R composition)
  differential_tests_passing: 14
report_path: docs/phase-3/catboost-8z4.39-report.md
vendor_clean: true
error_log: null
```

## IX. Definition of Done

- [x] All 10 Pool introspection methods have R equivalents (new or reused).
- [x] Has passing differential test against the Python oracle (14/14 pass).
- [x] `git -C vendor/catboost status --porcelain` empty (checked against the
      sibling checkout `vendor/catboost` used via `CATBOOSTR_VENDOR_SRC`; no
      `vendor/` exists in this worktree at all).
- [x] Report written with exact command output.
