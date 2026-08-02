# catboost-8z4.40 report — P3.3: Pool quantization support + CLI dataset-statistics mode

Date: 2026-08-02
Branch: `phase-3-pool-parity`
Worktree: `/home/dd/Gemini/catboost-phase-3-pool-parity`

## I. Objective (recap)

Add R equivalents of Python `Pool.quantize`, `Pool.is_quantized`,
`Pool.save_quantization_borders`, plus an R equivalent of CatBoost CLI's
`dataset-statistics` mode, each verified by a differential test against a
pinned Python/CLI oracle.

## II. Route taken for each capability

### `Pool.quantize` / `Pool.is_quantized` / `Pool.save_quantization_borders`

Read `catboost/python-package/catboost/_catboost.pyx` (`_quantize`,
`is_quantized`, `save_quantization_borders`, lines ~4899–4977 and
~5238–5262) to find upstream's own core entry points:

- `NCB::ConstructQuantizedPoolFromRawPool(TDataProviderPtr, NJson::TJsonValue, TQuantizedFeaturesInfoPtr)`
  (`catboost/libs/data/quantization.h`/`.cpp`) — builds the quantized
  objects data from raw pool data + plain JSON params. This is the exact
  function Python's `Pool.quantize()` calls.
- `dynamic_cast<const TQuantizedObjectsDataProvider*>(pool->ObjectsData.Get())`
  — the same runtime-type check Python's `is_quantized()`/
  `save_quantization_borders()` use to tell whether a Pool's `ObjectsData` is
  quantized.
- `NCB::SaveBordersAndNanModesToFileInMatrixnetFormat(TString, const TQuantizedFeaturesInfo&)`
  (`catboost/libs/data/borders_io.h`) — writes the borders file, same
  function Python's `save_quantization_borders()` calls.

No new libraries needed linking: `catboost-libs-data` (which provides
`quantization.h`/`borders_io.h`) was already linked into `catboostr` for the
pre-existing Pool accessors.

R glue (`R/catboost.R`):
- `catboost.pool.quantize(pool, params = list())` — reuses
  `catboost.train`'s own `process_synonyms()` +
  `prepare_train_export_parameters()` helpers to turn the same
  `params` list `catboost.train` accepts (`border_count`,
  `feature_border_type`, `nan_mode`, `per_float_feature_quantization`,
  `ignored_features`, ...) into the plain-JSON string
  `ConstructQuantizedPoolFromRawPool` expects, then calls
  `CatBoostPoolQuantize_R`, which mutates the Pool's `ObjectsData` in place
  (matching Python: `self.__pool.Get()[0].ObjectsData = quantizedObjects`).
- `catboost.pool.is_quantized(pool)` / `catboost.pool.save_quantization_borders(pool, output_file)`
  — thin `.Call()` wrappers.

### CLI `dataset-statistics` mode

Read `catboost/app/mode_dataset_statistics.cpp` (17 lines: parses argv into
`TCalculateStatisticsParams`, calls `NCB::CalculateDatasetStatisticsSingleHost(params)`)
and its backing implementation,
`catboost/private/libs/app_helpers/mode_dataset_statistics_helpers.{h,cpp}`
(338 lines total). `TCalculateStatisticsParams` is a plain struct
(`DatasetReadingParams`, `OutputPath`, `HistogramPath`, `ThreadCount`,
`BorderCount`, `OnlyGroupStatistics`, `OnlyLightStatistics`, ...) with a
separate `ProcessParams(argc, argv)` method that parses CLI flags into it —
`CalculateDatasetStatisticsSingleHost` itself takes the already-filled
struct and has no argv/CLI dependency. This is a genuine reusable library
entry point: the R glue fills the struct directly (mirroring
`CatBoostCreateFromFile_R`'s existing `TPathWithScheme`/
`TColumnarPoolFormatParams` wiring) and calls
`NCB::CalculateDatasetStatisticsSingleHost` in-process — no CLI binary
shell-out, as the ticket instructed to avoid unless no library path exists.

New CMake link dependencies added to `src/CMakeLists.txt`
(`target_link_libraries(catboostr PUBLIC ...)`):
`catboost-libs-dataset_statistics`, `private-libs-app_helpers` (the CMake
targets backing the two headers above, confirmed via
`vendor/catboost/catboost/{libs/dataset_statistics,private/libs/app_helpers}/CMakeLists.linux-x86_64.txt`).

R glue: `catboost.dataset_statistics(pool_path, cd_path = "", pairs_path = "", delimiter = "\t", has_header = FALSE, thread_count = -1, border_count = 254, only_group_statistics = FALSE, only_light_statistics = FALSE)`
writes to two `tempfile()`s (mirroring the CLI's own `-o`/`--histograms-path`
default-file convention), calls the native routine, reads both JSON files
back with `jsonlite::fromJSON`, returns `list(statistics = ..., histograms = ...)`
(`histograms` is `NULL` when `only_light_statistics = TRUE`, matching the
CLI mode's own behavior of skipping the second histogram pass — confirmed by
reading `CalculateDatasetStatisticsSingleHost`'s `if (!OnlyLightStatistics)`
guard, which only skips the histogram pass, not the statistics computation).

## III. Refcount correctness note (quantize)

`CatBoostPoolQuantize_R` follows the exact idiom `CatBoostFit_R` already
uses for borrowing a Pool handle into an intrusive-pointer API
(`pool->Ref()` before wrapping the raw `TPoolHandle` in a local
`TDataProviderPtr`): the R external-pointer finalizer
(`_Finalizer<TPoolHandle>`) `delete`s the Pool directly, bypassing
intrusive refcounting, so a local `TDataProviderPtr` going out of scope at
function return must not be allowed to drop the count to 0 and free the
object out from under the still-live R handle.

## IV. Step-by-step outcome

1. Researched Python's `_quantize`/`is_quantized`/`save_quantization_borders`
   and the CLI's `dataset-statistics` mode's C++ backing (Section II above).
2. Implemented `CatBoostPoolQuantize_R`, `CatBoostPoolIsQuantized_R`,
   `CatBoostPoolSaveQuantizationBorders_R`, `CatBoostDatasetStatistics_R` in
   `src/catboostr.cpp` (declared in `src/catboostr.h`, registered in
   `src/init.c`), calling directly into pinned vendor core — no
   reimplementation of quantization logic.
3. Added R wrappers `catboost.pool.quantize`, `catboost.pool.is_quantized`,
   `catboost.pool.save_quantization_borders`, `catboost.dataset_statistics`
   in `R/catboost.R` (+ `NAMESPACE` exports, + `man/*.Rd`).
4. Wrote differential tests:
   - `tests/testthat/test_pool_quantization.R` against a Python
     `catboost==1.2.10` oracle fixture generated by
     `tools/oracle/gen_pool_quantization_fixture.py` →
     `tests/fixtures/oracle/pool_quantization.json` (`is_quantized`
     before/after, `already_quantized`/`save_on_unquantized` error
     parity, and byte-for-byte comparison of the `save_quantization_borders`
     output file's text against the oracle's).
   - `tests/testthat/test_dataset_statistics.R` against the pinned CatBoost
     CLI v1.2.10 oracle binary (`tools/oracle/cli/bin/catboost-v1.2.10
     dataset-statistics`), run on the pre-existing smoke fixture
     (`tests/fixtures/oracle-cli/smoke_data.csv` + `smoke.cd`), with its
     JSON output committed as
     `tests/fixtures/oracle-cli/dataset_statistics_{stats,histograms}.json`.
5. Ran `R CMD INSTALL --preclean .` (exact output below) and both new test
   files (plus the full `testthat` suite for regressions).
6. Re-checked `git -C vendor/catboost status --porcelain` (empty, confirmed
   below).

## V. Output Schema (Strict)

```yaml
task_id: catboost-8z4.40
success: true
data:
  pool_methods_implemented: 3
  pool_methods_total: 3
  dataset_statistics_implemented: true
  differential_tests_passing: 9
report_path: docs/phase-3/catboost-8z4.40-report.md
vendor_clean: true
error_log: null
```

`differential_tests_passing: 9` = 4 passing expectations in
`test_dataset_statistics.R` (3 `test_that` blocks: statistics match,
histograms match, `only_light_statistics` behavior + statistics-still-match)
+ 5 passing expectations in `test_pool_quantization.R` (5 `test_that`
blocks: `is_quantized` before, `is_quantized` after, borders file
byte-for-byte, already-quantized error, save-on-unquantized error). 0
failing, 0 skipped, 0 errors in either file.

## VI. Definition of Done — checklist

- [x] `quantize`, `is_quantized`, `save_quantization_borders` have R
      equivalents with passing differential tests.
- [x] `dataset-statistics` has an R equivalent with passing differential
      test against the CLI oracle.
- [x] `git -C vendor/catboost status --porcelain` empty.
- [x] Report written with exact command output (below).

## VII. Exact command output

### Build (`R CMD INSTALL --preclean .`, tail)

```
[100%] Generating vcs_info.json
[100%] Generating __vcs_version__.c
[100%] Building CXX object catboostr-fork-build/CMakeFiles/catboostr.dir/catboostr.cpp.o
[100%] Building C object catboostr-fork-build/CMakeFiles/catboostr.dir/init.c.o
[100%] Building C object catboostr-fork-build/CMakeFiles/catboostr.dir/__vcs_version__.c.o
[100%] Linking CXX shared library libcatboostr.so
[100%] Built target catboostr
make: Leaving directory '/tmp/catboostr-build-persist/build'
*** installed /tmp/catboostr-build-persist/build/catboostr-fork-build/libcatboostr.so -> inst/libs/libcatboostr.so
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

(Built with `CATBOOSTR_VENDOR_SRC=/home/dd/Gemini/catboost/vendor/catboost`,
plus `CATBOOSTR_PYTHON3=/usr/bin/python3` `CATBOOSTR_CYTHON=/tmp/cyvenv/bin/cython`
to locate this sandbox's Python3/Cython — the fork's `configure` requires
both for the Cython-based C++ source generation step —
and `vendor/thirdparty/{openssl-3.5.7,ragel-6.10,yasm-1.3.0}.tar.gz` copied
in from the read-only main checkout's already-acquired copies (gitignored
`vendor/` in this worktree, not committed).)

### Differential test: `test_pool_quantization.R`

```
pool_quantization:
test_pool_quantization.R: .....

══ DONE ════════════════════════════════════════════════════════════════════════
```

### Differential test: `test_dataset_statistics.R`

```
dataset_statistics:
test_dataset_statistics.R: ....

══ DONE ════════════════════════════════════════════════════════════════════════
```

### Full `testthat` suite (regression check)

```
pool_quantization:
test_pool_quantization.R: .....
pool:
test_pool.R: ...................................................................
version_skew:
test_version_skew.R: .........
dataset_statistics:
test_dataset_statistics.R: ....
model:
on_trimmed_adult_dataset:
pool_introspection:
pool_metadata:
caret_parameter_tuning:

══ Skipped ═════════════════════════════════════════════════════════════════════
1. test caret train and parameter tuning on adult pool ('test_caret_parameter_tuning.R:35:3') - Reason: {caret} is not installed.

══ DONE ════════════════════════════════════════════════════════════════════════
```

All files pass; the single skip (`{caret}` not installed) is pre-existing
and unrelated to this change.

### Vendor cleanliness guard

```
$ git -C /home/dd/Gemini/catboost/vendor/catboost status --porcelain
$ echo exit:$?
exit:0
```

(empty output — no modifications to the read-only pinned vendor snapshot.)

## VIII. Files changed

- `src/catboostr.h` — declarations for `CatBoostPoolQuantize_R`,
  `CatBoostPoolIsQuantized_R`, `CatBoostPoolSaveQuantizationBorders_R`,
  `CatBoostDatasetStatistics_R`.
- `src/catboostr.cpp` — implementations (+ 4 new includes:
  `catboost/libs/data/{borders_io,quantization}.h`,
  `catboost/private/libs/app_helpers/mode_dataset_statistics_helpers.h`).
- `src/init.c` — `R_CallMethodDef` registration for the 4 new routines.
- `src/CMakeLists.txt` — link `catboost-libs-dataset_statistics` and
  `private-libs-app_helpers` into `catboostr`.
- `R/catboost.R` — `catboost.pool.quantize`, `catboost.pool.is_quantized`,
  `catboost.pool.save_quantization_borders`, `catboost.dataset_statistics`.
- `NAMESPACE` — exports for the 4 new functions.
- `man/catboost.pool.quantize.Rd`, `man/catboost.pool.is_quantized.Rd`,
  `man/catboost.pool.save_quantization_borders.Rd`,
  `man/catboost.dataset_statistics.Rd` — new Rd pages (hand-written,
  matching the existing pre-P3.3 Rd pages' format; `roxygen2::roxygenise()`
  was not used because it insists on doing a full package (re)build via
  `pkgbuild`, which does not thread this fork's custom `configure`
  environment variables).
- `tools/oracle/gen_pool_quantization_fixture.py` — Python oracle fixture
  generator.
- `tests/fixtures/oracle/pool_quantization.json`,
  `tests/fixtures/oracle/pool_quantization_borders.tsv` — committed Python
  oracle fixture.
- `tests/fixtures/oracle-cli/dataset_statistics_stats.json`,
  `tests/fixtures/oracle-cli/dataset_statistics_histograms.json` —
  committed CatBoost CLI v1.2.10 oracle fixture (generated from the
  pre-existing `tests/fixtures/oracle-cli/smoke_data.csv`/`smoke.cd`).
- `tests/testthat/test_pool_quantization.R`,
  `tests/testthat/test_dataset_statistics.R` — differential tests.
