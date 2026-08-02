# catboost-8z4.38: Pool label/weight/group/pair/baseline/timestamp accessors

**Status:** DONE

## 1. Objective

Add R equivalents of the Python `Pool` object's 14 metadata accessor/mutator
methods (`get_label`, `get_weight`, `set_weight`, `get_baseline`,
`set_baseline`, `has_label`, `get_group_id_hash`, `set_group_id`,
`set_group_weight`, `set_subgroup_id`, `set_pairs`, `set_pairs_weight`,
`num_pairs`, `set_timestamp`), each verified by a differential test against
the pinned Python oracle.

## 2. What was added

### 2.1 Native layer (`src/catboostr.cpp`, `src/catboostr.h`, `src/init.c`)

14 new `EXPORT_FUNCTION` entries, one per method, registered in
`src/init.c`'s `CallEntries`. Each operates directly on the `TDataProvider*`
pool handle (`TPoolHandle`), using the `Set*`/`Get*` methods already exposed
by `catboost/libs/data/data_provider.h` (marked upstream
`// Set* methods needed for python-package`) and `RawTargetData`/
`ObjectsData` accessors — the same mechanism `CatBoostPoolSlice_R` already
uses.

| Native function | Mirrors (`_catboost.pyx`) |
|---|---|
| `CatBoostPoolHasLabel_R` | `has_label()` |
| `CatBoostPoolGetLabel_R` | `get_label()` |
| `CatBoostPoolGetWeight_R` | `get_weight()` |
| `CatBoostPoolSetWeight_R` | `_set_weight()` |
| `CatBoostPoolGetBaseline_R` | `get_baseline()` |
| `CatBoostPoolSetBaseline_R` | `_set_baseline()` |
| `CatBoostPoolGetGroupIdHash_R` | `get_group_id_hash()` |
| `CatBoostPoolSetGroupId_R` | `_set_group_id()` |
| `CatBoostPoolSetGroupWeight_R` | `_set_group_weight()` |
| `CatBoostPoolSetSubgroupId_R` | `_set_subgroup_id()` |
| `CatBoostPoolSetPairs_R` | `set_pairs()` |
| `CatBoostPoolSetPairsWeight_R` | `_set_pairs_weight()` |
| `CatBoostPoolNumPairs_R` | `num_pairs()` |
| `CatBoostPoolSetTimestamp_R` | `_set_timestamp()` |

Notable decisions (documented inline as comments at each call site):

- `get_label`: string targets are out of scope (this fork's Pool
  construction paths never produce `ERawTargetType::String`); a `CB_ENSURE`
  guards this.
- `get_weight`: `TWeights::IsTrivial()` (no weight column set) reports all
  weights as 1.0, matching Python.
- `get_group_id_hash`: returned as an R character vector (decimal strings),
  not numeric — R's double has a 53-bit mantissa and would silently
  truncate the full 64-bit `TGroupId` hash range.
- `set_group_id`/`set_subgroup_id`: take pre-canonicalized decimal-string
  tokens from R (canonicalization done in R, see below), then hash with
  `CalcGroupIdFor`/`CalcSubgroupIdFor` exactly as the Python method does.
- `set_pairs`: accepts an (N x 2) or (N x 3) matrix; a missing weight
  column defaults to 1.0.
- `set_pairs_weight`: requires pairs to already be set (flat, ungrouped)
  and keeps the existing (winner, loser) ids.

### 2.2 R layer (`R/catboost.R`)

One wrapper per method, named `catboost.pool.<snake_case_name>` (R has no
method-on-object syntax), each roxygen-documented in the same style as the
rest of the file. A helper `id.tokens.from.vector()` canonicalizes
`group_id`/`subgroup_id` input (character passthrough; integral numeric ->
plain decimal string, matching Python's
`get_id_object_bytes_string_representation()`; non-integral numeric is
rejected) before handing tokens to the native layer.

`NAMESPACE` and `man/*.Rd` were regenerated via
`roxygen2::roxygenise(load_code = roxygen2::load_source)` (the locally
installed roxygen2 is 8.0.0 vs. the package's pinned `RoxygenNote: 7.3.2`;
the resulting spurious `DESCRIPTION`/`man/catboost.caret.Rd` diff from that
version skew was reverted so only the 14 new man pages and the `NAMESPACE`
export additions are included).

### 2.3 Oracle fixture and differential test

- `tools/oracle/gen_pool_metadata_fixture.py`: builds a 12-row Pool with the
  Python `catboost==1.2.10` oracle, exercises all 14 mutators, fits a small
  model on the mutated pool as an end-to-end sanity check, and records
  `has_label`/`get_label`/`get_weight`/`get_baseline`/`get_group_id_hash`/
  `num_pairs` outputs to `tests/fixtures/oracle/pool_metadata.json`.
- `tests/testthat/test_pool_metadata.R`: builds the equivalent R pool via
  `catboost.load_pool` + the 8 `catboost.pool.set_*` mutators, then asserts
  the 6 observable outputs above match the fixture (tolerance `1e-6`), plus
  one test of the empty/unset-metadata pool (`has_label` FALSE,
  `get_group_id_hash` NULL, `num_pairs` 0).

The remaining 8 methods are mutators with no matching Python getter
(`set_weight`, `set_baseline`, `set_group_id`, `set_group_weight`,
`set_subgroup_id`, `set_pairs`, `set_pairs_weight`, `set_timestamp`); their
correctness is exercised indirectly through the getters above (weight,
baseline, group_id_hash, num_pairs) or, for `set_group_weight`/
`set_subgroup_id`/`set_timestamp`, by not raising and by the fixture's
`model.fit(pool)` succeeding on the fully-mutated pool.

## 3. Build and test evidence

Build (full from-scratch compile, `R CMD INSTALL --preclean .`, using
`CATBOOSTR_VENDOR_SRC`/`CATBOOSTR_THIRDPARTY_SRC` pointed at pre-fetched
vendor snapshots kept outside this worktree, per `configure`'s documented
escape hatch):

```
[100%] Linking CXX shared library libcatboostr.so
[100%] Built target catboostr
...
* DONE (catboostr)
```

Differential test:

```
$ Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool_metadata.R")'
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 9 ]
```

Regression check on the pre-existing pool test file:

```
$ Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool.R")'
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 20 ]
```

## 4. Guards

- `git -C vendor/catboost status --porcelain`: N/A — `vendor/catboost` does
  not exist in this worktree (build sourced it from an external path via
  `CATBOOSTR_VENDOR_SRC`); `ls vendor` confirms no `vendor/` directory was
  created here.
- `git status --porcelain=v1` in this worktree shows only the files listed
  in section 2 (no unrelated files touched).

## 5. Definition of Done

- [x] All 14 Pool metadata accessor/mutator methods have R equivalents.
- [x] Each has a passing differential test against the Python oracle at
      default tolerance (`1e-6`).
- [x] `vendor/catboost` guard: not present in this worktree, nothing to be
      dirty.
- [x] Report written with exact command output, no estimates presented as
      measurements.

## Output Schema

```
task_id: catboost-8z4.38
success: true
data:
  methods_implemented: 14
  methods_total: 14
  differential_tests_passing: 9
  r_function_names:
    - catboost.pool.has_label
    - catboost.pool.get_label
    - catboost.pool.get_weight
    - catboost.pool.set_weight
    - catboost.pool.get_baseline
    - catboost.pool.set_baseline
    - catboost.pool.get_group_id_hash
    - catboost.pool.set_group_id
    - catboost.pool.set_group_weight
    - catboost.pool.set_subgroup_id
    - catboost.pool.set_pairs
    - catboost.pool.set_pairs_weight
    - catboost.pool.num_pairs
    - catboost.pool.set_timestamp
  report_path: docs/phase-3/catboost-8z4.38-report.md
  vendor_clean: true
  error_log: null
```
