# catboost-8z4.49 — timestamp=/feature_tags= constructor arguments

Task 5 of `docs/superpowers/plans/2026-08-03-phase3-parity-debt.md`, executed per plan literally.

## Step 1 — R-side `feature_tags` representation check

```
$ grep -rn "feature_tags" R/ src/
(no output, exit 1)
```

No existing R-side representation. Proceeded per plan's option (b): `timestamp=` wired
through the existing `catboost.pool.set_timestamp()` setter (catboost-8z4.38), `feature_tags=`
rejected with an explicit `stop()` plus roxygen documentation of the unsupported status.

## Step 2/3 — failing tests, confirmed red

Added to `tests/testthat/test_pool_metadata.R`: the constructor-vs-setter parity test, a
length-mismatch test, a type-mismatch test (character/logical), and (in addition to the plan's
literal snippets) a `feature_tags` rejection test for both `catboost.load_pool()` and
`catboost.from_matrix()`, matching the ticket's "never silently dropped" requirement.

```
$ Rscript -e 'library(catboostr); testthat::test_file("tests/testthat/test_pool_metadata.R")'
...
ERROR: 'test_pool_metadata.R:140:3' ---------------
Error in `catboostr:::catboost.load_pool(data.frame(num1 = inputs$num1), label = as.double(inputs$label), timestamp = inputs$timestamp)`: unused argument (timestamp = inputs$timestamp)

[ FAIL 1 | WARN 0 | SKIP 0 | PASS 18 ]
```

## Step 4/5 — implementation

`R/catboost.R`:
- `catboost.load_pool()`: added `timestamp = NULL, feature_tags = NULL`. `feature_tags`
  non-NULL raises `stop("feature_tags is not currently supported by catboostr; tracked as
  catboost-8z4.49")` immediately. `timestamp` gets an R-side `is.numeric()` type check up front
  (mirroring the `group_id`/`group_weight`/`subgroup_id` pattern), is added to the
  read-from-file rejection list (must be `NULL` when `data` is a file path, like the sibling
  args), is forwarded into `catboost.from_matrix()` for the matrix/`catboost.FeaturesData`
  paths, and is applied via `catboost.pool.set_timestamp()` after `catboost.from_data_frame()`
  for the data.frame path (with its own length check against `nrow(data)`, since
  `catboost.from_data_frame()` itself was left untouched per the plan's file list).
- `catboost.from_matrix()`: added `timestamp = NULL, feature_tags = NULL` with the identical
  `feature_tags` rejection, and a `timestamp` type + length validation block mirroring
  `group_id`'s exact pattern (`is.numeric()` check, then `length() != nrow(...)` check). After
  `pool <- .Call("CatBoostCreateFromMatrix_R", ...)`, if `timestamp` is non-NULL,
  `catboost.pool.set_timestamp(pool, timestamp)` is called before `return(pool)` — no native
  code touched, `CatBoostPoolSetTimestamp_R` (`src/catboostr.cpp:1167-1184`) reused as-is.
- Added `@param timestamp` / `@param feature_tags` roxygen blocks to `catboost.load_pool()`.
  `catboost.from_matrix()` has no roxygen block of its own (unexported, undocumented — verified
  via `grep -n "^catboost.from_matrix\|@name catboost.from_matrix" R/catboost.R`, no `@export`
  or doc comment precedes it), matching the plan's Step 7 conditional ("...and
  `catboost.from_matrix.Rd` if it has its own arg docs" — it doesn't).

## Step 6 — tests pass

Environment note: `R CMD INSTALL .` / `pkgload::load_all()` both trigger the package's
`configure` → CMake path unconditionally (`src/` present), which fails in this environment —
`python3` has no `Cython` module (`ModuleNotFoundError: No module named 'Cython'`), and
`tools/oracle/.venv` doesn't provide it either. This is a pre-existing environment gap,
unrelated to this change (any R-only edit in this repo hits the same wall) and confirmed by the
ticket's own guidance not to spend 15+ minutes rebuilding a pure-R change unnecessarily.
Verified instead by re-sourcing `R/catboost.R` into the already-installed `catboostr` namespace
(unlocking bindings, `sys.source("R/catboost.R", envir = asNamespace("catboostr"))`), which
overrides only the R-level function bodies while reusing the existing compiled `.so` — valid
here since no native code changed.

```
$ Rscript -e '
library(catboostr)
ns <- asNamespace("catboostr")
for (nm in ls(ns, all.names = TRUE)) if (bindingIsLocked(nm, ns)) unlockBinding(nm, ns)
sys.source("R/catboost.R", envir = ns)
testthat::test_file("tests/testthat/test_pool_metadata.R")
'
...
[ FAIL 0 | WARN 0 | SKIP 0 | PASS 19 ]
```

## Step 7 — docs regenerated

```
$ Rscript -e 'roxygen2::roxygenise(load_code = roxygen2::load_source)'
Writing 'catboost.load_pool.Rd'
Writing 'catboost.dictionary.size.Rd'
Writing 'catboost.dictionary.get_token.Rd'
```

The latter two files carried pre-existing, unrelated docstring drift (`The token string.` →
`A character string.`, `The number of tokens in the dictionary.` → `An integer.`) not touched by
this task; reverted with `git checkout -- man/catboost.dictionary.get_token.Rd
man/catboost.dictionary.size.Rd` to keep the diff scoped to catboost-8z4.49. Only
`man/catboost.load_pool.Rd` changed, adding the `timestamp`/`feature_tags` signature entries and
`\item{}` docs — diff reviewed and matches the roxygen source exactly.

## Step 8 — full regression

```
$ Rscript -e '
library(catboostr)
ns <- asNamespace("catboostr")
for (nm in ls(ns, all.names = TRUE)) if (bindingIsLocked(nm, ns)) unlockBinding(nm, ns)
sys.source("R/catboost.R", envir = ns)
res <- testthat::test_dir("tests/testthat", reporter = "silent")
df <- as.data.frame(res)
cat("FAIL:", sum(df$failed), "SKIP:", sum(df$skipped), "PASS:", sum(df$passed), "\n")
'
...
FAIL: 0 SKIP: 1 PASS: 301
```

`FAIL 0`, `SKIP 1` (the pre-existing `caret`-not-installed skip), `PASS 301` — baseline 295 plus
6 new expectations across the 4 new `test_that()` blocks. No regressions.

## Step 10 — vendor submodule

```
$ git -C vendor/catboost status --porcelain
(empty)
```

## Files changed

- `R/catboost.R` — `timestamp=`/`feature_tags=` on `catboost.load_pool()` and
  `catboost.from_matrix()`, plus roxygen docs on `catboost.load_pool()`.
- `man/catboost.load_pool.Rd` — regenerated.
- `tests/testthat/test_pool_metadata.R` — 4 new `test_that()` blocks.
