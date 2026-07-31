# catboost-8z4.19 (P1.0) — R package skeleton, DESCRIPTION, vendored upstream R sources

**Date:** 2026-07-31 (`date` -> `Fri Jul 31 09:07:15 AM CEST 2026`)
**Repo:** `/home/dd/Gemini/catboost`, branch `phase-0`
**Toolchain:** `R version 4.6.1 (2026-06-24) -- "Happy Hop"`, `Python 3.14.6`
**Result:** SUCCESS. All Section VI DoD items met. One out-of-scope defect found and reported (§8).

---

## 1. Upstream inventory (Step 1)

Source: `vendor/catboost/catboost/R-package/` (pin verified in §7).

```
$ cd vendor/catboost/catboost/R-package && for d in R man tests inst; do echo -n "$d: "; find $d -type f | wc -l; done
R: 3
man: 26
tests: 5
inst: 9
$ du -sh R man tests inst NAMESPACE
112K	R
136K	man
64K	tests
284K	inst
4.0K	NAMESPACE
```

```
$ find inst tests R -type f | sort
inst/CITATION
inst/extdata/adult.cd
inst/extdata/adult_test.1000
inst/extdata/adult_train.1000
inst/extdata/float_with_nan.cd
inst/extdata/float_with_nan.tsv
inst/extdata/multitarget.cd
inst/extdata/multitarget.train
inst/extdata/README.md
R/catboost.caret.R
R/catboost.R
R/install.R
tests/testthat.R
tests/testthat/test_caret_parameter_tuning.R
tests/testthat/test_model.R
tests/testthat/test_on_trimmed_adult_dataset.R
tests/testthat/test_pool.R
```

`inst` = 9 files = the 8 `inst/extdata/` fixtures the ticket names, plus `inst/CITATION`.
`tests` = 5 files = `tests/testthat.R` + 4 `tests/testthat/test_*.R`.
`man` = 26 `.Rd` files.

Confidence 98 (direct `find` output).

### Pre-copy rename-site measurement (against upstream, read-only)

```
$ grep -rn 'package *= *"catboost"' R man tests inst NAMESPACE | wc -l
31
$ grep -rno 'package *= *"catboost"' R man tests | cut -d: -f1 | sort | uniq -c
      2 man/catboost.load_pool.Rd
      3 man/catboost.train.Rd
      5 R/catboost.R
      1 tests/testthat/test_caret_parameter_tuning.R
     10 tests/testthat/test_model.R
      2 tests/testthat/test_on_trimmed_adult_dataset.R
      8 tests/testthat/test_pool.R
$ grep -rn 'library(catboost)' R man tests inst NAMESPACE
tests/testthat.R:2:library(catboost)
$ grep -rn 'catboost::' R man tests inst NAMESPACE
tests/testthat/test_caret_parameter_tuning.R:49:    method = catboost::catboost.caret,
tests/testthat/test_on_trimmed_adult_dataset.R:6:  pool <- catboost::catboost.load_pool(pool.path, column_description = column_description.path)
$ grep -rn 'test_check' tests
tests/testthat.R:4:test_check("catboost")
```

**All three ticket-supplied numbers CONFIRMED by independent measurement:**
31 `package =` sites (21 in `tests/`, 5 in `R/catboost.R`, 5 across **2** `man/` files);
8 `inst/extdata/` files. No discrepancy to flag. Confidence 97.

---

## 2. Copy into the fork (Step 2)

```
$ SRC=vendor/catboost/catboost/R-package
$ cp -a $SRC/R $SRC/man $SRC/inst .
$ cp -a $SRC/NAMESPACE .
$ mkdir -p tests && cp -a $SRC/tests/testthat.R $SRC/tests/testthat tests/
```

`cp -a` copies OUT of the pin; nothing is written into it. `tests/` was merged, not
replaced — only the two upstream entries were copied in.

```
$ find tests -maxdepth 2 -not -path 'tests/fixtures/*' | sort
tests
tests/fixtures
tests/testthat
tests/testthat.R
$ for d in R man inst/extdata tests/testthat; do echo -n "$d: "; find $d -type f | wc -l; done
R: 3
man: 26
inst/extdata: 8
tests/testthat: 4
$ ls inst/extdata inst
inst:
CITATION
extdata

inst/extdata:
adult.cd
adult_test.1000
adult_train.1000
float_with_nan.cd
float_with_nan.tsv
multitarget.cd
multitarget.train
README.md
$ ls tests/fixtures
oracle
oracle-cli
parity
```

**Final `tests/` layout:** `tests/fixtures/{oracle,oracle-cli,parity}` (Phase 0, intact,
still tracked — it does not appear in `git status --porcelain`, see §9) alongside the new
`tests/testthat/` and `tests/testthat.R`. No collision, no clobber. Confidence 98.

---

## 3. DESCRIPTION (Step 3)

Created at `/home/dd/Gemini/catboost/DESCRIPTION`. `Package: catboostr`,
`Version: 1.2.10`, `License: Apache License (== 2.0)`, `Imports: jsonlite`,
`Suggests: caret, testthat, tibble, e1071`, `Biarch: FALSE`, `Encoding: UTF-8`.

Upstream's full `Authors@R` roster and the `CatBoost DevTeam [aut, cph]` copyright
holder are preserved verbatim (Apache-2.0 attribution); the fork maintainer is added as
`cre`, and upstream's `cre` (Stanislav Kirillov) demoted to `aut` since a DESCRIPTION may
carry only one maintainer. Upstream's separate `Author:`/`Maintainer:` fields are dropped
in favour of `Authors@R` alone to avoid a conflicting maintainer declaration.
`SystemRequirements`, licence files and `.Rbuildignore` remain P1.6's (`catboost-8z4.14`)
scope, as the ticket states. Validity proven by §6. Confidence 95.

---

## 4. Rename script (Steps 4-6)

**Path:** `/home/dd/Gemini/catboost/tools/rename-package.py`

**Language choice — Python over R.** Both were viable. Python was chosen because the
transform is pure text: `re.subn` returns the substitution count per pattern for free
(the audit trail the DoD needs), and capture-group replacement `(package\s*=\s*)"catboost"`
-> `\1"catboostr"` preserves upstream's exact whitespace without R's double-escaping of
backslashes in regex string literals. It also removes any dependency on an R runtime for
what is a text edit. The shell allowlist permits `python3 <file>`, so the script is
executed as a file, never inline.

**Idempotence** is structural, not asserted: none of the four patterns can match its own
output (`catboostr::` does not contain the substring `catboost::`; `"catboostr"` does not
match `"catboost"`). The script also refuses to run if its target path contains
`vendor/catboost`.

**Targets:** `R/`, `man/`, `tests/testthat/`, `tests/testthat.R`, `NAMESPACE`.
`man/` is included deliberately — `R CMD check --as-cran` runs Rd examples, so a stale
`package = "catboost"` in an `.Rd` becomes an ERROR under P1.6's zero-ERROR gate.

### Run (Step 4)

```
$ python3 tools/rename-package.py
library: 1
colons: 2
test_check: 1
package_arg: 31
files changed: 8
  R/catboost.R
  man/catboost.load_pool.Rd
  man/catboost.train.Rd
  tests/testthat/test_caret_parameter_tuning.R
  tests/testthat/test_model.R
  tests/testthat/test_on_trimmed_adult_dataset.R
  tests/testthat/test_pool.R
  tests/testthat.R
```

Diff summary: 8 files, 35 substitutions total (1 + 2 + 1 + 31), exactly matching the
pre-copy measurement in §1. **2 of the 8 changed files are under `man/`** — the omission
the plan review caught is closed. No file was hand-edited.

### Idempotence re-run

```
$ python3 tools/rename-package.py
library: 0
colons: 0
test_check: 0
package_arg: 0
files changed: 0
```

### Completeness assertion (Step 5) — all four greps

```
$ for p in 'library(catboost)' 'catboost::' 'test_check("catboost")' 'package = "catboost"'; do
    echo "--- grep -rFn '$p' ---"; grep -rFn "$p" R man tests/testthat tests/testthat.R NAMESPACE; echo "exit=$?"; done
--- grep -rFn 'library(catboost)' ---
exit=1
--- grep -rFn 'catboost::' ---
exit=1
--- grep -rFn 'test_check("catboost")' ---
exit=1
--- grep -rFn 'package = "catboost"' ---
exit=1
```

`exit=1` with no output = grep found nothing. **Zero residuals on all four patterns.**
The whitespace-tolerant regex form agrees:

```
$ grep -rnE 'package[ ]*=[ ]*"catboost"' R man tests/testthat tests/testthat.R NAMESPACE | wc -l
0
```

Positive confirmations (the renamed forms are present, i.e. the text was rewritten, not deleted):

```
$ grep -rn 'library(catboostr)\|catboostr::\|test_check("catboostr")' R man tests/testthat tests/testthat.R
tests/testthat/test_caret_parameter_tuning.R:49:    method = catboostr::catboost.caret,
tests/testthat/test_on_trimmed_adult_dataset.R:6:  pool <- catboostr::catboost.load_pool(pool.path, column_description = column_description.path)
tests/testthat.R:2:library(catboostr)
tests/testthat.R:4:test_check("catboostr")
$ grep -rnoE 'package[ ]*=[ ]*"catboostr"' R man tests/testthat | wc -l
31
```

Confidence 98.

### Minimality / reproducibility (Step 6)

Fresh copy of the pin into scratch, script re-run against it, recursive diff against the
committed tree:

```
$ cp -a $SRC/{R,man,inst,NAMESPACE} $SCR/fresh/ && cp -a $SRC/tests/{testthat.R,testthat} $SCR/fresh/tests/
$ python3 tools/rename-package.py $SCR/fresh
library: 1
colons: 2
test_check: 1
package_arg: 31
files changed: 8
  ... (identical 8-file list)
$ diff -r --brief $SCR/fresh/R R; diff -r --brief $SCR/fresh/man man; diff -r --brief $SCR/fresh/inst inst;
  diff --brief $SCR/fresh/NAMESPACE NAMESPACE; diff -r --brief $SCR/fresh/tests/testthat tests/testthat;
  diff --brief $SCR/fresh/tests/testthat.R tests/testthat.R
DIFF_EXIT_ALL=0
```

**Empty diff.** The fork's `R/`, `man/`, `inst/`, `tests/testthat*` and `NAMESPACE` are
byte-for-byte reproducible from the pinned tree by running the committed script. The
strict-superset provenance holds. Confidence 97.

---

## 5. NAMESPACE verification (Step 7)

```
$ grep -n 'useDynLib' NAMESPACE
36:useDynLib(libcatboostr, .registration = TRUE)
```

Confirms the ticket's `NAMESPACE:36` claim exactly. The shared object keeps upstream's
`libcatboostr` name, so every existing `.Call` site stays valid.

All 19 `export()`ed symbols have a top-level definition in `R/`:

```
catboost.caret catboost.cv catboost.drop_unused_features catboost.eval_metrics
catboost.get_feature_importance catboost.get_model_params catboost.get_object_importance
catboost.get_plain_params catboost.load_model catboost.load_pool catboost.predict
catboost.restore_handle catboost.save_model catboost.save_pool catboost.shrink
catboost.staged_predict catboost.sum_models catboost.train
catboost.virtual_ensembles_predict
```
-> all `OK`, zero `MISS` (loop over `grep -oP '(?<=^export\()[^)]+' NAMESPACE` matched
against `^\s*<name>\s*(<-|=)` in `R/*.R`). Plus 8 `S3method()` registrations
(`dim/dimnames/head/tail/print/summary` on `catboost.Pool`, `predict/print/summary` on
`catboost.Model`). Confidence 96.

---

## 6. R CMD INSTALL (Step 8)

```
$ R CMD INSTALL --library=$SCR/rlib .
* installing *source* package ‘catboostr’ ...
** this is package ‘catboostr’ version ‘1.2.10’
** using staged installation
** R
** inst
** byte-compile and prepare package for lazy loading
** help
*** installing help indices
** building package indices
** testing if installed package can be loaded from temporary location
Error: package or namespace load failed for ‘catboostr’ in library.dynam(lib, package, package.lib):
 shared object ‘libcatboostr.so’ not found
Error: loading failed
Execution halted
ERROR: loading failed
* removing ‘.../scratchpad/rlib/catboostr’
(exit 1)
```

**This is a linking failure, not a metadata one — proven by what succeeded before it:**
DESCRIPTION parsed (`this is package ‘catboostr’ version ‘1.2.10’`), `** R` (all sources
parsed and byte-compiled), `** inst` (fixtures installed), `** help` + `*** installing
help indices` (all 26 Rd files parsed — a malformed Rd would have failed here),
`** building package indices`. The single error is `library.dynam` failing to find
`libcatboostr.so`, which is exactly the object P1.3/P1.4 will produce.

The ticket's premise is also confirmed: this DESCRIPTION had to be created here, since
without it `R CMD INSTALL` cannot proceed at all. Confidence 98.

---

## 7. Read-only pin verification (Step 9)

`vendor/catboost/` **is** a real git checkout (it has its own `.git`), so
`git -C vendor/catboost status` is meaningful here — contrary to the controller's
assumption that it is not. Both checks were run:

```
$ git -C vendor/catboost status --porcelain
              <- no output: clean
$ git -C vendor/catboost rev-parse HEAD
b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084
$ git -C vendor/catboost describe --tags
v1.2.10
$ find vendor/catboost -newermt '-3 hours' -type f | wc -l
0
```

Pin SHA and tag both match the ticket. Zero tracked modifications, zero untracked files,
and zero files anywhere under `vendor/catboost` with an mtime inside this session's
window. `vendor_clean: true`. Confidence 98.

**Method used** (as requested): `git status --porcelain` on the vendored checkout (valid
because it is its own repo) *plus* an mtime sweep `find vendor/catboost -newermt '-3 hours'`,
which catches writes that git would ignore (e.g. into gitignored paths).

---

## 8. FINDINGS / CONCERNS

### 8.1 BLOCKER for the caret test — a FIFTH package-name spelling the ticket does not list

```
$ grep -rnE '"catboost"|\(catboost\)' R man tests/testthat tests/testthat.R inst NAMESPACE DESCRIPTION
R/catboost.caret.R:6:                       library = "catboost",
```

Context (`R/catboost.caret.R:5-7`):

```r
catboost.caret <- list(label = "Catboost",
                       library = "catboost",
                       type = c("Regression", "Classification"))
```

The `library` element of a caret model specification is the list of **R package names**
caret loads before fitting (`caret::train` -> `checkInstall(models$library)`). After this
rename the installed package is `catboostr`, so this string points at a package that does
not exist. `tests/testthat/test_caret_parameter_tuning.R` calls
`caret::train(..., method = catboostr::catboost.caret, ...)` at line 49 and will therefore
fail once the shared library exists and the suite actually runs.

**I did NOT fix it.** The ticket enumerates exactly four spellings and Section III's
Boundary/Anti-pivot guards forbid widening scope unilaterally; adding a fifth pattern
would also change the diff that Step 6's minimality proof pins. This needs its own ticket,
folded into `tools/rename-package.py` as a fifth `SUBS` entry:
`(re.compile(r'(library\s*=\s*)"catboost"'), r'\1"catboostr"')`.

Confidence 92 on the defect (`file:line` quoted; caret's `$library` semantics are from
knowledge of caret's API, not measured here — measurement is impossible until
`libcatboostr.so` exists, which is P1.3/P1.4).

### 8.2 Spec 4.6 gap — reported as the DoD requires

`docs/superpowers/specs/2026-07-30-catboostr-design.md` §4.6 lists only
`library(catboost)`, `test_check("catboost")` and `catboost::`. It omits
`package = "catboost"` — which is by far the largest class at **31 of the 35** total
substitutions, and the only one reaching `man/`. Asserting only the three listed spellings
would let a DoD pass while the installed package cannot find its own fixtures
(`system.file()` returns `""`). §4.6 must be corrected to list four spellings — and, per
§8.1, arguably five. Confidence 95.

### 8.3 `inst/CITATION` was copied

Step 2 says "copy `inst/`", so `inst/CITATION` came along with `inst/extdata/`. It
contains only the two CatBoost arXiv bibentries and no package-name string, so it needed
no rename. Flagged only so P1.6 can decide whether the fork keeps, edits or drops it.
Confidence 97.

### 8.4 Report path discrepancy

The ticket's Section V schema names `docs/phase-1/P1.0-report.md`; the controller's
instruction named `docs/phase-1/catboost-8z4.19-report.md`. The controller's explicit
instruction was followed. Only this file was written.

---

## 9. Changed paths (no git write command was run)

`git status --porcelain` at finish:

```
?? .beads/PRIME.md          <- pre-existing, not mine
?? DESCRIPTION              <- new (§3)
?? NAMESPACE                <- vendored
?? R/                       <- vendored (3 files)
?? inst/                    <- vendored (9 files: 8 extdata + CITATION)
?? man/                     <- vendored (26 files)
?? tests/testthat.R         <- vendored
?? tests/testthat/          <- vendored (4 files)
?? tools/rename-package.py  <- new (§4)
```

Plus `docs/phase-1/catboost-8z4.19-report.md` (this file).

`tests/fixtures/` is absent from the list because it is already tracked and unmodified —
positive evidence it was neither clobbered nor touched.

No `git add`, `commit`, `push`, `checkout`, `stash` or `rm` was run. Nothing was installed
system-wide (`R CMD INSTALL --library=` targeted the scratch dir, and the install was
rolled back by R itself on failure). Confidence 98.
