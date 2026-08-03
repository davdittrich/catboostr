# catboost-8z4.47 — Integer label matrix routed class-label path, unlike Python

## I. Summary

`R/catboost.R` now coerces a plain (non-factor, non-character) **multi-column**
integer label matrix to double before passing it to `.Call`, so
`src/catboostr.cpp` dispatches `ERawTargetType::Float` instead of
`ERawTargetType::Integer` — matching Python's `Pool(data, label=<int 0/1
ndarray>)` handling for multi-target losses (e.g. `MultiLogloss`). Single-column
integer label vectors (regression/classification/ranking) are untouched, as
required by the ticket's scope boundary.

**Important finding from Step 3 (TDD reproduction):** the numeric-divergence
symptom named in the ticket (`MultiLogloss` predictions diverging from the
Python oracle when fed a raw integer label matrix) **does not reproduce** on
current (pre-fix) code — see §III. The fix implemented here is a genuine,
low-risk semantic/metadata-parity correction (aligning the C++
`ERawTargetType` tag with Python's dtype-independent target semantics), not a
fix for an observed numeric regression. This is reported transparently per
the evidence-before-claims rule rather than manufacturing a false "red" test.

## II. Python dispatch logic (read, `vendor/catboost/catboost/python-package/catboost/_catboost.pyx`, read-only)

Target-type classification (which `ERawTargetType` tag a numpy dtype maps to):

```python
# _catboost.pyx:4281-4290
cdef ERawTargetType _py_target_type_to_raw_target_data(py_label_type) noexcept:
    if py_label_type in (py_bool, np.bool_):
        return ERawTargetType_Boolean
    elif np.issubdtype(py_label_type, np.floating):
        return ERawTargetType_Float
    elif np.issubdtype(py_label_type, np.integer):
        return ERawTargetType_Integer
    else:
        return ERawTargetType_String
```

So Python **also** tags an int64 ndarray as `ERawTargetType_Integer` — dtype
classification itself is identical to R's `Rf_isInteger(targetParam)` check.
The actual multi-target value ingestion path always casts to `float`
regardless of that tag:

```python
# _catboost.pyx:4261-4278
def _set_label_from_num_nparray_objects_order(
    np.ndarray[numpy_num_or_bool_dtype, ndim=2] label,
    Py_ObjectsOrderBuilderVisitor py_builder_visitor
):
    ...
    for target_idx in xrange(target_count):
        for object_idx in xrange(object_count):
            builder_visitor[0].AddTarget(target_idx, object_idx, <float>label[object_idx][target_idx])
```

On the C++ core side (`vendor/catboost/catboost/private/libs/target/target_converter.cpp:584-593`),
the multi-label/multi-target converter used for `MultiLogloss`/`MultiRMSE`
explicitly **ignores** the raw target type:

```cpp
TVector<float> Process(
    ERawTargetType /* targetType */,
    const TRawTarget& rawTarget,
    NPar::ILocalExecutor* localExecutor
) override {
    TVector<float> result = ConvertRawToFloatTarget(rawTarget, localExecutor);
    ...
}
```

So for multi-target losses, neither Python nor CatBoost's core cares whether
the tag is `Integer` or `Float` — both paths always float-cast the values.
`R`'s `src/catboostr.cpp:307-320` sets the metaInfo tag from R storage mode
the same way Python's classifier does; the only place the tag has teeth is
`SetClassLabels` (src/catboostr.cpp:136-152), which is a no-op when
`classLabelsParam` is `R_NilValue` (always true on the plain, non-factor
label path) — so for the multi-target case in this repo, before this fix,
the tag mismatch had no observable effect on training/prediction, only on
metadata/documentation accuracy and on stricter validation
(`CheckContainsOnlyIntegers` in `target.cpp`, trivially satisfied by 0/1
integer data).

## III. Rule implemented (`R/catboost.R`, current lines ~267-276)

```r
if (!is.null(label) && !is.matrix(label))
    label <- as.matrix(label)
# A plain (non-factor, non-character) integer label matrix with more than
# one column is a multi-target numeric label (e.g. MultiLogloss/MultiRMSE),
# never a class-label encoding -- those only ever produce a single column
# via the is.factor() branch above, which sets class_labels and must keep
# its Integer storage mode untouched. Coerce to double so C++ dispatches
# ERawTargetType::Float here, matching Python's Pool(data, label=<int
# ndarray>) target-type semantics (catboost-8z4.47).
if (!is.null(label) && is.null(class_labels) && is.integer(label) && ncol(label) > 1L)
    storage.mode(label) <- "double"
if (!is.double(label) && !is.integer(label) && !is.null(label))
    stop("Unsupported label type, expecting double or int, got: ", typeof(label))
```

Scope: `ncol(label) > 1L` is the guard that limits the coercion to genuine
multi-target matrices, leaving binary classification, multiclass, regression,
ranking, and the `is.factor`/`is.character` paths (which always produce a
single column) completely untouched, per the ticket's explicit "do not
touch" list. `src/catboostr.cpp` was **not** modified — the R-side coercion
alone is sufficient, since it flips which `ERawTargetType` branch
`Rf_isReal`/`Rf_isInteger` selects.

Doc updates in the same commit: `R/catboost.R` `@param label` roxygen block
and `man/catboost.load_pool.Rd` — replaced the earlier "misroutes through the
class-label path" caveat (added doc-only in commit `002e4d5`) with a caveat
describing the new (fixed) auto-coercion behavior.

## IV. Test runs (verbatim)

### Step 2/3 — reproduction attempt on pre-fix code

Ran the exact bug scenario (integer 0/1 `multilogloss_label` matrix, no
`as.double()` workaround, loss `MultiLogloss`) directly against the
**pre-fix**, then-currently-installed `catboostr` build:

```
> label <- fixture$inputs$multilogloss_label
> is.integer(label)
[1] TRUE
> pool <- catboost.load_pool(fixture$inputs$features, label = label)
> model <- catboost.train(pool, params = multitarget_params("MultiLogloss"))
> prediction <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
> max(abs(prediction - fixture$expected$multilogloss_predict))
[1] 0
> identical(typeof(catboost.pool.get_label(pool)), typeof(fixture$expected$multilogloss_get_label))
[1] TRUE
```

`testthat` run of the reproduction as a real test, pre-fix:

```
Test passed with 2 successes 🥳.
```

i.e. test (a) ("integer matrix fails on current code") did **not** fail —
see §I/§II for the C++-level root cause of why not (`ERawTargetType` is
ignored by the multi-label target converter). Test (b) (single-target integer
vector + RMSE trains fine) also already passed pre-fix, as expected (that
path is untouched by the ticket).

### Step 7 — full regression suite, post-fix, post-rebuild

Build:
```
cd /home/dd/Gemini/catboost-phase-3-pool-parity
CATBOOSTR_VENDOR_SRC=/home/dd/Gemini/catboost/vendor/catboost \
CATBOOSTR_CYTHON=/tmp/catboostr-cython-venv/.venv/bin/cython \
R CMD INSTALL --preclean .
...
* DONE (catboostr)
```

Full suite:
```
Rscript -e 'library(catboostr); testthat::test_dir("tests/testthat")'
...
[ FAIL 0 | WARN 0 | SKIP 1 | PASS 279 ]
```

`test_multitarget_differential.R` specifically (post-fix, workaround removed,
new single-column regression guard test added):
```
multitarget_differential:
test_multitarget_differential.R: ..........
══ DONE ════════════════════════════════════════════════════════════════════════
```

### Step 8 — vendor pin cleanliness

```
git -C vendor/catboost status --porcelain
(no output — clean)
```

## V. Files changed

- `R/catboost.R` — multi-column integer label matrix coercion + `@param label` doc update.
- `man/catboost.load_pool.Rd` — matching hand-maintained Rd update (no `Roxygen:` field in `DESCRIPTION`; this package's man pages are hand-maintained, not `roxygen2::document()`-generated).
- `tests/testthat/test_multitarget_differential.R` — removed the now-unnecessary `as.double()` workaround from the `MultiLogloss` fit/predict test; added a scope-guard test confirming single-column integer label vectors keep the pre-existing (untouched) Integer-target regression path.
- `src/catboostr.cpp` — **not modified** (R-side coercion alone is sufficient; confirmed via full rebuild + test run).
- `vendor/catboost/` — untouched (`git -C vendor/catboost status --porcelain` empty).
