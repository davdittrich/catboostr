context("test_compute_training_options.R")

# catboost-azg differential test: catboost.compute_training_options()
# (R/catboost.R), the R equivalent of Python's
# catboost.utils.compute_training_options(). Both resolve a plain params
# dict to its final, fully-resolved training options *without* fitting a
# model -- unlike catboost.get_plain_params()/catboost.get_model_params(),
# which resolve params from an already-fitted model.
#
# Python's compute_training_options (_catboost.pyx:7095) calls the native
# GetTrainingOptions (python-package/catboost/helpers.cpp:205), which is
# just: PlainJsonToOptions -> ConvertParamsToCanonicalFormat -> LoadOptions
# -> SetDataDependentDefaults -> TCatBoostOptions::Save. The R export
# (CatBoostComputeTrainingOptions_R, src/catboostr.cpp) reproduces that exact
# call sequence directly (those calls are already linked into catboostr via
# private-libs-options, used elsewhere in catboostr.cpp's
# TrainModelDistributed) instead of linking the python-package-only
# helpers.cpp/.h, so this is the same native resolution path, not an R-side
# reimplementation.
#
# Since compute_training_options is a pure function of (params,
# train_meta_info, test_meta_info) with no wall-clock/GUID/training-metrics
# output, the whole resolved tree is deterministic and reproducible -- a
# full structural comparison against the pinned Python oracle is the
# appropriate test (no volatile-field carve-outs needed, unlike
# test_metadata.R's model-metadata comparison). Numeric leaves compare with
# tolerance rather than byte-for-byte JSON text: the native JSON writer
# (NJson::TJsonValue) truncates doubles to fewer significant digits than a
# full-precision re-encode uses, so two serializations of the identical
# double (e.g. the float32-widened learning_rate) can legitimately differ
# in trailing digits.
#
# Matrix row: catboost.utils.compute_training_options (tolerance 1e-12,
# loosened to 1e-9 here to absorb the serializer-precision gap above).
#
# Regenerate the Python fixture with:
#   uv run --frozen --project tools/oracle python3 tools/oracle/gen_compute_training_options_fixture.py

# Recursively compare two parsed-JSON trees, ignoring key order (THashMap
# iteration order is not guaranteed) and comparing numeric leaves with
# tolerance rather than byte-for-byte: the native JSON writer
# (NJson::TJsonValue, used on both the R and the Python side -- Python's
# compute_training_options returns the exact same TCatBoostOptions::Save()
# tree) truncates doubles to fewer significant digits than R's/Python's own
# JSON serializers use when re-encoding a parsed value, e.g. the float32
# learning_rate widens to 0.05000000074505806 in a full-precision re-encode
# but the native writer already emitted "0.05000000075" -- same underlying
# value, different serializer precision. This matches the matrix row's own
# declared elementwise/tolerance disposition (1e-12 is the row's nominal
# figure; a looser 1e-9 relative tolerance here absorbs exactly that
# serializer-precision gap without masking a real mismatch).
expect_trees_equal <- function(a, b, path = "") {
  if (is.list(a) || is.list(b)) {
    testthat::expect_true(is.list(a) && is.list(b), info = paste0("at ", path, ": one side is not a list"))
    a_names <- names(a)
    b_names <- names(b)
    if (!is.null(a_names) || !is.null(b_names)) {
      testthat::expect_identical(sort(a_names), sort(b_names), info = paste0("at ", path, ": key sets differ"))
      for (key in a_names) {
        expect_trees_equal(a[[key]], b[[key]], path = paste0(path, "$", key))
      }
    } else {
      testthat::expect_identical(length(a), length(b), info = paste0("at ", path, ": length differs"))
      for (i in seq_along(a)) {
        expect_trees_equal(a[[i]], b[[i]], path = paste0(path, "[", i, "]"))
      }
    }
  } else if (is.numeric(a) && is.numeric(b)) {
    testthat::expect_equal(a, b, tolerance = 1e-9, info = paste0("at ", path))
  } else {
    testthat::expect_identical(a, b, info = paste0("at ", path))
  }
}

# simplifyVector = TRUE for scalar inputs (params/meta_info, all short and
# flat -- convenient as plain R values); simplifyVector = FALSE for the
# expected nested options trees, matching the shape expect_trees_equal's
# raw (also simplifyVector = FALSE) native output is compared against.
fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "compute_training_options.json"),
  simplifyVector = TRUE
)
inputs <- fixture$inputs
expected <- fixture$expected
fixture_unsimplified <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "compute_training_options.json"),
  simplifyVector = FALSE
)
expected_unsimplified <- fixture_unsimplified$expected

test_that("catboost.compute_training_options resolves data-dependent defaults without training, matching the Python oracle", {
  options_with_test_raw <- .Call(
    "CatBoostComputeTrainingOptions_R",
    jsonlite::toJSON(inputs$params, auto_unbox = TRUE, digits = NA),
    jsonlite::toJSON(inputs$train_meta_info, auto_unbox = TRUE, digits = NA),
    jsonlite::toJSON(inputs$test_meta_info, auto_unbox = TRUE, digits = NA)
  )
  expect_trees_equal(
    jsonlite::fromJSON(options_with_test_raw, simplifyVector = FALSE),
    expected_unsimplified$options_with_test
  )
})

test_that("catboost.compute_training_options with no test_meta_info matches the Python oracle", {
  options_train_only_raw <- .Call(
    "CatBoostComputeTrainingOptions_R",
    jsonlite::toJSON(inputs$params, auto_unbox = TRUE, digits = NA),
    jsonlite::toJSON(inputs$train_meta_info, auto_unbox = TRUE, digits = NA),
    NULL
  )
  expect_trees_equal(
    jsonlite::fromJSON(options_train_only_raw, simplifyVector = FALSE),
    expected_unsimplified$options_train_only
  )
})

test_that("catboost.compute_training_options returns a parsed R list via the public wrapper", {
  options <- catboost.compute_training_options(
    params = inputs$params,
    train_meta_info = inputs$train_meta_info,
    test_meta_info = inputs$test_meta_info
  )
  expect_identical(options$boosting_options$iterations, expected$options_with_test$boosting_options$iterations)
  expect_equal(options$boosting_options$learning_rate, expected$options_with_test$boosting_options$learning_rate, tolerance = 1e-9)
  expect_identical(options$metric_options$objective_metric$type, expected$options_with_test$metric_options$objective_metric$type)
})

test_that("catboost.compute_training_options without test_meta_info returns a parsed R list", {
  options <- catboost.compute_training_options(
    params = inputs$params,
    train_meta_info = inputs$train_meta_info
  )
  expect_identical(options$boosting_options$iterations, expected$options_train_only$boosting_options$iterations)
})
