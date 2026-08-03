context("test_calc_feature_statistics.R")

# P4.1 (catboost-8z4.50) structural differential test: catboost.calc_feature_statistics
# (R/catboost.R) wraps the same vendor C++ entry point (NCB::GetBinarizedStatistics,
# catboost/private/libs/quantized_pool_analysis) the pinned Python catboost==1.2.10
# CatBoostClassifier.calc_feature_statistics() uses internally (_catboost.pyx
# _get_binarized_statistics), so per spec Sec 4.3's "Structural" row this compares
# canonical-JSON field-by-field (borders/binarized_feature/mean_target/
# mean_weighted_target/mean_prediction/objects_per_bin/predictions_on_varying_feature/
# cat_values) against the recorded oracle fixture, numeric leaves within tolerance.
# No plot/rendering comparison (plot=FALSE throughout, matching the fixture) --
# only a smoke test that the return value is the expected list class.
#
# R's catboost.load_pool()/from_matrix() pre-hash categorical columns into floats
# client-side (CatBoostHashStrings_R) before the data ever reaches the native
# TDataProvider, unlike Python's Pool (which retains the original strings
# server-side) -- so NCB::GetCatFeatureValues has nothing to recover from an
# R-built Pool's categorical column, and catboost.calc_feature_statistics()'s
# default (no cat_feature_values override) cat_values path is empty for such a
# Pool (a documented R-Pool-architecture limitation, not a P4.1 bug -- fixing it
# would mean reworking Pool construction, out of this ticket's boundary). This
# test supplies cat_feature_values explicitly for the categorical feature, the
# same escape hatch the Python docstring itself documents, to exercise the
# real vendor computation end-to-end and compare against the same oracle
# fixture Python's own default (string-retaining) Pool produced.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_calc_feature_statistics_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "calc_feature_statistics.json"),
  simplifyVector = TRUE
)
inputs <- fixture$inputs
expected <- fixture$expected

build_pool_and_model <- function() {
  data <- data.frame(
    num1 = inputs$num1,
    num2 = inputs$num2,
    cat1 = factor(inputs$cat1),
    stringsAsFactors = FALSE
  )
  pool <- catboost.load_pool(data, label = inputs$label)
  catboost.pool.set_feature_names(pool, inputs$feature_names)

  model <- catboost.train(pool, params = list(
    iterations = 10, depth = 3, loss_function = "Logloss",
    random_seed = 42, thread_count = 1, logging_level = "Silent",
    one_hot_max_size = 4
  ))
  list(pool = pool, model = model)
}

fixture_env <- build_pool_and_model()
pool <- fixture_env$pool
model <- fixture_env$model

TOL <- 1e-6

expect_stat_equal <- function(actual, expected_stat) {
  for (field in names(expected_stat)) {
    if (field == "cat_values") {
      expect_equal(actual[[field]], expected_stat[[field]])
    } else {
      expect_equal(as.numeric(actual[[field]]), as.numeric(expected_stat[[field]]), tolerance = TOL)
    }
  }
}

test_that("calc_feature_statistics: feature = NULL (all features) matches Python oracle", {
  result <- catboost.calc_feature_statistics(
    model, pool,
    cat_feature_values = list(cat1 = inputs$cat1)
  )
  expect_type(result, "list")
  expect_setequal(names(result), inputs$feature_names)
  for (fname in inputs$feature_names) {
    expect_stat_equal(result[[fname]], expected$all_features_default[[fname]])
  }
})

test_that("calc_feature_statistics: single float feature + explicit prediction_type matches Python oracle", {
  result <- catboost.calc_feature_statistics(
    model, pool,
    feature = "num1", prediction_type = "RawFormulaVal"
  )
  expect_type(result, "list")
  expect_true(is.null(result$cat_values))
  expect_stat_equal(result, expected$single_float)
})

test_that("calc_feature_statistics: single categorical feature (one-hot) matches Python oracle", {
  result <- catboost.calc_feature_statistics(
    model, pool,
    feature = "cat1", cat_feature_values = inputs$cat1
  )
  expect_type(result, "list")
  expect_true(is.null(result$borders))
  expect_stat_equal(result, expected$single_cat)
})

test_that("calc_feature_statistics: numeric feature index + plain-vector cat_feature_values matches Python oracle", {
  # Regression test (task reviewer finding on commit bef3b7d): `feature` given
  # as a 0-based numeric index (not a name) resolves internally to the pool's
  # actual feature name ("cat1"); a plain (non-list) `cat_feature_values`
  # vector must still be picked up under that resolved name, not silently
  # dropped because it was keyed by the raw index ("2") instead.
  cat1_idx0 <- match("cat1", inputs$feature_names) - 1L
  result <- catboost.calc_feature_statistics(
    model, pool,
    feature = cat1_idx0, cat_feature_values = inputs$cat1
  )
  expect_type(result, "list")
  expect_true(is.null(result$borders))
  expect_stat_equal(result, expected$single_cat)
})

test_that("calc_feature_statistics: rejects an unknown prediction_type", {
  expect_error(
    catboost.calc_feature_statistics(model, pool, feature = "num1", prediction_type = "Bogus"),
    "Unknown prediction type"
  )
})

test_that("calc_feature_statistics: rejects an unknown feature name", {
  expect_error(
    catboost.calc_feature_statistics(model, pool, feature = "does_not_exist"),
    "No feature named"
  )
})
