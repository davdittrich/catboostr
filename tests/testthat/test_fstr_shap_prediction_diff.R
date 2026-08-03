context("test_fstr_shap_prediction_diff.R")

# P4.4 (catboost-8z4.53) argument-value differential test:
# catboost.get_feature_importance(type = "ShapInteractionValues") and
# catboost.get_feature_importance(type = "PredictionDiff") (R/catboost.R:3130).
#
# Both types route through the model's native fstr entry point
# (src/catboostr.cpp CatBoostCalcRegularFeatureEffect_R) the same way Python's
# CatBoost._calc_fstr() (catboost/python-package/catboost/core.py:3555) routes
# through CalcShapFeatureInteractionMulti / GetFeatureImportances(PredictionDiff, ...)
# -- both wrap the identical C++ engine. This test compares R's numeric output
# to the pinned Python catboost==1.2.10 oracle fixture, within tolerance.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_fstr_shap_prediction_diff_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "fstr_shap_prediction_diff.json"),
  simplifyVector = TRUE
)
inputs <- fixture$inputs
expected <- fixture$expected

TOL <- 1e-6

build_model <- function() {
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label, feature_names = as.list(inputs$feature_names))
  catboost.train(pool, params = list(
    iterations = 10, depth = 2, loss_function = "Logloss",
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))
}

model <- build_model()

shap_pool <- catboost.load_pool(
  data.frame(
    num1 = inputs$num1[seq_len(inputs$shap_interaction_rows)],
    num2 = inputs$num2[seq_len(inputs$shap_interaction_rows)]
  ),
  label = inputs$label[seq_len(inputs$shap_interaction_rows)],
  feature_names = as.list(inputs$feature_names)
)

diff_rows <- inputs$prediction_diff_rows + 1  # Python is 0-indexed
diff_pool <- catboost.load_pool(
  data.frame(num1 = inputs$num1[diff_rows], num2 = inputs$num2[diff_rows]),
  feature_names = as.list(inputs$feature_names)
)

test_that("get_feature_importance: ShapInteractionValues matches Python oracle", {
  result <- catboost.get_feature_importance(model, shap_pool, type = "ShapInteractionValues")

  expect_equal(dim(result), dim(expected$shap_interaction_values))
  expect_equal(
    as.numeric(result),
    as.numeric(expected$shap_interaction_values),
    tolerance = TOL
  )
})

test_that("get_feature_importance: PredictionDiff matches Python oracle", {
  result <- catboost.get_feature_importance(model, diff_pool, type = "PredictionDiff")

  expect_equal(as.numeric(result), as.numeric(expected$prediction_diff), tolerance = TOL)
})

test_that("get_feature_importance: PredictionDiff rejects a pool without exactly 2 rows", {
  expect_error(
    catboost.get_feature_importance(model, shap_pool, type = "PredictionDiff"),
    "must contain exactly 2 rows"
  )
})

test_that("get_feature_importance: ShapInteractionValues requires a pool", {
  expect_error(
    catboost.get_feature_importance(model, type = "ShapInteractionValues"),
    "pool is required"
  )
})

test_that("get_feature_importance: PredictionDiff requires a pool", {
  expect_error(
    catboost.get_feature_importance(model, type = "PredictionDiff"),
    "pool is required"
  )
})
