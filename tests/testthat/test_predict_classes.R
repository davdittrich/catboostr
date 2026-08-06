context("test_predict_classes.R")

# catboost-8z4.81 differential test: R's catboost.predict() method parity
# against all four Python estimator classes -- CatBoost, CatBoostClassifier,
# CatBoostRegressor, CatBoostRanker (catboost/python-package/catboost/
# core.py), with prediction_type="RawFormulaVal" EXPLICITLY requested on
# both sides. predict() IS overridden per-subclass with different DEFAULT
# prediction_type values -- CatBoostClassifier.predict() defaults to
# prediction_type='Class' (core.py:5552), CatBoostRegressor.predict()
# resolves its default via _get_default_prediction_type() (core.py:6183,
# 6320-6329) to 'Exponent'/'RMSEWithUncertainty' for some losses, and
# CatBoostRanker.predict() hard-codes 'RawFormulaVal' and drops the kwarg
# entirely -- but all four produce identical RawFormulaVal output when that
# prediction_type is explicitly requested on both sides, which is exactly
# what this test does. Default-prediction_type parity itself (R's
# catboost.predict() always defaults to RawFormulaVal regardless of model
# type) is NOT covered here and is out of scope for this ticket
# (catboost-8z4.81); see the follow-up ticket filed for that gap. R's single
# catboost.predict() function (R/catboost.R:3792, delegates to
# predict.catboost.Model) is therefore the parity target for all 4
# CatBoost{,Classifier,Regressor,Ranker}.predict matrix rows; what varies
# per row is only the loss/task shape the model was fit with, so this
# fixture reuses the same per-class loss families as
# test_eval_metrics_classes.R / test_get_feature_importance_classes.R:
# MultiClass base, Logloss classifier, RMSE regressor, YetiRank ranker.
# Existing predict coverage (test_model.R, test_multitarget_differential.R)
# checks self-consistency or multi-target losses only, never a per-class
# RawFormulaVal value against a Python oracle for these four loss
# families -- this file closes that gap.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_predict_classes_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "predict_classes.json"),
  simplifyVector = TRUE
)

TOL <- 1e-6

COMMON_PARAMS <- list(iterations = 10, depth = 2, random_seed = 42,
                       thread_count = 1, logging_level = "Silent")

check_class <- function(class_name, loss_function, group_id = NULL) {
  inputs <- fixture$inputs[[class_name]]
  expected <- fixture$expected[[class_name]]
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)

  if (is.null(group_id)) {
    pool <- catboost.load_pool(data, label = inputs$label)
  } else {
    pool <- catboost.load_pool(data, label = inputs$label, group_id = group_id)
  }
  catboost.pool.set_feature_names(pool, inputs$feature_names)

  model <- catboost.train(pool, params = c(list(loss_function = loss_function), COMMON_PARAMS))

  result <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")

  expect_equal(as.numeric(result), as.numeric(expected), tolerance = TOL, info = class_name)
}

test_that("predict: CatBoost (base class, MultiClass) matches Python oracle", {
  check_class("CatBoost", "MultiClass")
})

test_that("predict: CatBoostClassifier (Logloss) matches Python oracle", {
  check_class("CatBoostClassifier", "Logloss")
})

test_that("predict: CatBoostRegressor (RMSE) matches Python oracle", {
  check_class("CatBoostRegressor", "RMSE")
})

test_that("predict: CatBoostRanker (YetiRank) matches Python oracle", {
  check_class("CatBoostRanker", "YetiRank", group_id = fixture$inputs$CatBoostRanker$group_id)
})
