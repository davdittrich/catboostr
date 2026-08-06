context("test_shrink_classes.R")

# catboost-8z4.81 differential test: R's catboost.shrink() method parity
# against all four Python estimator classes -- CatBoost, CatBoostClassifier,
# CatBoostRegressor, CatBoostRanker (catboost/python-package/catboost/
# core.py). shrink() is defined once on the CatBoost base class and is not
# overridden by any of the three subclasses; both languages route to the
# same native truncation entry point (src/catboostr.cpp's
# CatBoostShrinkModel_R vs. the Python _object.Truncate binding), so R's
# single catboost.shrink() function (R/catboost.R:4160) is the parity
# target for all 4 CatBoost{,Classifier,Regressor,Ranker}.shrink matrix
# rows. shrink() has no output of its own (Python returns None, R returns
# the native call's status), so -- same pattern as
# test_drop_unused_features_classes.R -- parity is verified via
# predict(RawFormulaVal) taken before and after the same truncation
# [ntree_start=2, ntree_end=8) on a 10-tree model, reusing the same
# per-class loss families as test_predict_classes.R /
# test_eval_metrics_classes.R: MultiClass base, Logloss classifier, RMSE
# regressor, YetiRank ranker.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_shrink_classes_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "shrink_classes.json"),
  simplifyVector = TRUE
)

TOL <- 1e-6

COMMON_PARAMS <- list(iterations = 10, depth = 2, random_seed = 42,
                       thread_count = 1, logging_level = "Silent")
NTREE_START <- fixture$ntree_start
NTREE_END <- fixture$ntree_end

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

  before <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(as.numeric(before), as.numeric(expected$before), tolerance = TOL,
               info = paste(class_name, "before shrink"))

  status <- catboost.shrink(model, ntree_end = NTREE_END, ntree_start = NTREE_START)
  expect_true(status)

  after <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(as.numeric(after), as.numeric(expected$after), tolerance = TOL,
               info = paste(class_name, "after shrink"))
}

test_that("shrink: CatBoost (base class, MultiClass) matches Python oracle", {
  check_class("CatBoost", "MultiClass")
})

test_that("shrink: CatBoostClassifier (Logloss) matches Python oracle", {
  check_class("CatBoostClassifier", "Logloss")
})

test_that("shrink: CatBoostRegressor (RMSE) matches Python oracle", {
  check_class("CatBoostRegressor", "RMSE")
})

test_that("shrink: CatBoostRanker (YetiRank) matches Python oracle", {
  check_class("CatBoostRanker", "YetiRank", group_id = fixture$inputs$CatBoostRanker$group_id)
})
