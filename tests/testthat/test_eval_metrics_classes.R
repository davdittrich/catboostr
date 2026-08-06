context("test_eval_metrics_classes.R")

# catboost-8z4.79 differential test: R's catboost.eval_metrics() method
# parity against all four Python estimator classes -- CatBoost,
# CatBoostClassifier, CatBoostRegressor, CatBoostRanker
# (catboost/python-package/catboost/core.py). eval_metrics() is defined once
# on the CatBoost base class (core.py:3234) and is not overridden by any of
# the three subclasses, so R's single catboost.eval_metrics() function
# (R/catboost.R) is the parity target for all four
# CatBoost{,Classifier,Regressor,Ranker}.eval_metrics matrix rows; what
# varies per class is only the loss/task shape a model is fit with, so this
# fixture pins one fit per class covering a distinct loss family (MultiClass
# via the base class, Logloss via Classifier, RMSE via Regressor, YetiRank
# via Ranker). All four funnel through the same native entry point
# (src/catboostr.cpp's CatBoostEvalMetrics_R -> TMetricsPlotCalcer,
# catboost/private/libs/algo/plot.h), matching the tolerance already used
# for that entry point's other oracle-python parity tests (test_compare.R,
# test_pool_metadata.R): 1e-6.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_eval_metrics_classes_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "eval_metrics_classes.json"),
  simplifyVector = TRUE
)

TOL <- 1e-6

COMMON_PARAMS <- list(iterations = 10, depth = 2, random_seed = 42,
                       thread_count = 1, logging_level = "Silent")

check_class <- function(class_name, loss_function, label, metrics, group_id = NULL) {
  inputs <- fixture$inputs[[class_name]]
  expected <- fixture$expected[[class_name]]
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)

  if (is.null(group_id)) {
    pool <- catboost.load_pool(data, label = label)
  } else {
    pool <- catboost.load_pool(data, label = label, group_id = group_id)
  }
  catboost.pool.set_feature_names(pool, inputs$feature_names)

  model <- catboost.train(pool, params = c(list(loss_function = loss_function), COMMON_PARAMS))

  result <- catboost.eval_metrics(pool = pool, model = model, metrics = as.list(metrics),
                                   ntree_start = 0L, ntree_end = 0L, eval_period = 1)

  for (metric_name in metrics) {
    expect_equal(
      as.numeric(result[[metric_name]]),
      as.numeric(expected[[metric_name]]),
      tolerance = TOL,
      info = paste(class_name, metric_name)
    )
  }
}

test_that("eval_metrics: CatBoost (base class, MultiClass) matches Python oracle", {
  check_class("CatBoost", "MultiClass", fixture$inputs$CatBoost$label,
              fixture$inputs$CatBoost$metrics)
})

test_that("eval_metrics: CatBoostClassifier (Logloss) matches Python oracle", {
  check_class("CatBoostClassifier", "Logloss", fixture$inputs$CatBoostClassifier$label,
              fixture$inputs$CatBoostClassifier$metrics)
})

test_that("eval_metrics: CatBoostRegressor (RMSE) matches Python oracle", {
  check_class("CatBoostRegressor", "RMSE", fixture$inputs$CatBoostRegressor$label,
              fixture$inputs$CatBoostRegressor$metrics)
})

test_that("eval_metrics: CatBoostRanker (YetiRank) matches Python oracle", {
  check_class("CatBoostRanker", "YetiRank", fixture$inputs$CatBoostRanker$label,
              fixture$inputs$CatBoostRanker$metrics, group_id = fixture$inputs$CatBoostRanker$group_id)
})
