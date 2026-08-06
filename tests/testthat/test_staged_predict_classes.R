context("test_staged_predict_classes.R")

# catboost-8z4.81 differential test: R's catboost.staged_predict() (default
# prediction_type="RawFormulaVal") method parity against all four Python
# estimator classes -- CatBoost, CatBoostClassifier, CatBoostRegressor,
# CatBoostRanker (catboost/python-package/catboost/core.py).
# staged_predict() is defined once on the CatBoost base class
# (core.py:2328) and is not overridden in a way that changes its numeric
# output by any of the three subclasses -- CatBoostRanker.staged_predict()
# does override it, but only to hard-code prediction_type="RawFormulaVal"
# and drop the kwarg entirely (same override pattern as
# CatBoostRanker.predict(), see test_predict_classes.R); the values compared
# here are RawFormulaVal on both sides regardless. R's single
# catboost.staged_predict() function (R/catboost.R:3822) is therefore the
# parity target for all 4 CatBoost{,Classifier,Regressor,Ranker}.
# staged_predict matrix rows; what varies per row is only the loss/task
# shape the model was fit with, so this fixture reuses the same per-class
# loss families as test_predict_classes.R / test_eval_metrics_classes.R:
# MultiClass base, Logloss classifier, RMSE regressor, YetiRank ranker.
# 10 trees, eval_period=3 -> 4 stages (cumulative trees [0:3), [0:6), [0:9),
# [0:10)); each stage's raw approx accumulates across calls
# (R/catboost.R:3843 `approx <<- approx + current_approx`), matching
# Python's cumulative-through-ntree_end staged semantics.
# Existing coverage (test_model.R:409-425) only checks R's
# catboost.staged_predict self-consistency against catboost.predict at two
# checkpoints, never against a Python oracle -- this file closes that gap.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_staged_predict_classes_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "staged_predict_classes.json"),
  simplifyVector = FALSE
)

TOL <- 1e-6

COMMON_PARAMS <- list(iterations = 10, depth = 2, random_seed = 42,
                       thread_count = 1, logging_level = "Silent")
EVAL_PERIOD <- fixture$eval_period

check_class <- function(class_name, loss_function, group_id = NULL) {
  inputs <- fixture$inputs[[class_name]]
  expected_stages <- fixture$expected[[class_name]]
  num1 <- unlist(inputs$num1)
  num2 <- unlist(inputs$num2)
  label <- unlist(inputs$label)
  data <- data.frame(num1 = num1, num2 = num2)

  if (is.null(group_id)) {
    pool <- catboost.load_pool(data, label = label)
  } else {
    pool <- catboost.load_pool(data, label = label, group_id = unlist(group_id))
  }
  catboost.pool.set_feature_names(pool, unlist(inputs$feature_names))

  model <- catboost.train(pool, params = c(list(loss_function = loss_function), COMMON_PARAMS))

  staged <- catboost.staged_predict(model, pool, ntree_start = 0, ntree_end = 0,
                                     eval_period = EVAL_PERIOD, prediction_type = "RawFormulaVal")

  for (i in seq_along(expected_stages)) {
    actual <- staged$nextElem()
    # actual is either a plain vector (single-column loss) or a
    # nrow x ncol matrix built byrow=TRUE (R/catboost.R:3843-3846) --
    # i.e. actual[r, c] is row r's c-th raw value, matching the JSON
    # fixture's row-major nested-list layout (expected_stages[[i]][[r]][[c]]).
    # Reconstruct expected as the same nrow x ncol matrix (rbind of
    # per-row numeric vectors) and compare both sides via a row-major
    # flatten (as.numeric(t(.))) so single- and multi-column stages
    # align element-for-element regardless of R's column-major storage.
    expected_mat <- do.call(rbind, lapply(expected_stages[[i]], function(r) as.numeric(unlist(r))))
    actual_mat <- as.matrix(actual)
    expect_equal(as.numeric(t(actual_mat)), as.numeric(t(expected_mat)), tolerance = TOL,
                 info = paste(class_name, "stage", i))
  }
  expect_error(staged$nextElem(), "StopIteration")
}

test_that("staged_predict: CatBoost (base class, MultiClass) matches Python oracle", {
  check_class("CatBoost", "MultiClass")
})

test_that("staged_predict: CatBoostClassifier (Logloss) matches Python oracle", {
  check_class("CatBoostClassifier", "Logloss")
})

test_that("staged_predict: CatBoostRegressor (RMSE) matches Python oracle", {
  check_class("CatBoostRegressor", "RMSE")
})

test_that("staged_predict: CatBoostRanker (YetiRank) matches Python oracle", {
  check_class("CatBoostRanker", "YetiRank", group_id = fixture$inputs$CatBoostRanker$group_id)
})
