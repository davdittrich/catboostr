context("test_get_object_importance_classes.R")

# catboost-8z4.80 differential test: R's catboost.get_object_importance()
# parity against Python's model.get_object_importance() (catboost/
# python-package/catboost/core.py:3603) across the four estimator classes.
# get_object_importance() is defined once on the CatBoost base class and is
# not overridden by any of the three subclasses, so R's single
# catboost.get_object_importance() function (R/catboost.R:4110) is the
# parity target for all four CatBoost{,Classifier,Regressor,Ranker}.
# get_object_importance matrix rows -- but the underlying LeafInfluence
# derivative calculator (catboost/private/libs/documents_importance/
# ders_helpers.cpp:76-95) only implements a fixed allow-list of loss
# functions and raises for everything else, including MultiClass and every
# ranking loss:
#   - CatBoostClassifier (Logloss) and CatBoostRegressor (RMSE) are
#     allow-listed: real numeric LeafInfluence values, checked below against
#     a pinned Python oracle fixture.
#   - CatBoost (MultiClass) is NOT allow-listed: both languages raise the
#     same ders_helpers.cpp:95 error, already covered by
#     test_object_importance_multiclass.R (catboost-8z4.54).
#   - CatBoostRanker (YetiRank) is NOT allow-listed either: same error,
#     checked below directly (new coverage) against the exact message the
#     fixture generator captured from Python's own exception (not inferred
#     from reading the C++ source), same pinning style as
#     test_object_importance_multiclass.R's MultiClass case.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_get_object_importance_classes_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "get_object_importance_classes.json"),
  simplifyVector = TRUE
)

TOL <- 1e-6
EVAL_N <- 5

COMMON_PARAMS <- list(iterations = 10, depth = 2, random_seed = 42,
                       thread_count = 1, logging_level = "Silent")

split_pools <- function(inputs, group_id = NULL) {
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  n <- nrow(data)
  eval_idx <- seq_len(EVAL_N)
  train_idx <- (EVAL_N + 1):n

  if (is.null(group_id)) {
    eval_pool <- catboost.load_pool(data[eval_idx, ], label = inputs$label[eval_idx])
    train_pool <- catboost.load_pool(data[train_idx, ], label = inputs$label[train_idx])
  } else {
    eval_pool <- catboost.load_pool(data[eval_idx, ], label = inputs$label[eval_idx], group_id = group_id[eval_idx])
    train_pool <- catboost.load_pool(data[train_idx, ], label = inputs$label[train_idx], group_id = group_id[train_idx])
  }
  catboost.pool.set_feature_names(eval_pool, inputs$feature_names)
  catboost.pool.set_feature_names(train_pool, inputs$feature_names)
  list(eval_pool = eval_pool, train_pool = train_pool)
}

check_class <- function(class_name, loss_function) {
  inputs <- fixture$inputs[[class_name]]
  expected <- fixture$expected[[class_name]]
  pools <- split_pools(inputs)

  model <- catboost.train(pools$train_pool, params = c(list(loss_function = loss_function), COMMON_PARAMS))
  result <- catboost.get_object_importance(model, pools$eval_pool, pools$train_pool)

  expect_equal(as.numeric(result$indices), as.numeric(expected$indices), tolerance = TOL, info = paste(class_name, "indices"))
  expect_equal(as.numeric(result$scores), as.numeric(expected$scores), tolerance = TOL, info = paste(class_name, "scores"))
}

test_that("get_object_importance: CatBoostClassifier (Logloss) matches Python oracle", {
  check_class("CatBoostClassifier", "Logloss")
})

test_that("get_object_importance: CatBoostRegressor (RMSE) matches Python oracle", {
  check_class("CatBoostRegressor", "RMSE")
})

test_that("get_object_importance: CatBoostRanker (YetiRank) raises the same ders_helpers.cpp:95 error as Python (#869-style)", {
  data <- data.frame(num1 = c(0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25,
                               -10.0, 11.5, 1.5, -2.5, 3.25, 4.0, -5.75, 6.5, -7.25, 8.0))
  label <- c(0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0)
  group_id <- rep(1:5, each = 4)
  pool <- catboost.load_pool(data, label = label, group_id = group_id)
  model <- catboost.train(pool, params = c(list(loss_function = "YetiRank"), COMMON_PARAMS))

  expect_error(
    catboost.get_object_importance(model, pool, pool),
    regexp = fixture$expected$CatBoostRanker$error,
    fixed = TRUE
  )
})
