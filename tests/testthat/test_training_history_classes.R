context("test_training_history_classes.R")

# catboost-8z4.118 (P10.D) differential test: R's
# catboost.get_best_iteration()/catboost.get_best_score()/
# catboost.get_evals_result() functions and the classes_/best_iteration_/
# best_score_/evals_result_ fields create.model.base() attaches to every
# model object, against Python catboost==1.2.10's
# CatBoost.best_iteration_/best_score_/evals_result_/classes_ properties and
# get_best_iteration()/get_best_score()/get_evals_result() methods
# (catboost/python-package/catboost/core.py:1851-2098). All three get_*/*_
# pairs are defined once on the CatBoost base class and are not overridden
# by any of the three subclasses, so this single R implementation is the
# parity target for all 28 matrix rows (7 families x CatBoost/
# CatBoostClassifier/CatBoostRegressor/CatBoostRanker).
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_training_history_classes_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "training_history_classes.json"),
  simplifyVector = TRUE
)

TOL <- 1e-6

fit_class <- function(class_name, loss_function, group_id = NULL, eval_group_id = NULL) {
  inputs <- fixture$inputs[[class_name]]
  learn_data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  eval_data <- data.frame(num1 = inputs$eval_num1, num2 = inputs$eval_num2)
  learn_pool <- catboost.load_pool(learn_data, label = inputs$label, group_id = group_id)
  eval_pool <- catboost.load_pool(eval_data, label = inputs$eval_label, group_id = eval_group_id)
  catboost.pool.set_feature_names(learn_pool, inputs$feature_names)
  catboost.pool.set_feature_names(eval_pool, inputs$feature_names)
  catboost.train(learn_pool, eval_pool, params = list(
    loss_function = loss_function, iterations = 30, depth = 2, random_seed = 42,
    thread_count = 1, logging_level = "Silent", early_stopping_rounds = 5
  ))
}

# names(list()) is NULL but names(<empty named list from jsonlite>) is
# character(0) -- both mean "no keys", so normalize before comparing (an
# empty best_score_/evals_result_ is itself a real, oracle-verified case:
# see the CatBoostRanker fixture, where LearnBestError is empty).
sorted_names <- function(x) {
  n <- names(x)
  if (is.null(n)) character(0) else sort(n)
}

check_best_score <- function(actual, expected) {
  expect_equal(sorted_names(actual), sorted_names(expected))
  for (set_name in names(expected)) {
    expect_equal(sorted_names(actual[[set_name]]), sorted_names(expected[[set_name]]))
    for (metric_name in names(expected[[set_name]])) {
      expect_equal(actual[[set_name]][[metric_name]], expected[[set_name]][[metric_name]], tolerance = TOL)
    }
  }
}

check_evals_result <- function(actual, expected) {
  expect_equal(sorted_names(actual), sorted_names(expected))
  for (set_name in names(expected)) {
    expect_equal(sorted_names(actual[[set_name]]), sorted_names(expected[[set_name]]))
    for (metric_name in names(expected[[set_name]])) {
      expect_equal(as.numeric(actual[[set_name]][[metric_name]]),
                   as.numeric(expected[[set_name]][[metric_name]]), tolerance = TOL)
    }
  }
}

check_class <- function(class_name, loss_function, group_id = NULL, eval_group_id = NULL) {
  model <- fit_class(class_name, loss_function, group_id, eval_group_id)
  expected <- fixture$expected[[class_name]]

  # classes_ field.
  expect_equal(as.numeric(model$classes_), as.numeric(expected$classes_), tolerance = TOL)

  # best_iteration_ field / catboost.get_best_iteration().
  expect_equal(model$best_iteration_, as.integer(expected$best_iteration_))
  expect_equal(catboost.get_best_iteration(model), as.integer(expected$get_best_iteration))

  # best_score_ field / catboost.get_best_score().
  check_best_score(model$best_score_, expected$best_score_)
  check_best_score(catboost.get_best_score(model), expected$get_best_score)

  # evals_result_ field / catboost.get_evals_result().
  check_evals_result(model$evals_result_, expected$evals_result_)
  check_evals_result(catboost.get_evals_result(model), expected$get_evals_result)
}

test_that("training-history introspection matches Python oracle (CatBoost, MultiClass)", {
  check_class("CatBoost", "MultiClass")
})

test_that("training-history introspection matches Python oracle (CatBoostClassifier, Logloss)", {
  check_class("CatBoostClassifier", "Logloss")
})

test_that("training-history introspection matches Python oracle (CatBoostRegressor, RMSE)", {
  check_class("CatBoostRegressor", "RMSE")
})

test_that("training-history introspection matches Python oracle (CatBoostRanker, YetiRank)", {
  inputs <- fixture$inputs$CatBoostRanker
  check_class("CatBoostRanker", "YetiRank", group_id = inputs$group_id, eval_group_id = inputs$eval_group_id)
})

test_that("classes_ is empty for a non-classification model (CatBoostRegressor)", {
  model <- fit_class("CatBoostRegressor", "RMSE")
  expect_equal(length(model$classes_), 0)
})

test_that("training-history introspection matches Python oracle without an eval set (BestIteration/TestMetricsHistory undefined, learn history still recorded)", {
  inputs <- fixture$inputs$CatBoostRegressorNoEvalSet
  expected <- fixture$expected$CatBoostRegressorNoEvalSet
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label)
  catboost.pool.set_feature_names(pool, inputs$feature_names)
  model <- catboost.train(pool, params = list(
    loss_function = "RMSE", iterations = 30, depth = 2, random_seed = 42,
    thread_count = 1, logging_level = "Silent"
  ))

  expect_null(model$best_iteration_)
  expect_null(catboost.get_best_iteration(model))
  expect_null(expected$best_iteration_)

  check_best_score(model$best_score_, expected$best_score_)
  check_best_score(catboost.get_best_score(model), expected$get_best_score)

  check_evals_result(model$evals_result_, expected$evals_result_)
  check_evals_result(catboost.get_evals_result(model), expected$get_evals_result)
})
