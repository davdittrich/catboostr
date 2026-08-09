context("test_advanced_eval_plotting_probability.R")

# catboost-8z4.120 (P10.F) differential test: R's
# catboost.create_metric_calcer()/catboost.get_test_eval()/
# catboost.get_test_evals()/catboost.plot_predictions()/
# catboost.plot_partial_dependence() (each x4 estimator classes) plus the
# CatBoostClassifier-only catboost.predict_log_proba()/
# catboost.staged_predict_log_proba()/catboost.get_probability_threshold()/
# catboost.set_probability_threshold(), against Python catboost==1.2.10's
# CatBoost.create_metric_calcer()/get_test_eval(s)()/plot_predictions()/
# plot_partial_dependence() and CatBoostClassifier.predict_log_proba()/
# staged_predict_log_proba()/get_probability_threshold()/
# set_probability_threshold() (catboost/python-package/catboost/core.py:
# 1830-1850, 3354-3374, 3847-4020, 5655-5914).
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_advanced_eval_plotting_probability_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "advanced_eval_plotting_probability.json"),
  simplifyVector = TRUE
)

TOL <- 1e-6

fit_class <- function(class_name, loss_function, group_id = NULL, eval_group_id = NULL) {
  inputs <- fixture$inputs[[class_name]]
  train_data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  train_pool <- catboost.load_pool(train_data, label = inputs$label, group_id = group_id)
  catboost.pool.set_feature_names(train_pool, inputs$feature_names)

  eval_pool <- NULL
  if (!is.null(inputs$eval_num1)) {
    eval_data <- data.frame(num1 = inputs$eval_num1, num2 = inputs$eval_num2)
    eval_pool <- catboost.load_pool(eval_data, label = inputs$eval_label, group_id = eval_group_id)
    catboost.pool.set_feature_names(eval_pool, inputs$feature_names)
  }

  # use_best_model = FALSE: matches the oracle fixture (see
  # gen_advanced_eval_plotting_probability_fixture.py's COMMON) -- otherwise
  # CatBoost's default (True whenever an eval set is given) truncates the
  # model to best_iteration_+1 trees.
  model <- catboost.train(train_pool, eval_pool,
                          params = list(loss_function = loss_function, iterations = 10, depth = 2,
                                        random_seed = 42, thread_count = 1, logging_level = "Silent",
                                        use_best_model = FALSE))
  list(model = model, train_pool = train_pool, eval_pool = eval_pool)
}

check_test_eval <- function(actual, expected) {
  # `expected` came from a JSON list of per-object rows (each a list of
  # per-class values) -- unlist() flattens it row-major. `actual` is an R
  # vector (single-dimension models) or matrix (multiclass, rows = objects,
  # columns = classes); as.matrix()+t() flattens it row-major to match.
  expect_equal(as.numeric(t(as.matrix(actual))), as.numeric(unlist(expected)), tolerance = TOL)
}

check_metric_calcer <- function(actual, expected) {
  expect_equal(sort(names(actual)), sort(names(expected)))
  for (metric_name in names(expected)) {
    expect_equal(as.numeric(actual[[metric_name]]), as.numeric(expected[[metric_name]]), tolerance = TOL)
  }
}

check_plot_predictions <- function(actual, expected) {
  expect_equal(length(actual), NROW(expected))
  for (i in seq_len(NROW(expected))) {
    for (key in names(expected[i, ])) {
      a <- actual[[i]][[key]]
      e <- expected[i, key][[1]]
      expect_equal(as.numeric(unlist(a)), as.numeric(unlist(e)), tolerance = TOL)
    }
  }
}

check_class <- function(class_name, loss_function, group_id = NULL, eval_group_id = NULL,
                        n_metric_chunks = 2) {
  fitted <- fit_class(class_name, loss_function, group_id, eval_group_id)
  model <- fitted$model
  expected <- fixture$expected[[class_name]]
  metrics <- as.list(fixture$inputs[[class_name]]$metrics)

  # catboost.get_test_eval() / catboost.get_test_evals(). Python's
  # get_test_evals() returns a length-1 list wrapping the same values
  # get_test_eval() returns (single eval set); jsonlite's simplifyVector
  # collapses that length-1 wrapping away on the fixture side, so
  # `expected$get_test_evals` is directly comparable (not double-indexed).
  check_test_eval(catboost.get_test_eval(model, fitted$eval_pool), expected$get_test_eval)
  test_evals <- catboost.get_test_evals(model, fitted$eval_pool)
  expect_equal(length(test_evals), 1L)
  check_test_eval(test_evals[[1]], expected$get_test_evals)

  # catboost.create_metric_calcer()
  calcer <- catboost.create_metric_calcer(model, metrics)
  if (n_metric_chunks == 1) {
    calcer$add(fitted$train_pool)
  } else {
    n <- catboost.pool.num_row(fitted$train_pool)
    half <- n %/% 2
    calcer$add(catboost.pool.slice(fitted$train_pool, 0, half))
    calcer$add(catboost.pool.slice(fitted$train_pool, half, n - half))
  }
  check_metric_calcer(calcer$eval_metrics(), expected$create_metric_calcer)

  # catboost.plot_predictions()
  predict_type <- fixture$inputs[[class_name]]$predict_type
  actual_predictions <- catboost.plot_predictions(model, fitted$train_pool, features_to_change = list(0L),
                                                   prediction_type = predict_type)
  check_plot_predictions(actual_predictions, expected$plot_predictions)
}

test_that("advanced eval/plotting matches Python oracle (CatBoost, MultiClass)", {
  check_class("CatBoost", "MultiClass")
})

test_that("advanced eval/plotting matches Python oracle (CatBoostClassifier, Logloss)", {
  check_class("CatBoostClassifier", "Logloss")
})

test_that("advanced eval/plotting matches Python oracle (CatBoostRegressor, RMSE)", {
  check_class("CatBoostRegressor", "RMSE")
})

test_that("advanced eval/plotting matches Python oracle (CatBoostRanker, YetiRank)", {
  inputs <- fixture$inputs$CatBoostRanker
  check_class("CatBoostRanker", "YetiRank", group_id = inputs$group_id,
             eval_group_id = inputs$eval_group_id, n_metric_chunks = 1)
})

test_that("plot_partial_dependence matches Python oracle (CatBoost base, Logloss)", {
  inputs <- fixture$inputs$CatBoostPartialDependence
  train_data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  train_pool <- catboost.load_pool(train_data, label = inputs$label)
  catboost.pool.set_feature_names(train_pool, inputs$feature_names)
  model <- catboost.train(train_pool, params = list(loss_function = "Logloss", iterations = 10, depth = 2,
                                                    random_seed = 42, thread_count = 1, logging_level = "Silent"))
  actual <- catboost.plot_partial_dependence(model, train_pool, features = 0L)
  expect_equal(as.numeric(actual), as.numeric(unlist(fixture$expected$CatBoostPartialDependence$plot_partial_dependence)),
              tolerance = TOL)
})

test_that("plot_partial_dependence matches Python oracle (CatBoostClassifier, Logloss)", {
  fitted <- fit_class("CatBoostClassifier", "Logloss")
  actual <- catboost.plot_partial_dependence(fitted$model, fitted$train_pool, features = 0L)
  expect_equal(as.numeric(actual), as.numeric(unlist(fixture$expected$CatBoostClassifier$plot_partial_dependence)),
              tolerance = TOL)
})

test_that("plot_partial_dependence matches Python oracle (CatBoostRegressor, RMSE)", {
  fitted <- fit_class("CatBoostRegressor", "RMSE")
  actual <- catboost.plot_partial_dependence(fitted$model, fitted$train_pool, features = 0L)
  expect_equal(as.numeric(actual), as.numeric(unlist(fixture$expected$CatBoostRegressor$plot_partial_dependence)),
              tolerance = TOL)
})

test_that("plot_partial_dependence matches Python oracle (CatBoostRanker, YetiRank)", {
  inputs <- fixture$inputs$CatBoostRanker
  fitted <- fit_class("CatBoostRanker", "YetiRank", group_id = inputs$group_id, eval_group_id = inputs$eval_group_id)
  actual <- catboost.plot_partial_dependence(fitted$model, fitted$train_pool, features = 0L)
  expect_equal(as.numeric(actual), as.numeric(unlist(fixture$expected$CatBoostRanker$plot_partial_dependence)),
              tolerance = TOL)
})

test_that("predict_log_proba/staged_predict_log_proba/probability_threshold match Python oracle (CatBoostClassifier)", {
  fitted <- fit_class("CatBoostClassifier", "Logloss")
  model <- fitted$model
  expected <- fixture$expected$CatBoostClassifier

  actual_log_proba <- catboost.predict_log_proba(model, fitted$train_pool)
  expect_equal(as.numeric(actual_log_proba), as.numeric(unlist(expected$predict_log_proba)), tolerance = TOL)

  # jsonlite simplifies a list of same-shaped 2D batches into a 3D array
  # (batch, object, class); slice along the first dimension per batch.
  expected_staged <- expected$staged_predict_log_proba
  it <- catboost.staged_predict_log_proba(model, fitted$train_pool, eval_period = 3)
  for (i in seq_len(dim(expected_staged)[1])) {
    batch <- it$nextElem()
    expect_equal(as.numeric(t(as.matrix(batch))), as.numeric(t(expected_staged[i, , ])), tolerance = TOL)
  }

  expect_equal(catboost.get_probability_threshold(model), expected$get_probability_threshold_default, tolerance = TOL)
  catboost.set_probability_threshold(model, 0.3)
  expect_equal(catboost.get_probability_threshold(model), expected$get_probability_threshold_after_set, tolerance = TOL)
  catboost.set_probability_threshold(model, NULL)
  expect_equal(catboost.get_probability_threshold(model), 0.5, tolerance = TOL)
})

test_that("catboost.set_probability_threshold validates its argument", {
  fitted <- fit_class("CatBoostClassifier", "Logloss")
  expect_error(catboost.set_probability_threshold(fitted$model, "not a number"))
  expect_error(catboost.set_probability_threshold(fitted$model, 1.5))
  expect_error(catboost.set_probability_threshold(fitted$model, -0.1))
})
