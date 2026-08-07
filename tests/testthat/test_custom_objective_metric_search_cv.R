context("test_custom_objective_metric_search_cv.R")

# P6.5 (catboost-8z4.94): extends catboost-8z4.89/.90's custom_objective /
# custom_eval_metric_object bridge (already wired into catboost.train(), see
# test_custom_objective.R / test_custom_eval_metric.R) to catboost.cv(),
# catboost.grid_search(), catboost.randomized_search(),
# catboost.select_features() (metric only -- NCB::SelectFeatures has no
# TCustomObjectiveDescriptor parameter, verified against
# select_features.h/recursive_features_elimination.*) and
# catboost.eval_feature() (both). Reuses the same
# BuildCustomObjectiveDescriptor/BuildCustomMetricDescriptor trampolines
# CatBoostFit_R already calls (src/r_custom_objective.h, src/r_custom_metric.h)
# -- no new bridge mechanism.

set.seed(20260807)
n <- 300
features <- data.frame(
  x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n), x5 = rnorm(n)
)
label <- with(features, 2 * x1 - 1.5 * x2 + 0.5 * x3 + rnorm(n, sd = 0.3))

# Mirrors TRMSEError exactly (error_functions.h:379-403), same as
# test_custom_objective.R's rmse_custom_objective.
rmse_custom_objective <- list(
  calc_ders_range = function(approx, target, weight) {
    w <- if (is.null(weight)) 1 else weight
    cbind(w * (target - approx), w * (-1))
  }
)

# Mirrors TRMSEMetric exactly (metric.cpp:716-748), same as
# test_custom_eval_metric.R's rmse_custom_eval_metric.
rmse_custom_eval_metric <- list(
  evaluate = function(approx, target, weight) {
    w <- if (is.null(weight)) rep(1, length(target)) else weight
    diff <- approx[, 1] - target
    list(error = sum(w * diff^2), weight = sum(w))
  },
  is_max_optimal = function() FALSE,
  get_final_error = function(error) sqrt(error[1] / (error[2] + 1e-38))
)

# boost_from_average = FALSE: a user-defined loss_function is rejected
# outright if boost_from_average is TRUE (catboost_options.cpp:705-709, not
# in the built-in-loss allowlist) -- same reason test_custom_objective.R pins
# it, needed here so the built-in-vs-custom comparison isn't confounded by a
# differing starting bias, and so both runs accept the same params value.
common_cv_params <- function() {
  list(
    iterations = 20,
    learning_rate = 0.1,
    depth = 4,
    random_seed = 1,
    logging_level = "Silent",
    boost_from_average = FALSE
  )
}

# --- (a) catboost.cv() with a custom objective vs a built-in-loss run -------

test_that("catboost.cv: custom_objective reimplementing RMSE matches built-in RMSE cv_results", {
  pool <- catboost.load_pool(features, label = label)

  cv_builtin <- catboost.cv(
    pool, params = c(common_cv_params(), list(loss_function = "RMSE")),
    fold_count = 3, partition_random_seed = 0, shuffle = FALSE
  )
  cv_custom <- catboost.cv(
    pool, params = c(common_cv_params(), list(eval_metric = "RMSE")),
    fold_count = 3, partition_random_seed = 0, shuffle = FALSE,
    custom_objective = rmse_custom_objective
  )

  expect_equal(cv_custom[["test.RMSE.mean"]], cv_builtin[["test.RMSE.mean"]], tolerance = 1e-6)
  expect_equal(cv_custom[["test.RMSE.std"]], cv_builtin[["test.RMSE.std"]], tolerance = 1e-6)
})

test_that("catboost.cv: omitting custom_objective/custom_eval_metric_object reproduces prior output", {
  pool <- catboost.load_pool(features, label = label)
  result <- catboost.cv(
    pool, params = c(common_cv_params(), list(loss_function = "RMSE")),
    fold_count = 3, partition_random_seed = 0
  )
  expect_true(is.data.frame(result))
  expect_true("test.RMSE.mean" %in% names(result))
  expect_equal(nrow(result), common_cv_params()$iterations)
})

# Column-name-agnostic accessor: a bare custom_eval_metric_object (no custom
# objective) forces eval_metric to the "PythonUserDefinedPerObject" marker
# (apply_custom_eval_metric_params, R/catboost.R), so cv()'s reported column
# is named after that marker, not "RMSE" -- unlike the custom_objective case
# above, where eval_metric = "RMSE" was set explicitly and stays literal.
cv_test_metric_col <- function(df, suffix) {
  cols <- grep(paste0("^test\\..*\\.", suffix, "$"), names(df), value = TRUE)
  df[[cols[1]]]
}

test_that("catboost.cv: custom_eval_metric_object (no custom objective) reimplementing RMSE matches built-in RMSE cv_results", {
  pool <- catboost.load_pool(features, label = label)

  cv_builtin <- catboost.cv(
    pool, params = c(common_cv_params(), list(loss_function = "RMSE")),
    fold_count = 3, partition_random_seed = 0, shuffle = FALSE
  )
  cv_custom <- catboost.cv(
    pool, params = c(common_cv_params(), list(loss_function = "RMSE")),
    fold_count = 3, partition_random_seed = 0, shuffle = FALSE,
    custom_eval_metric_object = rmse_custom_eval_metric
  )

  expect_equal(cv_test_metric_col(cv_custom, "mean"), cv_test_metric_col(cv_builtin, "mean"), tolerance = 1e-6)
  expect_equal(cv_test_metric_col(cv_custom, "std"), cv_test_metric_col(cv_builtin, "std"), tolerance = 1e-6)
})

# --- (e) error-propagation: R closure throws mid-training (catboost.cv) ----
# catboost.train's side is covered by test_custom_objective.R/
# test_custom_eval_metric.R's own "a closure that throws ... surfaces as a
# catchable R error" tests (catboost-8z4.91); this covers catboost.cv's.

test_that("catboost.cv: a closure that throws inside calc_ders_range surfaces as a catchable R error", {
  pool <- catboost.load_pool(features, label = label)
  throwing_objective <- list(
    calc_ders_range = function(approx, target, weight) stop("boom from cv calc_ders_range")
  )
  expect_error(
    catboost.cv(
      pool, params = c(common_cv_params(), list(eval_metric = "RMSE")),
      fold_count = 3, partition_random_seed = 0,
      custom_objective = throwing_objective
    ),
    "error in R custom objective's calc_ders_range"
  )
  # Session survives: an unrelated subsequent call still works.
  result <- catboost.cv(
    pool, params = c(common_cv_params(), list(loss_function = "RMSE")),
    fold_count = 3, partition_random_seed = 0
  )
  expect_true(is.data.frame(result))
})

test_that("catboost.cv: a closure that throws inside evaluate surfaces as a catchable R error", {
  pool <- catboost.load_pool(features, label = label)
  throwing_metric <- list(
    evaluate = function(approx, target, weight) stop("boom from cv evaluate"),
    is_max_optimal = function() FALSE
  )
  expect_error(
    catboost.cv(
      pool, params = c(common_cv_params(), list(loss_function = "RMSE")),
      fold_count = 3, partition_random_seed = 0,
      custom_eval_metric_object = throwing_metric
    ),
    "error in R custom eval metric's evaluate"
  )
  result <- catboost.cv(
    pool, params = c(common_cv_params(), list(loss_function = "RMSE")),
    fold_count = 3, partition_random_seed = 0
  )
  expect_true(is.data.frame(result))
})

# --- (b) grid_search/randomized_search refit forwarding ---------------------
#
# A single-value grid isolates the fix from the search itself: best_params is
# deterministic, so any mismatch between the refit model and a directly
# trained custom-objective model can only come from the refit step silently
# dropping the custom objective/metric.

single_value_grid <- list(depth = c(4), learning_rate = c(0.1))

test_that("catboost.grid_search: refit = TRUE forwards custom_objective/custom_eval_metric_object", {
  pool <- catboost.load_pool(features, label = label)

  result <- catboost.grid_search(
    single_value_grid, pool,
    params = list(iterations = 20, random_seed = 1, logging_level = "Silent", boost_from_average = FALSE),
    cv = 3, refit = TRUE, verbose = FALSE,
    custom_objective = rmse_custom_objective,
    custom_eval_metric_object = rmse_custom_eval_metric
  )

  expect_false(is.null(result$model))

  direct_model <- catboost.train(
    pool,
    params = list(
      iterations = 20, random_seed = 1, logging_level = "Silent", boost_from_average = FALSE,
      depth = 4, learning_rate = 0.1
    ),
    custom_objective = rmse_custom_objective,
    custom_eval_metric_object = rmse_custom_eval_metric
  )

  refit_predict <- catboost.predict(result$model, pool, prediction_type = "RawFormulaVal")
  direct_predict <- catboost.predict(direct_model, pool, prediction_type = "RawFormulaVal")
  expect_equal(refit_predict, direct_predict, tolerance = 1e-12, check.attributes = FALSE)
})

test_that("catboost.randomized_search: refit = TRUE forwards custom_objective/custom_eval_metric_object", {
  pool <- catboost.load_pool(features, label = label)

  result <- catboost.randomized_search(
    single_value_grid, pool,
    params = list(iterations = 20, random_seed = 1, logging_level = "Silent", boost_from_average = FALSE),
    cv = 3, n_iter = 1, refit = TRUE, verbose = FALSE,
    custom_objective = rmse_custom_objective,
    custom_eval_metric_object = rmse_custom_eval_metric
  )

  expect_false(is.null(result$model))

  direct_model <- catboost.train(
    pool,
    params = list(
      iterations = 20, random_seed = 1, logging_level = "Silent", boost_from_average = FALSE,
      depth = 4, learning_rate = 0.1
    ),
    custom_objective = rmse_custom_objective,
    custom_eval_metric_object = rmse_custom_eval_metric
  )

  refit_predict <- catboost.predict(result$model, pool, prediction_type = "RawFormulaVal")
  direct_predict <- catboost.predict(direct_model, pool, prediction_type = "RawFormulaVal")
  expect_equal(refit_predict, direct_predict, tolerance = 1e-12, check.attributes = FALSE)
})

test_that("catboost.grid_search: omitting the new arguments reproduces prior (built-in-loss) behavior", {
  pool <- catboost.load_pool(features, label = label)
  result <- catboost.grid_search(
    single_value_grid, pool,
    params = list(iterations = 20, loss_function = "RMSE", random_seed = 1, logging_level = "Silent"),
    cv = 3, refit = TRUE, verbose = FALSE
  )
  expect_equal(result$params$depth, 4)
  expect_equal(result$params$learning_rate, 0.1)
  expect_false(is.null(result$model))
})

test_that("catboost.randomized_search: omitting the new arguments reproduces prior (built-in-loss) behavior", {
  pool <- catboost.load_pool(features, label = label)
  result <- catboost.randomized_search(
    single_value_grid, pool,
    params = list(iterations = 20, loss_function = "RMSE", random_seed = 1, logging_level = "Silent"),
    cv = 3, n_iter = 1, refit = TRUE, verbose = FALSE
  )
  expect_equal(result$params$depth, 4)
  expect_equal(result$params$learning_rate, 0.1)
  expect_false(is.null(result$model))
})

# --- (g) grid_search/randomized_search: the SEARCH ITSELF (src/catboostr.cpp
# GridSearch/RandomizedSearch call sites, NOT CrossValidate) actually uses the
# supplied custom objective/metric, not just the refit step (b) above already
# covers. A multi-value grid gives the search something to differentiate
# between; refit = FALSE isolates the search from the refit-forwarding fix.

multi_value_grid <- list(depth = c(3, 4), learning_rate = c(0.05, 0.1))

test_that("catboost.grid_search: the search itself (not just refit) uses custom_objective/custom_eval_metric_object", {
  pool <- catboost.load_pool(features, label = label)

  result_custom <- catboost.grid_search(
    multi_value_grid, pool,
    params = list(iterations = 20, random_seed = 1, logging_level = "Silent", boost_from_average = FALSE),
    cv = 3, refit = FALSE, verbose = FALSE,
    custom_objective = rmse_custom_objective,
    custom_eval_metric_object = rmse_custom_eval_metric
  )
  result_builtin <- catboost.grid_search(
    multi_value_grid, pool,
    params = list(iterations = 20, loss_function = "RMSE", random_seed = 1, logging_level = "Silent", boost_from_average = FALSE),
    cv = 3, refit = FALSE, verbose = FALSE
  )

  expect_equal(result_custom$params$depth, result_builtin$params$depth)
  expect_equal(result_custom$params$learning_rate, result_builtin$params$learning_rate)
  expect_equal(
    cv_test_metric_col(result_custom$cv_results, "mean"),
    cv_test_metric_col(result_builtin$cv_results, "mean"),
    tolerance = 1e-6
  )
})

test_that("catboost.randomized_search: the search itself (not just refit) uses custom_objective/custom_eval_metric_object", {
  pool <- catboost.load_pool(features, label = label)
  sample_grid <- list(depth = c(3, 4, 5), learning_rate = c(0.05, 0.1, 0.15))

  result_custom <- catboost.randomized_search(
    sample_grid, pool,
    params = list(iterations = 20, random_seed = 1, logging_level = "Silent", boost_from_average = FALSE),
    cv = 3, n_iter = 3, refit = FALSE, verbose = FALSE,
    custom_objective = rmse_custom_objective,
    custom_eval_metric_object = rmse_custom_eval_metric
  )
  result_builtin <- catboost.randomized_search(
    sample_grid, pool,
    params = list(iterations = 20, loss_function = "RMSE", random_seed = 1, logging_level = "Silent", boost_from_average = FALSE),
    cv = 3, n_iter = 3, refit = FALSE, verbose = FALSE
  )

  expect_equal(result_custom$params$depth, result_builtin$params$depth)
  expect_equal(result_custom$params$learning_rate, result_builtin$params$learning_rate)
  expect_equal(
    cv_test_metric_col(result_custom$cv_results, "mean"),
    cv_test_metric_col(result_builtin$cv_results, "mean"),
    tolerance = 1e-6
  )
})

# --- (c) select_features (metric only) and eval_feature (both) smoke tests --

test_that("catboost.select_features: custom_eval_metric_object (metric only) runs and returns a valid summary", {
  learn_pool <- catboost.load_pool(features, label = label)

  result <- catboost.select_features(
    learn_pool,
    features_for_select = c(0, 1, 2, 3, 4),
    num_features_to_select = 3,
    params = list(iterations = 10, random_seed = 1, logging_level = "Silent", boost_from_average = FALSE),
    steps = 1,
    train_final_model = TRUE,
    custom_eval_metric_object = rmse_custom_eval_metric
  )

  expect_equal(length(result$selected_features), 3)
  expect_equal(
    sort(c(result$selected_features, result$eliminated_features)),
    c(0, 1, 2, 3, 4)
  )
  expect_false(is.null(result$model))
})

test_that("catboost.select_features: has no custom_objective formal argument", {
  expect_false("custom_objective" %in% names(formals(catboost.select_features)))
})

test_that("catboost.eval_feature: custom_objective + custom_eval_metric_object run without error", {
  pool <- catboost.load_pool(features, label = label)

  result <- catboost.eval_feature(
    pool,
    features_to_evaluate = list(c(0L), c(1L)),
    params = list(iterations = 10, random_seed = 1, logging_level = "Silent", boost_from_average = FALSE,
                  train_dir = file.path(tempdir(), paste0("eval_feature_custom_", as.integer(runif(1, 0, 1e9))))),
    eval_mode = "OneVsOthers",
    fold_count = 2,
    relative_fold_size = 0.2,
    custom_objective = rmse_custom_objective,
    custom_eval_metric_object = rmse_custom_eval_metric
  )

  expect_equal(length(result$p_value), 2)
  expect_false(anyNA(unlist(result$metric_delta)))
})

# --- (d) backward-compatibility spot-check: omitting the new arguments -----

test_that("catboost.select_features: omitting custom_eval_metric_object reproduces prior output", {
  learn_pool <- catboost.load_pool(features, label = label)
  result <- catboost.select_features(
    learn_pool,
    features_for_select = c(0, 1, 2, 3, 4),
    num_features_to_select = 3,
    params = list(iterations = 10, random_seed = 1, logging_level = "Silent"),
    steps = 1,
    train_final_model = FALSE
  )
  expect_equal(length(result$selected_features), 3)
})

test_that("catboost.eval_feature: omitting the new arguments reproduces prior output", {
  pool <- catboost.load_pool(features, label = label)
  result <- catboost.eval_feature(
    pool,
    features_to_evaluate = list(c(0L)),
    params = list(iterations = 10, loss_function = "RMSE", random_seed = 1, logging_level = "Silent",
                  train_dir = file.path(tempdir(), paste0("eval_feature_compat_", as.integer(runif(1, 0, 1e9))))),
    eval_mode = "OneVsNone",
    fold_count = 2,
    relative_fold_size = 0.2
  )
  expect_equal(length(result$p_value), 1)
})
