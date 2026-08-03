context("test_grid_search.R")

# P5.2 (catboost-8z4.59) -- catboost.grid_search / catboost.randomized_search.
#
# Both call the exact same native NCB::GridSearch / NCB::RandomizedSearch
# entry points (catboost/private/libs/hyperparameter_tuning/hyperparameter_tuning.h)
# that Python's CatBoost._tune_hyperparams calls via its Cython wrapper
# (_catboost.pyx:4403 self._object._tune_hyperparams -> cpdef at
# _catboost.pyx:5933-6001: `if n_iter == -1: GridSearch(...) else: RandomizedSearch(...)`),
# rather than an R-side loop re-implementing the search over catboost.cv --
# the grid/quantization-parameter enumeration order, train/test-split reuse,
# and (for randomized_search) TRandom(seed=0)-seeded combination sampling all
# live in that native code, so calling it directly is what makes
# best-params/cv_results match the Python oracle bit-for-bit rather than
# merely approximately.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_grid_search_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "grid_search.json"),
  simplifyVector = TRUE
)

pool <- catboost.load_pool(fixture$inputs$features, label = fixture$inputs$label)

base_params <- function() {
  p <- fixture$base_params
  p$verbose <- NULL
  p$logging_level <- "Silent"
  return(p)
}

# jsonlite flattens a single-grid dict-of-vectors dict into a named list of
# vectors already -- exactly the named-list form catboost.grid_search's
# param_grid accepts directly (prepare_grid_json wraps it in an array itself).
param_grid <- as.list(fixture$param_grid)
param_distributions <- as.list(fixture$param_distributions)

expect_cv_results_match <- function(actual, expected_cols) {
  for (col in names(expected_cols)) {
    if (col == "iterations") next # R's cv_results has no iterations column, matching catboost.cv
    # data.frame()'s default check.names=TRUE sanitizes "-" to "." (same
    # convention catboost.cv's own data.frame(result) already relies on).
    r_col <- gsub("-", ".", col, fixed = TRUE)
    expect_true(r_col %in% names(actual), info = paste("missing column", r_col))
    expect_equal(actual[[r_col]], expected_cols[[col]], tolerance = 1e-12, check.attributes = FALSE)
  }
}

test_that("grid_search: best params, cv_results and refit prediction match the Python oracle", {
  expected <- fixture$expected$grid_search

  result <- catboost.grid_search(
    param_grid, pool, params = base_params(),
    cv = 3, partition_random_seed = 0,
    calc_cv_statistics = TRUE, search_by_train_test_split = TRUE,
    refit = TRUE, shuffle = TRUE, stratified = FALSE, train_size = 0.8,
    verbose = FALSE
  )

  expect_equal(result$params$depth, expected$params$depth)
  expect_equal(result$params$learning_rate, expected$params$learning_rate, tolerance = 1e-12)

  expect_cv_results_match(result$cv_results, expected$cv_results)

  expect_equal(result$model$tree_count, expected$refit_tree_count)
  refit_predict <- catboost.predict(result$model, pool, prediction_type = "RawFormulaVal")
  expect_equal(refit_predict, expected$refit_predict, tolerance = 1e-12, check.attributes = FALSE)
})

test_that("randomized_search: best params, cv_results and refit prediction match the Python oracle", {
  expected <- fixture$expected$randomized_search

  result <- catboost.randomized_search(
    param_distributions, pool, params = base_params(),
    cv = 3, n_iter = 6, partition_random_seed = 0,
    calc_cv_statistics = TRUE, search_by_train_test_split = TRUE,
    refit = TRUE, shuffle = TRUE, stratified = FALSE, train_size = 0.8,
    verbose = FALSE
  )

  expect_equal(result$params$depth, expected$params$depth)
  expect_equal(result$params$learning_rate, expected$params$learning_rate, tolerance = 1e-12)

  expect_cv_results_match(result$cv_results, expected$cv_results)

  expect_equal(result$model$tree_count, expected$refit_tree_count)
  refit_predict <- catboost.predict(result$model, pool, prediction_type = "RawFormulaVal")
  expect_equal(refit_predict, expected$refit_predict, tolerance = 1e-12, check.attributes = FALSE)
})

test_that("grid_search: a list of multiple grids explores each independently (no crash, valid result)", {
  multi_grid <- list(
    list(depth = c(3), learning_rate = c(0.3)),
    list(depth = c(5), learning_rate = c(0.05))
  )
  result <- catboost.grid_search(
    multi_grid, pool, params = base_params(),
    cv = 3, refit = FALSE, verbose = FALSE
  )
  expect_true(result$params$depth %in% c(3, 5))
  expect_true(!is.null(result$cv_results))
  expect_null(result$model)
})

test_that("randomized_search rejects a non-positive n_iter", {
  expect_error(
    catboost.randomized_search(param_distributions, pool, n_iter = 0),
    "n_iter should be a positive number"
  )
})
