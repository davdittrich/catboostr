context("test_tree_internals.R")

# catboost-8z4.119 (P10.E) differential test: R's catboost.get_borders(),
# catboost.save_borders(), catboost.get_leaf_values(),
# catboost.get_leaf_weights(), catboost.get_tree_leaf_counts(),
# catboost.set_leaf_values(), catboost.calc_leaf_indexes(), and
# catboost.iterate_leaf_indexes() against Python catboost==1.2.10's
# CatBoost.get_borders()/save_borders()/get_leaf_values()/get_leaf_weights()/
# get_tree_leaf_counts()/set_leaf_values()/calc_leaf_indexes()/
# iterate_leaf_indexes() (catboost/python-package/catboost/core.py:2112-2152,
# 3132-3195, 3809-3827). All 8 methods are defined exactly once (4 on
# _CatBoostBase, 4 on CatBoost itself) and never overridden by
# CatBoostClassifier/CatBoostRegressor/CatBoostRanker, so one fixture model
# (a CatBoostRegressor) is the parity target for all 32 matrix rows (8
# families x 4 sklearn classes). A second, MultiClass fixture model
# (ApproxDimension == 3) additionally covers get_leaf_values()/
# set_leaf_values()'s flat array being leaf_count * ApproxDimension long,
# not just leaf_count -- the one non-obvious part of this API family.
# get_leaf_weights() is not exercised for MultiClass: upstream's own
# _catboost.pyx:6108-6115 has a size-mismatch assert that raises
# AssertionError for every ApproxDimension > 1 model, so there is no correct
# Python oracle value to pin (see gen_tree_internals_fixture.py). R's
# catboost.get_leaf_weights() is unaffected (always leaf_count long) and is
# already covered by the RMSE case above.
#
# set_leaf_values()'s mutation is verified against Python's own observable
# behavior (confirmed empirically against the pinned oracle, not assumed):
# TFullModel caches a compiled formula evaluator (model.h's
# TFullModel::Evaluator) that is NOT invalidated by a raw leaf-value
# mutation, so predict() on the same in-memory handle keeps returning
# pre-mutation values until the model is saved and reloaded (only
# SetScaleAndBias() resets that cache; SetLeafValues() does not). The
# fixture therefore records predictions_after_set from a save+reload
# round trip, and the R test does the same round trip via
# catboost.save_model()/catboost.load_model() rather than predicting
# directly on the mutated handle.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_tree_internals_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "tree_internals.json"),
  simplifyVector = TRUE
)

build_pool_and_model <- function(fixture) {
  inputs <- fixture$inputs
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = as.double(inputs$label))
  catboost.pool.set_feature_names(pool, inputs$feature_names)
  model <- catboost.train(pool, params = list(
    loss_function = "RMSE", iterations = 10, depth = 2, random_seed = 42,
    thread_count = 1, logging_level = "Silent"
  ))
  list(pool = pool, model = model)
}

test_that("get_borders matches Python oracle", {
  built <- build_pool_and_model(fixture)
  borders <- catboost.get_borders(built$model)
  expected <- fixture$expected$borders
  expect_equal(sort(names(borders)), sort(names(expected)))
  for (key in names(expected)) {
    expect_equal(as.numeric(borders[[key]]), as.numeric(expected[[key]]), tolerance = 1e-6)
  }
})

test_that("save_borders output matches Python oracle byte-for-byte", {
  built <- build_pool_and_model(fixture)
  out <- tempfile()
  on.exit(unlink(out))
  catboost.save_borders(built$model, out)
  borders_text <- readChar(out, file.info(out)$size)
  expect_equal(borders_text, fixture$expected$borders_text)
})

test_that("get_leaf_values matches Python oracle", {
  built <- build_pool_and_model(fixture)
  expect_equal(catboost.get_leaf_values(built$model), fixture$expected$leaf_values, tolerance = 1e-6)
})

test_that("get_leaf_weights matches Python oracle", {
  built <- build_pool_and_model(fixture)
  expect_equal(catboost.get_leaf_weights(built$model), fixture$expected$leaf_weights, tolerance = 1e-6)
})

test_that("get_tree_leaf_counts matches Python oracle", {
  built <- build_pool_and_model(fixture)
  expect_equal(catboost.get_tree_leaf_counts(built$model), as.integer(fixture$expected$tree_leaf_counts))
})

test_that("calc_leaf_indexes (full tree range) matches Python oracle", {
  built <- build_pool_and_model(fixture)
  leaf_indexes <- catboost.calc_leaf_indexes(built$model, built$pool)
  # jsonlite::fromJSON(simplifyVector = TRUE) already simplifies a JSON
  # array-of-equal-length-arrays into an R matrix -- use it as-is, do not
  # unlist()+matrix() it again (that scrambles row/column order).
  expected <- fixture$expected$leaf_indexes_full
  expect_equal(unname(leaf_indexes), unname(expected))
})

test_that("calc_leaf_indexes (restricted ntree_start/ntree_end) matches Python oracle", {
  built <- build_pool_and_model(fixture)
  leaf_indexes <- catboost.calc_leaf_indexes(built$model, built$pool, ntree_start = 1, ntree_end = 4)
  expected <- fixture$expected$leaf_indexes_range
  expect_equal(unname(leaf_indexes), unname(expected))
})

test_that("iterate_leaf_indexes matches Python oracle and calc_leaf_indexes", {
  built <- build_pool_and_model(fixture)
  leaf_list <- catboost.iterate_leaf_indexes(built$model, built$pool)
  expected <- fixture$expected$leaf_indexes_iter
  expect_equal(length(leaf_list), nrow(expected))
  for (i in seq_along(leaf_list)) {
    expect_equal(unname(leaf_list[[i]]), as.integer(expected[i, ]))
  }
  leaf_matrix <- catboost.calc_leaf_indexes(built$model, built$pool)
  for (i in seq_along(leaf_list)) {
    expect_equal(unname(leaf_list[[i]]), unname(leaf_matrix[i, ]))
  }
})

test_that("set_leaf_values matches Python oracle (leaf values immediately; predictions after save+reload)", {
  built <- build_pool_and_model(fixture)
  new_leaf_values <- fixture$expected$new_leaf_values
  catboost.set_leaf_values(built$model, new_leaf_values)
  expect_equal(catboost.get_leaf_values(built$model), fixture$expected$leaf_values_after_set, tolerance = 1e-6)

  # predict() on the same in-memory handle still reflects the pre-mutation
  # formula evaluator (see file header) -- reload from disk first, matching
  # the oracle fixture's own recording method.
  out <- tempfile(fileext = ".cbm")
  on.exit(unlink(out))
  catboost.save_model(built$model, out)
  reloaded <- catboost.load_model(out)
  predictions <- catboost.predict(reloaded, built$pool, prediction_type = "RawFormulaVal")
  expect_equal(as.numeric(predictions), as.numeric(fixture$expected$predictions_after_set), tolerance = 1e-6)
})

test_that("set_leaf_values rejects a wrong-length vector", {
  built <- build_pool_and_model(fixture)
  expect_error(catboost.set_leaf_values(built$model, c(1.0, 2.0)))
})

build_multiclass_pool_and_model <- function(fixture) {
  inputs <- fixture$inputs
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = as.double(inputs$multiclass_label))
  catboost.pool.set_feature_names(pool, inputs$feature_names)
  model <- catboost.train(pool, params = list(
    loss_function = "MultiClass", iterations = 10, depth = 2, random_seed = 42,
    thread_count = 1, logging_level = "Silent"
  ))
  list(pool = pool, model = model)
}

test_that("get_leaf_values / set_leaf_values match Python oracle for a MultiClass model (ApproxDimension > 1)", {
  built <- build_multiclass_pool_and_model(fixture)
  expected <- fixture$expected

  leaf_values <- catboost.get_leaf_values(built$model)
  # ApproxDimension == 3 -> flat array is leaf_count * 3 long, not leaf_count.
  expect_equal(length(leaf_values), 3L * sum(catboost.get_tree_leaf_counts(built$model)))
  expect_equal(leaf_values, expected$multiclass_leaf_values, tolerance = 1e-6)
  expect_equal(catboost.get_tree_leaf_counts(built$model), as.integer(expected$multiclass_tree_leaf_counts))

  catboost.set_leaf_values(built$model, expected$multiclass_new_leaf_values)
  expect_equal(catboost.get_leaf_values(built$model), expected$multiclass_leaf_values_after_set, tolerance = 1e-6)

  out <- tempfile(fileext = ".cbm")
  on.exit(unlink(out))
  catboost.save_model(built$model, out)
  reloaded <- catboost.load_model(out)
  predictions <- catboost.predict(reloaded, built$pool, prediction_type = "RawFormulaVal")
  # jsonlite::fromJSON(simplifyVector = TRUE) already simplifies this into an
  # R matrix -- use it as-is (see the calc_leaf_indexes tests above for why
  # unlist()+matrix() on an already-simplified field is wrong).
  expect_equal(unname(as.matrix(predictions)), unname(expected$multiclass_predictions_after_set), tolerance = 1e-6)
})
