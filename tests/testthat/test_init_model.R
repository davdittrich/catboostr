context("test_init_model.R")

# P5.1 (catboost-8z4.58) -- init_model support in catboost.train (continue
# training from an existing model).
#
# Reuses the vendor's existing TrainModel(..., initModel, ...) overload
# (catboost/libs/train_lib/train_model.h:153) -- the same one Python's
# _CatBoost._train passes to via _catboost.pyx -- rather than reimplementing
# continuation. R only threads a TFullModel* handle through the existing
# CatBoostFit_R native call.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_init_model_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "init_model.json"),
  simplifyVector = TRUE
)

as_pool <- function(rows) {
  catboost.load_pool(fixture$inputs$features[rows, , drop = FALSE],
                      label = fixture$inputs$label[rows])
}

split <- fixture$inputs$split
base_rows <- seq_len(split)
continue_rows <- seq(split + 1, nrow(fixture$inputs$features))

base_params <- function() {
  p <- fixture$base_params
  p$verbose <- NULL
  p$logging_level <- "Silent"
  return(p)
}

continue_params <- function() {
  p <- fixture$continue_params
  p$verbose <- NULL
  p$logging_level <- "Silent"
  return(p)
}

test_that("init_model: continued training matches Python oracle predictions", {
  base_pool <- as_pool(base_rows)
  base_model <- catboost.train(base_pool, params = base_params())

  base_predict <- catboost.predict(base_model, base_pool, prediction_type = "RawFormulaVal")
  expect_equal(base_predict, fixture$expected$base_predict, tolerance = 1e-6, check.attributes = FALSE)

  continue_pool <- as_pool(continue_rows)
  continued_model <- catboost.train(continue_pool, params = continue_params(), init_model = base_model)

  continued_predict <- catboost.predict(continued_model, continue_pool, prediction_type = "RawFormulaVal")
  expect_equal(continued_predict, fixture$expected$continued_predict, tolerance = 1e-6, check.attributes = FALSE)

  all_pool <- as_pool(seq_len(nrow(fixture$inputs$features)))
  continued_predict_all <- catboost.predict(continued_model, all_pool, prediction_type = "RawFormulaVal")
  expect_equal(continued_predict_all, fixture$expected$continued_predict_all, tolerance = 1e-6, check.attributes = FALSE)
})

test_that("init_model: tree count accumulates across the continuation, matching the oracle", {
  base_pool <- as_pool(base_rows)
  base_model <- catboost.train(base_pool, params = base_params())
  expect_equal(base_model$tree_count, fixture$expected$base_tree_count)

  continue_pool <- as_pool(continue_rows)
  continued_model <- catboost.train(continue_pool, params = continue_params(), init_model = base_model)
  expect_equal(continued_model$tree_count, fixture$expected$continued_tree_count)
})

test_that("init_model accepts a model file path, same as a catboost.Model object", {
  base_pool <- as_pool(base_rows)
  base_model <- catboost.train(base_pool, params = base_params())

  model_path <- tempfile(fileext = ".cbm")
  catboost.save_model(base_model, model_path)

  continue_pool <- as_pool(continue_rows)
  continued_from_path <- catboost.train(continue_pool, params = continue_params(), init_model = model_path)

  continued_predict <- catboost.predict(continued_from_path, continue_pool, prediction_type = "RawFormulaVal")
  expect_equal(continued_predict, fixture$expected$continued_predict, tolerance = 1e-6, check.attributes = FALSE)
})

test_that("init_model rejects invalid argument types with a clear error", {
  learn_pool <- as_pool(base_rows)
  expect_error(
    catboost.train(learn_pool, params = base_params(), init_model = 42),
    "Expected catboost.Model or a path to a model file"
  )
})

test_that("catboost-8z4.58: catboost.train without init_model is unaffected (backward compat)", {
  # Same fixture/params as the pre-Phase-5 usage pattern (positional args,
  # no init_model) -- proves the new 4th argument is fully optional and
  # doesn't perturb existing call sites or their output.
  pool <- as_pool(base_rows)
  model <- catboost.train(pool, NULL, base_params())
  prediction <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(prediction, fixture$expected$base_predict, tolerance = 1e-6, check.attributes = FALSE)
  expect_equal(model$tree_count, fixture$expected$base_tree_count)
})
