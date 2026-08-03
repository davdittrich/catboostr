context("test_multitarget_differential.R")

# P3.7 (catboost-8z4.44) part a: differential test for multi-target losses.
#
# This exercises NO new implementation -- R's catboost.load_pool /
# catboost.train / catboost.predict already handle multi-target label
# matrices (tests/testthat/test_pool.R:39,:53 cover the self-consistent
# case). What was missing, and what the Phase 3 gate asks for, is a
# comparison against the pinned Python oracle rather than against R itself.
#
# Scope guard: docs/phase-2/P2.4-report.md "Finding 2" root-caused the
# multi-target failure to catboost.get_object_importance only; fit/predict
# were already correct. get_object_importance is deliberately NOT touched
# here (its MultiClass/multi-target fix belongs to Phase 4).
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_multitarget_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "multitarget.json"),
  simplifyVector = TRUE
)

multitarget_params <- function(loss_function) {
  params <- fixture$params
  params$verbose <- NULL
  params$loss_function <- loss_function
  params$logging_level <- "Silent"
  return(params)
}

test_that("multitarget: MultiRMSE pool construction matches Python oracle", {
  pool <- catboost.load_pool(fixture$inputs$features, label = fixture$inputs$multirmse_label)
  expect_equal(catboost.pool.num_row(pool), fixture$expected$num_row)
  expect_equal(catboost.pool.num_col(pool), fixture$expected$num_col)
  label <- catboost.pool.get_label(pool)
  expect_equal(dim(label), dim(fixture$expected$multirmse_get_label))
  expect_equal(label, fixture$expected$multirmse_get_label,
               tolerance = 1e-6, check.attributes = FALSE)
})

test_that("multitarget: MultiRMSE fit/predict matches Python oracle", {
  pool <- catboost.load_pool(fixture$inputs$features, label = fixture$inputs$multirmse_label)
  model <- catboost.train(pool, params = multitarget_params("MultiRMSE"))
  prediction <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(dim(prediction), dim(fixture$expected$multirmse_predict))
  expect_equal(prediction, fixture$expected$multirmse_predict,
               tolerance = 1e-6, check.attributes = FALSE)
})

test_that("multitarget: MultiLogloss pool construction matches Python oracle", {
  pool <- catboost.load_pool(fixture$inputs$features,
                             label = fixture$inputs$multilogloss_label)
  label <- catboost.pool.get_label(pool)
  expect_equal(label, fixture$expected$multilogloss_get_label,
               tolerance = 1e-6, check.attributes = FALSE)
})

test_that("multitarget: MultiLogloss fit/predict matches Python oracle", {
  # The oracle passes an integer 0/1 label matrix; keep it double on the R
  # side so the target is read as Float exactly as Python's does (an integer
  # matrix would be routed through the class-label path instead).
  label <- matrix(as.double(fixture$inputs$multilogloss_label),
                  nrow = nrow(fixture$inputs$multilogloss_label))
  pool <- catboost.load_pool(fixture$inputs$features, label = label)
  model <- catboost.train(pool, params = multitarget_params("MultiLogloss"))
  prediction <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(dim(prediction), dim(fixture$expected$multilogloss_predict))
  expect_equal(prediction, fixture$expected$multilogloss_predict,
               tolerance = 1e-6, check.attributes = FALSE)
})
