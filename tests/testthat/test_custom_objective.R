context("test_custom_objective.R")

# P6.3 (catboost-8z4.89): custom R objective wired into catboost.train() via
# the P6.2 bridge (src/r_callback_bridge.h, src/r_custom_objective.h/.cpp).
#
# Smoke test: reimplement RMSE's own derivatives (TRMSEError,
# vendor/catboost/catboost/private/libs/algo_helpers/error_functions.h:379-403
# -- Der1 = weight*(target-approx), Der2 = weight*(-1), user applies the
# weight itself, matching Python's calc_ders_range contract) via
# calc_ders_range and confirm the resulting model matches a built-in
# loss_function = "RMSE" run bit-for-bit (same random_seed => same tree
# structure when the gradients/hessians match exactly), AND confirm training
# actually used more than one concurrently active worker (multithreading was
# NOT silently downgraded to thread_count = 1) via the bridge's own
# active-worker instrumentation (catboost-8z4.93/.91).

set.seed(20260807)
n <- 300
features <- data.frame(
  x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n), x5 = rnorm(n)
)
label <- with(features, 2 * x1 - 1.5 * x2 + 0.5 * x3 + rnorm(n, sd = 0.3))

common_params <- function() {
  list(
    iterations = 30,
    learning_rate = 0.1,
    depth = 4,
    random_seed = 1,
    logging_level = "Silent",
    # RMSE defaults boost_from_average to TRUE when unset (data-dependent
    # default, options_helper.cpp's AdjustBoostFromAverageDefaultValue --
    # limited to a fixed allowlist of built-in losses); a user-defined
    # loss_function is REJECTED outright if boost_from_average is set TRUE
    # (catboost_options.cpp:705-709 -- not in that allowlist, "PythonUser
    # DefinedPerObject" included). Pin it FALSE for both runs so this
    # reimplemented-RMSE comparison isn't confounded by a differing
    # starting bias, and so it's a value both loss functions accept.
    boost_from_average = FALSE
  )
}

# custom_objective params, matching common_params() plus the eval_metric a
# user-defined loss_function requires (catboost/libs/metrics/metric.cpp:
# "If loss function is a user defined object, then the eval metric must be
# specified" -- there is no default metric to infer from an opaque R
# closure).
custom_objective_params <- function() {
  c(common_params(), list(eval_metric = "RMSE"))
}

# Mirrors TRMSEError exactly (error_functions.h:379-403): the custom-objective
# path does NOT auto-multiply by weight (unlike the built-in dispatcher --
# see IDerCalcer::CalcDersRangeImpl, error_functions.cpp:46-56), so the R
# closure applies it itself, exactly as Python's calc_ders_range contract
# requires.
rmse_custom_objective <- list(
  calc_ders_range = function(approx, target, weight) {
    w <- if (is.null(weight)) 1 else weight
    der1 <- w * (target - approx)
    der2 <- w * (-1)
    cbind(der1, der2)
  }
)

test_that("custom_objective validates its structure before reaching native code", {
  pool <- catboost.load_pool(features, label = label)
  expect_error(
    catboost.train(pool, params = common_params(), custom_objective = "not a list"),
    "must be a list"
  )
  expect_error(
    catboost.train(pool, params = common_params(), custom_objective = list(foo = 1)),
    "calc_ders_range.*calc_ders_multi"
  )
  expect_error(
    catboost.train(
      pool,
      params = c(common_params(), list(loss_function = "RMSE")),
      custom_objective = rmse_custom_objective
    ),
    "PythonUserDefinedPerObject"
  )
})

test_that("custom_objective calc_ders_range reimplementing RMSE matches built-in RMSE", {
  pool <- catboost.load_pool(features, label = label)

  model_builtin <- catboost.train(
    pool, params = c(common_params(), list(loss_function = "RMSE"))
  )
  model_custom <- catboost.train(
    pool, params = custom_objective_params(), custom_objective = rmse_custom_objective
  )

  pred_builtin <- catboost.predict(model_builtin, pool, prediction_type = "RawFormulaVal")
  pred_custom <- catboost.predict(model_custom, pool, prediction_type = "RawFormulaVal")

  expect_equal(pred_custom, pred_builtin, tolerance = 1e-6)
})

test_that("custom_objective training preserves multithreading (bridge instrumentation)", {
  # CatBoost only splits CalcDersRange's objects across multiple TBB blocks
  # once the pool has >= 10000 rows (AdjustBlockSize,
  # approx_updater_helpers.h) -- below that it runs as a single block on one
  # thread regardless of thread_count, which is not evidence of anything.
  # thread_count is a Pool-construction option (catboost.load_pool), not a
  # catboost.train() params key.
  set.seed(1)
  big_n <- 12000
  big_features <- data.frame(x1 = rnorm(big_n), x2 = rnorm(big_n))
  big_label <- with(big_features, x1 - x2 + rnorm(big_n, sd = 0.3))
  pool <- catboost.load_pool(big_features, label = big_label, thread_count = 4)

  catboost.train(
    pool,
    params = c(custom_objective_params(), list(iterations = 3, depth = 2)),
    custom_objective = rmse_custom_objective
  )
  max_active_workers <- .Call("CatBoostLastCustomObjectiveMaxActiveWorkers_R")
  expect_gt(max_active_workers, 1L)
})

test_that("built-in-loss training is unaffected by custom_objective plumbing", {
  pool <- catboost.load_pool(features, label = label)
  model <- catboost.train(pool, params = c(common_params(), list(loss_function = "RMSE")))
  pred <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_length(pred, n)
  expect_false(anyNA(pred))
})
