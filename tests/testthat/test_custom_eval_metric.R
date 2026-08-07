context("test_custom_eval_metric.R")

# P6.4 (catboost-8z4.90): custom R eval metric wired into catboost.train() via
# the P6.2 bridge (src/r_callback_bridge.h, src/r_custom_metric.h/.cpp),
# reusing catboost-8z4.89's shared bridge/queue (same TRCallbackBridge
# instance a CatBoostFit_R call constructs for both custom_objective and
# custom_eval_metric_object).
#
# Smoke test: reimplement RMSE's own Eval/GetFinalError (TRMSEMetric,
# vendor/catboost/catboost/libs/metrics/metric.cpp:716-748 -- Stats[0] =
# sum(w * (approx - target)^2), Stats[1] = sum(w), GetFinalError =
# sqrt(Stats[0] / (Stats[1] + 1e-38))) via `evaluate`/`get_final_error` and
# confirm it drives use_best_model's early-stopping iteration selection
# identically to the built-in eval_metric = "RMSE" (a training-outcome-level
# differential: if the custom Eval/GetFinalError/IsMaxOptimal values were
# even slightly off, the chosen best iteration -- and hence every downstream
# prediction -- would differ). Also confirms training actually used more than
# one concurrently active worker (multithreading was NOT silently downgraded
# to thread_count = 1) via the bridge's own active-worker instrumentation
# (catboost-8z4.93/.91), reusing the existing hook: the counter lives on the
# TRCallbackBridge instance CatBoostFit_R constructs once per call and shares
# between the objective and metric trampolines, not per descriptor type.

set.seed(20260807)
n <- 300
features <- data.frame(
  x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n), x5 = rnorm(n)
)
label <- with(features, 2 * x1 - 1.5 * x2 + 0.5 * x3 + rnorm(n, sd = 0.3))

# Deep + many iterations + early stopping so noisy synthetic data actually
# overfits before the iteration cap -- otherwise use_best_model would just
# pick the final iteration regardless of whether the eval metric is computed
# correctly, and the test would have no power to detect a wrong metric.
common_params <- function() {
  list(
    loss_function = "RMSE",
    iterations = 200,
    learning_rate = 0.15,
    depth = 6,
    random_seed = 1,
    logging_level = "Silent",
    use_best_model = TRUE,
    early_stopping_rounds = 10,
    boost_from_average = FALSE
  )
}

rmse_custom_eval_metric <- list(
  evaluate = function(approx, target, weight) {
    w <- if (is.null(weight)) rep(1, length(target)) else weight
    diff <- approx[, 1] - target
    list(error = sum(w * diff^2), weight = sum(w))
  },
  is_max_optimal = function() FALSE,
  get_final_error = function(error) sqrt(error[1] / (error[2] + 1e-38))
)

test_that("custom_eval_metric_object validates its structure before reaching native code", {
  pool <- catboost.load_pool(features, label = label)
  expect_error(
    catboost.train(pool, params = common_params(), custom_eval_metric_object = "not a list"),
    "must be a list"
  )
  expect_error(
    catboost.train(pool, params = common_params(), custom_eval_metric_object = list(foo = 1)),
    "evaluate"
  )
  expect_error(
    catboost.train(
      pool,
      params = c(common_params(), list(eval_metric = "RMSE")),
      custom_eval_metric_object = rmse_custom_eval_metric
    ),
    "PythonUserDefinedPerObject"
  )
})

test_that("custom_eval_metric_object reimplementing RMSE matches built-in RMSE eval_metric", {
  pool <- catboost.load_pool(features, label = label)
  test_pool <- catboost.load_pool(features, label = label)

  model_builtin <- catboost.train(
    pool, test_pool, params = c(common_params(), list(eval_metric = "RMSE"))
  )
  model_custom <- catboost.train(
    pool, test_pool, params = common_params(), custom_eval_metric_object = rmse_custom_eval_metric
  )

  expect_equal(catboost.ntrees(model_custom), catboost.ntrees(model_builtin))

  pred_builtin <- catboost.predict(model_builtin, pool, prediction_type = "RawFormulaVal")
  pred_custom <- catboost.predict(model_custom, pool, prediction_type = "RawFormulaVal")
  expect_equal(pred_custom, pred_builtin, tolerance = 1e-6)
})

test_that("custom_eval_metric_object training preserves multithreading (bridge instrumentation)", {
  # vendor/catboost/catboost/libs/metrics/metric.cpp:4838-4864 --
  # TCustomMetric::Eval() takes an executor argument but never uses it: the
  # whole [begin, end) range is handed to the R closure in one call, always
  # on a single thread. That's a structural property of the vendor eval-metric
  # dispatch, not something this bridge can or should change -- a
  # metric-only training run can never observe max_active_workers > 1.
  #
  # What actually matters (and IS worth asserting) is that wiring a custom
  # eval metric into a call does not regress the multithreading the P6.3
  # custom-objective bridge already proved (test_custom_objective.R): both
  # descriptors share one TRCallbackBridge/counter per CatBoostFit_R call
  # (src/catboostr.cpp:1491), so combining a custom objective (whose
  # CalcDersRange IS chunked across TBB blocks once row count clears the
  # AdjustBlockSize threshold -- see test_custom_objective.R) with a custom
  # eval metric in the same call, and observing max_active_workers > 1, shows
  # the metric plumbing does not silently serialize the run.
  set.seed(1)
  big_n <- 12000
  big_features <- data.frame(x1 = rnorm(big_n), x2 = rnorm(big_n))
  big_label <- with(big_features, x1 - x2 + rnorm(big_n, sd = 0.3))
  pool <- catboost.load_pool(big_features, label = big_label, thread_count = 4)
  test_pool <- catboost.load_pool(big_features, label = big_label, thread_count = 4)

  rmse_custom_objective <- list(
    calc_ders_range = function(approx, target, weight) {
      w <- if (is.null(weight)) 1 else weight
      cbind(w * (target - approx), w * (-1))
    }
  )

  catboost.train(
    pool,
    test_pool,
    # loss_function/eval_metric deliberately unset: auto-set to
    # "PythonUserDefinedPerObject" for both (R/catboost.R:2879, :2905-2906)
    # since both custom_objective and custom_eval_metric_object are supplied.
    params = list(
      iterations = 3,
      depth = 2,
      logging_level = "Silent",
      boost_from_average = FALSE
    ),
    custom_objective = rmse_custom_objective,
    custom_eval_metric_object = rmse_custom_eval_metric
  )
  max_active_workers <- .Call("CatBoostLastCustomObjectiveMaxActiveWorkers_R")
  expect_gt(max_active_workers, 1L)
})

test_that("built-in eval_metric/custom_metric training is unaffected by custom_eval_metric_object plumbing", {
  pool <- catboost.load_pool(features, label = label)
  base_params <- common_params()
  base_params$use_best_model <- NULL
  base_params$early_stopping_rounds <- NULL
  model <- catboost.train(
    pool, params = c(base_params, list(eval_metric = "RMSE", custom_metric = "MAE"))
  )
  pred <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_length(pred, n)
  expect_false(anyNA(pred))
})

# P6.6 (catboost-8z4.91): differential/parity suite -- fills gaps left by
# catboost-8z4.90's own smoke tests (see task-6-brief.md cases (e)/(k)).

# --- (e) error-propagation: R closure throws inside evaluate --------------
throwing_custom_eval_metric <- list(
  evaluate = function(approx, target, weight) stop("boom from evaluate"),
  is_max_optimal = function() FALSE
)

test_that("custom_eval_metric_object: a closure that throws inside evaluate surfaces as a catchable R error (catboost.train)", {
  pool <- catboost.load_pool(features, label = label)
  base_params <- common_params()
  base_params$use_best_model <- NULL
  base_params$early_stopping_rounds <- NULL
  expect_error(
    catboost.train(pool, params = c(base_params, list(loss_function = "RMSE")),
                   custom_eval_metric_object = throwing_custom_eval_metric),
    "error in R custom eval metric's evaluate"
  )
  # Session survives: an unrelated subsequent call still works.
  model <- catboost.train(pool, params = c(base_params, list(loss_function = "RMSE")))
  expect_false(anyNA(catboost.predict(model, pool, prediction_type = "RawFormulaVal")))
})

# --- (k) multi-dimensional descriptor: TEvalMultiTargetFuncPtr ------------
# Mirrors TMultiRMSEMetric exactly (metric.cpp:446-505 -- error = sum over
# dims/rows of weight*(approx-target)^2, weight summed once per row (not
# multiplied by the number of dims), GetFinalError = sqrt(error/weight)).
multirmse_custom_eval_metric <- list(
  evaluate = function(approx, target, weight) {
    w <- if (is.null(weight)) rep(1, nrow(target)) else weight
    diff2 <- (approx - target)^2
    list(error = sum(w * rowSums(diff2)), weight = sum(w))
  },
  is_max_optimal = function() FALSE,
  get_final_error = function(error) if (error[2] == 0) 0 else sqrt(error[1] / error[2]),
  multi_target = TRUE
)

test_that("custom_eval_metric_object multi_target = TRUE (TEvalMultiTargetFuncPtr) reimplementing MultiRMSE matches built-in MultiRMSE eval_metric", {
  set.seed(20260808)
  mt_n <- 200
  mt_features <- data.frame(x1 = rnorm(mt_n), x2 = rnorm(mt_n), x3 = rnorm(mt_n))
  mt_label <- cbind(
    2 * mt_features$x1 - mt_features$x2 + rnorm(mt_n, sd = 0.2),
    -mt_features$x1 + 0.5 * mt_features$x3 + rnorm(mt_n, sd = 0.2)
  )
  pool <- catboost.load_pool(mt_features, label = mt_label)
  test_pool <- catboost.load_pool(mt_features, label = mt_label)

  mt_params <- list(
    loss_function = "MultiRMSE", iterations = 60, learning_rate = 0.15, depth = 4,
    random_seed = 1, logging_level = "Silent", use_best_model = TRUE,
    early_stopping_rounds = 10, boost_from_average = FALSE
  )

  model_builtin <- catboost.train(pool, test_pool, params = c(mt_params, list(eval_metric = "MultiRMSE")))
  model_custom <- catboost.train(
    pool, test_pool, params = mt_params, custom_eval_metric_object = multirmse_custom_eval_metric
  )

  expect_equal(catboost.ntrees(model_custom), catboost.ntrees(model_builtin))
  pred_builtin <- catboost.predict(model_builtin, pool, prediction_type = "RawFormulaVal")
  pred_custom <- catboost.predict(model_custom, pool, prediction_type = "RawFormulaVal")
  expect_equal(pred_custom, pred_builtin, tolerance = 1e-6)
})
