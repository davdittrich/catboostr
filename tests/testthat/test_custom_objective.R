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

# Synthetic-data preamble (set.seed/n/features/label) lives in
# helper-custom-callbacks.R (auto-sourced by testthat).

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

# catboost-8z4.97: catboost.train used to apply_custom_objective_params()/
# apply_custom_eval_metric_params() BEFORE process_synonyms(params), unlike
# catboost.grid_search/catboost.randomized_search (which already ran
# process_synonyms() first). A conflicting loss_function supplied via the
# 'objective' alias (process_synonyms' loss_function/objective group) was
# therefore invisible to apply_custom_objective_params's params$loss_function
# check -- it saw NULL, silently defaulted to "PythonUserDefinedPerObject",
# and left the stray 'objective' = "RMSE" key for process_synonyms to
# resolve afterward, masking the same conflict the loss_function-spelled
# case above correctly rejects. Now that process_synonyms() runs first,
# the alias resolves to params$loss_function before validation, so this
# must fail the same way as the loss_function-spelled case.
test_that("custom_objective: a conflicting loss_function supplied via the 'objective' alias is rejected the same way as 'loss_function' (catboost-8z4.97)", {
  pool <- catboost.load_pool(features, label = label)
  expect_error(
    catboost.train(
      pool,
      params = c(common_params(), list(objective = "RMSE")),
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

# P6.6 (catboost-8z4.91): differential/parity suite -- fills gaps left by
# catboost-8z4.89/.90/.94's own smoke tests (see task-6-brief.md cases
# (e)/(f)/(j)/(k)/(l)). Reuses this file's `features`/`label`/`common_params`.

# --- (e) error-propagation: R closure throws inside calc_ders_range -------
# The closure's own stop() message does not cross the C++ boundary --
# RunCalcDersRangeOnMainThread (r_custom_objective.cpp:51) catches the
# R-side error and rethrows a fixed std::runtime_error with generic text;
# that generic text, not "boom from calc_ders_range", is what
# catboost.train() actually raises.
throwing_custom_objective <- list(
  calc_ders_range = function(approx, target, weight) stop("boom from calc_ders_range")
)

test_that("custom_objective: a closure that throws inside calc_ders_range surfaces as a catchable R error (catboost.train)", {
  pool <- catboost.load_pool(features, label = label)
  expect_error(
    catboost.train(pool, params = custom_objective_params(), custom_objective = throwing_custom_objective),
    "error in R custom objective's calc_ders_range"
  )
  # Session survives: an unrelated subsequent call still works (no
  # PROTECT-stack / R_GlobalEnv corruption from the failed callback).
  model <- catboost.train(pool, params = c(common_params(), list(loss_function = "RMSE")))
  expect_false(anyNA(catboost.predict(model, pool, prediction_type = "RawFormulaVal")))
})

# --- (f) background-thread-death: TrainModel itself throws, unrelated to --
# the R closure -- exercises catboost-8z4.93's TRCallbackBridge::Run()
# WorkError_ propagation (r_callback_bridge.cpp: the background std::thread's
# exception is captured via std::exception_ptr and rethrown on the main
# thread once DrainLoop() returns), NOT the R-closure-throws path above.
# depth = 17 with the default SymmetricTree grow policy trips
# TObliviousTreeLearnerOptions::Validate()'s CB_ENSURE(MaxDepth <= 16,
# "Maximum tree depth is 16") (oblivious_tree_options.cpp:128), deep inside
# TrainModel() -- i.e. on the background thread, after the custom-objective
# descriptor is already wired up and running.
test_that("custom_objective: an invalid training config that throws inside TrainModel itself surfaces as a catchable R error within a bounded time (background-thread death)", {
  pool <- catboost.load_pool(features, label = label)
  started <- Sys.time()
  expect_error(
    catboost.train(
      pool,
      params = modifyList(custom_objective_params(), list(depth = 17)),
      custom_objective = rmse_custom_objective
    ),
    "Maximum tree depth is 16"
  )
  expect_lt(as.numeric(difftime(Sys.time(), started, units = "secs")), 60)
})

# --- (j) interrupt-responsiveness: the drain loop's wait is bounded -------
# The bridge mechanism itself (TRCallbackBridge::DrainLoop()'s bounded
# wait_for(kDrainPollMs) + R_CheckUserInterrupt() poll) is unit-tested
# directly, with an actual InterruptPolls() count assertion, by
# test_r_callback_bridge.R's "the drain loop polls for interrupts while the
# queue is idle" test -- that is the real assertion that Ctrl-C stays
# responsive (catboost-upw tracks restoring true tryCatch(interrupt=)
# semantics separately; today it surfaces as an ordinary R error, per
# r_callback_bridge.h's KNOWN DEVIATION comment). What is NOT covered
# elsewhere is that a real catboost.train() run with a custom objective
# active actually goes through that same bounded-wait mechanism end-to-end,
# rather than some other, unbounded blocking path: this test proves the run
# completes (does not hang) within a generous wall-clock bound.
test_that("custom_objective training's background/main-thread handoff completes within a bounded time (does not hang, case j)", {
  set.seed(1)
  big_n <- 12000
  big_features <- data.frame(x1 = rnorm(big_n), x2 = rnorm(big_n))
  big_label <- with(big_features, x1 - x2 + rnorm(big_n, sd = 0.3))
  pool <- catboost.load_pool(big_features, label = big_label, thread_count = 4)

  started <- Sys.time()
  catboost.train(
    pool,
    params = modifyList(custom_objective_params(), list(iterations = 5, depth = 2)),
    custom_objective = rmse_custom_objective
  )
  expect_lt(as.numeric(difftime(Sys.time(), started, units = "secs")), 60)
})

# --- (k) multi-dimensional descriptor: calc_ders_multi / CalcDersMultiTarget
# Mirrors TMultiRMSEError exactly (error_functions.h -- der1[i] =
# weight*(target[i]-approx[i]), diagonal Hessian der2[i] = -weight).
# leaf_estimation_method = "Gradient" sidesteps the Hessian so the R closure
# only needs to return der1 (der2 = NULL), matching what the built-in loss
# computes when no Hessian is requested.
multirmse_custom_objective <- list(
  calc_ders_multi = function(approx, target, weight) {
    list(der1 = weight * (target - approx), der2 = NULL)
  }
)

test_that("custom_objective calc_ders_multi (CalcDersMultiTarget) reimplementing MultiRMSE matches built-in MultiRMSE", {
  set.seed(20260808)
  mt_n <- 200
  mt_features <- data.frame(x1 = rnorm(mt_n), x2 = rnorm(mt_n), x3 = rnorm(mt_n))
  mt_label <- cbind(
    2 * mt_features$x1 - mt_features$x2 + rnorm(mt_n, sd = 0.2),
    -mt_features$x1 + 0.5 * mt_features$x3 + rnorm(mt_n, sd = 0.2)
  )
  pool <- catboost.load_pool(mt_features, label = mt_label)

  mt_params <- list(
    iterations = 20, learning_rate = 0.1, depth = 4, random_seed = 1,
    logging_level = "Silent", boost_from_average = FALSE,
    leaf_estimation_method = "Gradient"
  )

  model_builtin <- catboost.train(pool, params = c(mt_params, list(loss_function = "MultiRMSE")))
  model_custom <- catboost.train(
    pool,
    params = c(mt_params, list(loss_function = "PythonUserDefinedMultiTarget", eval_metric = "MultiRMSE")),
    custom_objective = multirmse_custom_objective
  )

  pred_builtin <- catboost.predict(model_builtin, pool, prediction_type = "RawFormulaVal")
  pred_custom <- catboost.predict(model_custom, pool, prediction_type = "RawFormulaVal")
  expect_equal(pred_custom, pred_builtin, tolerance = 1e-6)
})

# --- (l) logging-through-queue: verbose training with a custom objective --
# active must produce non-garbled log output via LogFromAnyThread()'s
# queue-rerouting path (r_callback_bridge.cpp's Log()/DrainLoop()
# ServiceRequest handling of log-line requests, the same fire-and-forget
# path test_r_callback_bridge.R's "log" mode exercises directly on the
# bridge) without corrupting the session.
test_that("custom_objective: verbose training produces non-garbled log output through the callback queue (case l)", {
  pool <- catboost.load_pool(features, label = label)
  out <- capture.output(
    model <- catboost.train(
      pool,
      params = modifyList(custom_objective_params(), list(logging_level = "Verbose", iterations = 5)),
      custom_objective = rmse_custom_objective
    )
  )
  expect_gt(length(out), 0L)
  expect_true(any(grepl("learn", out, ignore.case = TRUE)))
  expect_false(anyNA(catboost.predict(model, pool, prediction_type = "RawFormulaVal")))

  # Session survives: an unrelated subsequent call still works.
  model2 <- catboost.train(pool, params = c(common_params(), list(loss_function = "RMSE")))
  expect_false(anyNA(catboost.predict(model2, pool, prediction_type = "RawFormulaVal")))
})
