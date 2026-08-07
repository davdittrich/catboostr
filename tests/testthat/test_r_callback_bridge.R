context("test_r_callback_bridge.R")

# P6.2 (catboost-8z4.93): standalone smoke test of the fork-owned R-callback
# bridge (src/r_callback_bridge.h/.cpp), exercised directly through
# .Call("CatBoostRCallbackBridgeSelfTest_R", ...) -- NOT through
# catboost.train(), which is wired up later (catboost-8z4.89/.90/.94).
#
# The self-test entry point runs the bridge with plain std::threads standing in
# for CatBoost's TBB workers, so a bridge regression fails here regardless of
# what the training code does.
#
# Modes:
#   "roundtrip" -- M threads each marshal closure(i) onto R's main thread
#   "throw"     -- the background thread throws; must surface as an R error
#   "orphan"    -- the background thread dies while a worker still wants the
#                  queue; that worker's Call() must throw, not hang
#   "log"       -- log lines emitted off the main thread reach the console
#                  through the queue (the SetCustomLoggingFunction path)

bridge_self_test <- function(mode, closure = function(x) x * 2 + 1,
                             n_threads = 4L, n_items = 64L) {
  .Call("CatBoostRCallbackBridgeSelfTest_R", mode, closure,
        as.integer(n_threads), as.integer(n_items))
}

test_that("bridge round-trips R callback results from background threads", {
  n_items <- 400L
  res <- bridge_self_test("roundtrip", n_items = n_items)

  # (a) every request came back with the value R computed for it.
  expect_equal(res$results, (0:(n_items - 1)) * 2 + 1)

  # Instrumentation hook for catboost-8z4.91: more than one worker was doing
  # non-callback work at the same time, i.e. blocking on the queue does not
  # serialize the whole run.
  expect_gt(res$max_active_workers, 1L)
  # And no worker is left counted as active once Run() has returned.
  expect_equal(res$active_workers, 0L)

  expect_gt(res$interrupt_polls, 0L)
})

test_that("the drain loop polls for interrupts while the queue is idle", {
  # "idle": the background thread makes no requests at all for 400 ms, so the
  # drain loop is parked on an empty queue the whole time. An indefinite
  # condition-variable wait would poll R_CheckUserInterrupt() zero times; the
  # bounded 100 ms wait_for() polls roughly four times. This is the assertion
  # that Ctrl-C stays responsive during long non-callback training phases.
  res <- bridge_self_test("idle", n_items = 400L)
  expect_gte(res$interrupt_polls, 3L)
})

test_that("R-level errors inside a callback surface as an R error, not a hang", {
  expect_error(
    bridge_self_test("roundtrip", closure = function(x) stop("boom"), n_items = 8L),
    "error in R callback"
  )
})

test_that("the R session survives a failed callback (no PROTECT corruption)", {
  try(bridge_self_test("roundtrip", closure = function(x) stop("boom"), n_items = 8L),
      silent = TRUE)
  gc()
  res <- bridge_self_test("roundtrip", n_items = 16L)
  expect_equal(res$results, (0:15) * 2 + 1)
})

test_that("a background-thread exception is detected and surfaces as an R error", {
  started <- Sys.time()
  expect_error(bridge_self_test("throw", n_items = 16L),
               "boom from the background thread")
  # Bounded detection time, not a hang.
  expect_lt(as.numeric(difftime(Sys.time(), started, units = "secs")), 30)
})

test_that("a worker that wants the queue after the producer died throws, not hangs", {
  started <- Sys.time()
  err <- tryCatch(bridge_self_test("orphan", n_items = 0L),
                  error = function(e) conditionMessage(e))
  expect_match(err, "boom from the background thread")
  # 1 == the late Call() threw TRBridgeInterrupted instead of blocking.
  expect_match(err, "orphan_call_threw=1", fixed = TRUE)
  expect_lt(as.numeric(difftime(Sys.time(), started, units = "secs")), 30)
})

test_that("Call() from R's main thread errors instead of deadlocking", {
  # "reentrant": a queued action (which runs on the main thread) calls Call()
  # again. That inner request would be enqueued onto a queue whose only reader
  # is the now-blocked drain loop -- an unkillable hang. The guard in Call()
  # turns it into a diagnosable error. This test hangs forever if the guard is
  # removed, so its 30 s bound is the real assertion.
  started <- Sys.time()
  expect_error(bridge_self_test("reentrant", n_items = 4L),
               "invoked on R's main thread")
  expect_lt(as.numeric(difftime(Sys.time(), started, units = "secs")), 30)
})

test_that("log lines emitted off the main thread reach R's console", {
  n_items <- 8L
  out <- capture.output(res <- bridge_self_test("log", n_items = n_items))
  expect_equal(res$results, as.numeric(0:(n_items - 1)))
  expect_true(all(paste("bridge log line", 0:(n_items - 1)) %in% out))
})
