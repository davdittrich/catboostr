context("test_distributed_training.R")

# P8.2 (catboost-8z4.103): differential test -- R's distributed-training path
# (node_type = "Master" / file_with_hosts / node_port) against a REAL,
# independently-built CatBoost CLI binary (tools/oracle/cli/bin/catboost-v1.2.10)
# acting as the worker, not R's own catboost.run_worker() (that binding is
# catboost-8z4.102's test_run_worker.R -- a startup smoke test against R
# itself, not a real master/worker round-trip).
#
# Two assertions against the SAME live CLI worker:
#   (a) catboost.train(..., node_type = "Master") -- a real master/worker fit,
#       compared against a single-host fit on the same data (P8.4,
#       catboost-8z4.105 made this path reach the training engine; before that
#       it was blocked by a vendored CB_ENSURE and this assertion was an
#       expect_error() documenting the block).
#   (b) catboost.select_features(..., node_type = "Master") against the same
#       distributed params -- this path does NOT go through the blocked
#       free function (NCB::SelectFeatures calls TCPUModelTrainer::TrainModel
#       directly), and passes for real against the live CLI worker.
#
# Worker process model: CatBoost's CLI `run-worker` mode (mode_run_worker.cpp)
# takes no port-availability probe of its own, so free-port selection follows
# catboost-8z4.102's test_run_worker.R pattern exactly: base R's
# socketConnection(port = 0, server = TRUE) cannot be used to pre-probe a free
# port (it blocks inside accept() as part of the call establishing the
# connection, not just at read/write time -- catboost-8z4.102 hit this during
# development). Instead: pick a random high port, spawn the process on it,
# and poll readiness with a short-timeout CLIENT connection attempt
# (server = FALSE, which does not block indefinitely); retry with a fresh
# port up to 3 times.
#
# file_with_hosts format: one "host:port" per line (confirmed against
# vendor/catboost/library/cpp/par/par_host.cpp:34-44 TRootEnvironment ctor,
# which reads the file line-by-line and only falls back to the master's own
# defaultSlavePort -- itself just a placeholder value, GetUnusedNodePort() ==
# 0 -- when a line omits ":port"; system_options.cpp:54 IsMaster() /
# system_options.cpp:62 IsWorker() confirm node_type = "Master" plus a
# non-empty file_with_hosts is what selects the real distributed master path
# rather than the single-host default).
#
# The master's own node_port (systemOptions.NodePort, master.cpp:39-45) is a
# second, independent port from the worker's --node-port: it is where this R
# process's own par-framework listener binds. Its default, GetUnusedNodePort()
# == 0 (system_options.h:33), is passed straight through to NPar::RunMaster
# (master.cpp:47-52) unchanged when node_port is omitted from params, and
# par_network.cpp:72 (`if (port == 0)`) treats that as "bind an OS-assigned
# free port" -- collision-free by construction, unlike picking an explicit
# port ourselves and hoping nothing else holds it. So node_port is omitted
# from the master-side params below rather than set explicitly: an earlier
# version of this test picked an explicit random master port with a
# retry-on-failure wrapper, but a real collision on that port crashes the
# par library inside a C++ assert (a hard process abort, not a catchable R
# condition), so the retry logic could never actually run. Omitting node_port
# removes the collision class entirely instead of working around it.

CLI_BIN <- testthat::test_path("..", "..", "tools", "oracle", "cli", "bin", "catboost-v1.2.10")

testthat::skip_if_not(
  file.exists(CLI_BIN) && file.access(CLI_BIN, mode = 1) == 0,
  paste0(
    "real CatBoost CLI binary not present/executable at ", CLI_BIN,
    " -- run tools/oracle/cli/acquire.sh (per-checkout, gitignored artifact)"
  )
)

worker_is_listening <- function(port) {
  con <- tryCatch(
    socketConnection(host = "127.0.0.1", port = port, server = FALSE, timeout = 1),
    error = function(e) NULL
  )
  if (is.null(con)) return(FALSE)
  close(con)
  TRUE
}

# Spawns the real CLI binary as `run-worker --node-port <port>` in the
# background. system2(wait = FALSE) does not return a PID, so the worker's
# PID is recovered after the fact via `pgrep -f`, matched on the
# --node-port value (unique per spawn attempt -- a fresh random port every
# retry -- so this cannot cross-match a leftover process from a prior
# attempt). If the process hasn't shown up under pgrep by the time the
# polling window below gives up, sweep for it once more by the same
# pattern and kill it rather than leaking it to the caller.
spawn_cli_worker <- function(port) {
  logfile <- tempfile()
  system2(CLI_BIN, c("run-worker", "--node-port", port), wait = FALSE,
          stdout = logfile, stderr = logfile)

  pattern <- sprintf("run-worker --node-port %d$", port)
  pid <- NA_integer_
  for (i in 1:50) {
    found <- suppressWarnings(
      system2("pgrep", c("-f", shQuote(pattern)), stdout = TRUE, stderr = FALSE)
    )
    if (length(found) >= 1 && nzchar(found[1])) {
      pid <- suppressWarnings(as.integer(found[1]))
      if (!is.na(pid)) break
    }
    Sys.sleep(0.1)
  }
  if (is.na(pid)) {
    # ponytail: covers the race where the process appears just after the
    # last pgrep check above -- sweep once more and kill anything matching
    # this attempt's unique port pattern so it isn't leaked to the caller.
    system2("pkill", c("-f", shQuote(pattern)), stdout = FALSE, stderr = FALSE)
  }
  pid
}

# Starts a live CLI worker on a free port (retrying up to 3 times) and
# writes the one-line file_with_hosts it implies. Returns a list with
# $pid/$port/$hosts_path, or stops with an informative error after 3 failed
# attempts.
start_cli_worker <- function() {
  for (attempt in 1:3) {
    port <- sample(20000:60000, 1)
    pid <- spawn_cli_worker(port)
    if (is.na(pid)) next

    listening <- FALSE
    for (i in 1:20) {
      if (worker_is_listening(port)) {
        listening <- TRUE
        break
      }
      Sys.sleep(0.25)
    }
    if (listening) {
      hosts_path <- tempfile()
      writeLines(sprintf("127.0.0.1:%d", port), hosts_path)
      return(list(pid = pid, port = port, hosts_path = hosts_path))
    }
    tools::pskill(pid)
  }
  stop("CLI worker never became connectable on any of 3 attempted ports")
}

kill_cli_worker <- function(worker) {
  # SIGTERM (tools::pskill's default) was observed not to terminate a live
  # run-worker process promptly (still alive 2+ seconds later in manual
  # testing); SIGKILL does, reliably and immediately. Teardown must be
  # unconditional (DoD: no orphan process survives an assertion failure),
  # so SIGKILL is used directly rather than a SIGTERM-then-wait-then-SIGKILL
  # escalation that would just add latency for no observed benefit here.
  if (!is.null(worker) && !is.na(worker$pid)) tools::pskill(worker$pid, signal = tools::SIGKILL)
}

set.seed(20260808)
n <- 200
features <- data.frame(
  x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), x4 = rnorm(n), x5 = rnorm(n)
)
label <- with(features, 1.5 * x1 - 2 * x2 + 0.5 * x3 + rnorm(n, sd = 0.2))

base_train_params <- list(
  loss_function = "RMSE",
  iterations = 30,
  depth = 4,
  random_seed = 42,
  thread_count = 1,
  allow_writing_files = FALSE,
  logging_level = "Silent"
)

test_that("distributed catboost.train (Master + real CLI worker) matches single-host training", {
  # P8.4 (catboost-8z4.105): CatBoostFit_R now routes node_type != "SingleHost"
  # to TrainModelDistributed (src/catboostr.cpp), which drives
  # TCPUModelTrainer::TrainModel directly instead of the plain-JSON
  # TrainModel() free function whose CB_ENSURE
  # (vendor/catboost/catboost/libs/train_lib/train_model.cpp:1632) forbids
  # distributed training. Single-host calls still go through that free
  # function unchanged -- result_single below is produced by the unchanged
  # path and is the reference the distributed fit is compared against.
  pool <- catboost.load_pool(features, label = label)

  model_single <- catboost.train(pool, params = base_train_params)
  pred_single <- catboost.predict(model_single, pool)

  worker <- NULL
  on.exit(kill_cli_worker(worker), add = TRUE)
  worker <- start_cli_worker()

  dist_params <- c(base_train_params, list(
    node_type = "Master",
    file_with_hosts = worker$hosts_path
  ))

  model_dist <- catboost.train(pool, params = dist_params)
  pred_dist <- catboost.predict(model_dist, pool)

  kill_cli_worker(worker)
  worker <- NULL

  # Tolerance justification. Distributed training is NOT a
  # bit-for-bit-reproducible variant of single-host training: the master
  # ships binarized features to the workers and each split's histograms are
  # computed per shard (TCPUModelTrainer::TrainModel's !IsSingleHost()
  # branch), so floating-point summation order -- and hence the chosen splits
  # and leaf values -- legitimately differ from the single-host run even with
  # an identical random_seed. Exact-equality (or a tight per-prediction
  # epsilon) is therefore the wrong gate here, the same reason assertion (b)
  # below compares select_features structurally.
  #
  # What IS asserted is that the distributed fit really trained the same
  # model family on the same data: the same number of trees, predictions that
  # track the single-host model's almost perfectly, and a train RMSE within a
  # margin of it. Margins are set from the observed run (max abs prediction
  # difference 0.63 == 0.23 * sd(label); correlation 0.9966; RMSE 0.2323
  # distributed vs 0.2146 single-host == +8.2%) with ~3x headroom, so a
  # genuine regression -- a worker that silently contributes nothing, a
  # dropped iteration, an untrained model -- fails, while shard-order noise
  # does not. sd(label) is ~2.70, so both RMSEs are ~8% of it: both models
  # genuinely learned the signal.
  expect_equal(model_dist$tree_count, model_single$tree_count)
  expect_gt(cor(pred_dist, pred_single), 0.99)
  expect_lt(max(abs(pred_dist - pred_single)), 0.7 * sd(label))

  rmse <- function(pred) sqrt(mean((pred - label)^2))
  expect_lt(rmse(pred_dist), 1.25 * rmse(pred_single))
})

test_that("distributed catboost.select_features (Master + real CLI worker) is structurally consistent with single-host", {
  pool <- catboost.load_pool(features, label = label)

  select_params <- list(
    loss_function = "RMSE",
    iterations = 15,
    depth = 3,
    random_seed = 42,
    thread_count = 1,
    allow_writing_files = FALSE,
    logging_level = "Silent"
  )

  result_single <- catboost.select_features(
    pool,
    features_for_select = c(0, 1, 2, 3, 4),
    num_features_to_select = 3,
    params = select_params,
    steps = 1,
    train_final_model = FALSE
  )

  worker <- NULL
  on.exit(kill_cli_worker(worker), add = TRUE)
  worker <- start_cli_worker()

  dist_params <- c(select_params, list(
    node_type = "Master",
    file_with_hosts = worker$hosts_path
  ))

  result_dist <- catboost.select_features(
    pool,
    features_for_select = c(0, 1, 2, 3, 4),
    num_features_to_select = 3,
    params = dist_params,
    steps = 1,
    train_final_model = FALSE
  )

  kill_cli_worker(worker)
  worker <- NULL

  # Structural, not numeric, comparison: distributed feature elimination goes
  # through the same par-framework sharding as assertion (a)'s training call,
  # so the exact elimination order/SHAP strengths are not asserted to match
  # single-host bit-for-bit -- only that the distributed run completed
  # without error and produced a result of the same shape (ticket explicitly
  # leaves the exact comparison method to the implementer).
  expect_null(result_dist$model)
  expect_setequal(names(result_dist), names(result_single))
  expect_equal(length(result_dist$selected_features), 3)
  expect_equal(
    sort(c(result_dist$selected_features, result_dist$eliminated_features)),
    c(0, 1, 2, 3, 4)
  )
  expect_type(result_dist$loss_graph, typeof(result_single$loss_graph))
})
