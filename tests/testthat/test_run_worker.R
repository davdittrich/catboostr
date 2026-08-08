context("test_run_worker.R")

# P8.1 (catboost-8z4.102) smoke test: catboost.run_worker() is a blocking
# call (RunWorker/NPar::RunSlave never returns during normal distributed-
# training operation, same as the CLI's `run-worker` mode). This is a
# startup smoke test only -- it confirms the worker process starts and can
# be observed running, not a real master/worker round-trip (that's
# catboost-8z4.103's differential test against the real CLI binary).
#
# Free-port selection: base R's socketConnection(server = TRUE) blocks
# until a client connects (it is not a "bind and return" primitive), so it
# cannot be used to pre-probe for a free port. Instead: pick a random
# high port, spawn the worker there, and poll readiness with a short-
# timeout CLIENT connection attempt (server = FALSE, which does not
# block indefinitely). Retry with a fresh port up to 3 times if the
# worker never becomes connectable (covers the rare case another
# process already holds the chosen port).

spawn_worker <- function(port) {
  pidfile <- tempfile()
  rscript <- file.path(R.home("bin"), "Rscript")
  # system2(wait = FALSE) does not return a PID, and Rscript -e runs a
  # single process (no extra fork), so have the worker report its own PID
  # as its first action before blocking in catboost.run_worker().
  worker_expr <- sprintf(
    "writeLines(as.character(Sys.getpid()), %s); catboostr::catboost.run_worker(node_port = %dL, thread_count = 1L)",
    shQuote(pidfile), port
  )
  system2(rscript, c("-e", shQuote(worker_expr)), wait = FALSE,
          stdout = FALSE, stderr = FALSE)

  pid <- NA_integer_
  for (i in 1:50) {
    if (file.exists(pidfile) && file.info(pidfile)$size > 0) {
      pid <- suppressWarnings(as.integer(readLines(pidfile, n = 1)))
      if (!is.na(pid)) break
    }
    Sys.sleep(0.1)
  }
  pid
}

worker_is_listening <- function(port) {
  con <- tryCatch(
    socketConnection(host = "127.0.0.1", port = port, server = FALSE,
                      timeout = 1),
    error = function(e) NULL
  )
  if (is.null(con)) return(FALSE)
  close(con)
  TRUE
}

test_that("catboost.run_worker starts and can be observed running, then killed", {
  pid <- NA_integer_
  on.exit({
    if (!is.na(pid)) tools::pskill(pid)
  }, add = TRUE)

  listening <- FALSE
  for (attempt in 1:3) {
    port <- sample(20000:60000, 1)
    pid <- spawn_worker(port)
    if (is.na(pid)) next

    for (i in 1:20) {
      if (worker_is_listening(port)) {
        listening <- TRUE
        break
      }
      Sys.sleep(0.25)
    }
    if (listening) break

    tools::pskill(pid)
    pid <- NA_integer_
  }

  expect_false(is.na(pid), info = "worker process never started (3 attempts)")
  expect_true(listening,
              info = "worker never became connectable on its port (3 attempts)")

  # tools::pskill(pid, signal = 0) sends no signal, just checks the process
  # exists and is killable by this user -- confirms the worker is actually
  # running, not just that the shell wrapper started.
  expect_true(tools::pskill(pid, signal = 0),
              info = "worker process not observed running")

  expect_true(tools::pskill(pid), info = "failed to kill worker process")
})
