context("test_pool_structural.R")

# P3.4 differential test: build the R equivalent of tools/oracle's pinned
# Python catboost==1.2.10 Pool (tools/oracle/gen_pool_structural_fixture.py)
# and assert the observable catboost.pool.slice/catboost.pool.train_eval_split
# output matches what the fixture oracle script recorded
# (tests/fixtures/oracle/pool_structural.json), plus catboost.pool.save's
# binary output is readable back by the same pinned Python oracle. That last
# leg shells out to tools/oracle/read_quantized_pool.py via uv at test time:
# R can read the file itself (catboost.load_pool("quantized://...",
# column_description = "") works), but only an independently built, pinned
# catboost wheel proves the on-disk format rather than in-tree self-consistency.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_structural_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "pool_structural.json"),
  simplifyVector = TRUE
)

build_pool <- function(fixture) {
  inputs <- fixture$inputs
  catboost.load_pool(
    data.frame(num1 = inputs$num1, num2 = inputs$num2),
    label = as.double(inputs$label)
  )
}

build_categorical_pool <- function(fixture) {
  inputs <- fixture$inputs
  cat1 <- rep(c("a", "b"), length.out = length(inputs$num1))
  catboost.load_pool(
    data.frame(num1 = inputs$num1, cat1 = as.factor(cat1)),
    label = as.double(inputs$label)
  )
}

sort_rows <- function(mat) {
  mat[do.call(order, as.data.frame(mat)), , drop = FALSE]
}

test_that("pool slice: features/label match Python oracle for a contiguous numeric range", {
  pool <- build_pool(fixture)
  sliced <- catboost.pool.slice(pool, fixture$inputs$slice_offset, fixture$inputs$slice_size)

  expect_equal(dim(sliced), c(fixture$inputs$slice_size, dim(pool)[2]))
  expect_equal(unname(catboost.pool.get_features(sliced)), fixture$expected$slice$features)
  expect_equal(unname(catboost.pool.get_label(sliced)), as.double(fixture$expected$slice$label))
})

test_that("pool slice: group_id, baseline and feature names survive the slice", {
  # Python's Pool.slice() builds the sliced Pool with TDataProvider::GetSubset,
  # which carries every column through; catboost.pool.slice must not silently
  # drop group ids, baseline or feature names (which a dense round-trip does).
  inputs <- fixture$inputs
  n <- length(inputs$num1)
  group_id <- as.integer(rep(seq_len(n / 2), each = 2))
  baseline <- matrix(as.double(seq_len(n)) / 10, ncol = 1)
  pool <- catboost.load_pool(
    data.frame(num1 = inputs$num1, num2 = inputs$num2),
    label = as.double(inputs$label),
    group_id = group_id,
    baseline = baseline
  )
  offset <- inputs$slice_offset
  size <- inputs$slice_size
  rows <- seq.int(offset + 1, offset + size)

  sliced <- catboost.pool.slice(pool, offset, size)

  expect_equal(dim(sliced), c(size, 2))
  expect_equal(catboost.pool.get_feature_names(sliced), c("num1", "num2"))
  expect_equal(dimnames(sliced)[[2]], c("num1", "num2"))
  # baseline is stored as float32 in the Pool, hence the float tolerance
  expect_equal(catboost.pool.get_baseline(sliced), baseline[rows, , drop = FALSE],
               tolerance = 1e-6)
  expect_equal(
    catboost.pool.get_group_id_hash(sliced),
    catboost.pool.get_group_id_hash(pool)[rows]
  )
})

test_that("pool slice: categorical features are preserved, matching Python's Pool.slice()", {
  pool <- build_categorical_pool(fixture)
  sliced <- catboost.pool.slice(pool, fixture$inputs$slice_offset, fixture$inputs$slice_size)

  expect_equal(dim(sliced), c(fixture$inputs$slice_size, dim(pool)[2]))
  expect_equal(catboost.pool.get_cat_feature_indices(sliced),
               catboost.pool.get_cat_feature_indices(pool))
  expect_equal(catboost.pool.get_feature_names(sliced), c("num1", "cat1"))
})

test_that("pool train_eval_split: has_time=TRUE (no shuffle) matches Python oracle exactly", {
  pool <- build_pool(fixture)
  split <- catboost.pool.train_eval_split(
    pool,
    has_time = TRUE,
    is_classification = FALSE,
    eval_fraction = fixture$inputs$eval_fraction,
    save_eval_pool = TRUE
  )

  expect_equal(unname(catboost.pool.get_label(split$train)), as.double(fixture$expected$has_time_split$train_label))
  expect_equal(unname(catboost.pool.get_features(split$train)), fixture$expected$has_time_split$train_features)
  expect_equal(unname(catboost.pool.get_label(split$eval)), as.double(fixture$expected$has_time_split$eval_label))
  expect_equal(unname(catboost.pool.get_features(split$eval)), fixture$expected$has_time_split$eval_features)
})

test_that("pool train_eval_split: is_classification=TRUE (stratified, shuffled) matches Python oracle exactly", {
  pool <- build_pool(fixture)
  split <- catboost.pool.train_eval_split(
    pool,
    has_time = FALSE,
    is_classification = TRUE,
    eval_fraction = fixture$inputs$eval_fraction,
    save_eval_pool = TRUE
  )

  expect_equal(unname(catboost.pool.get_label(split$train)), as.double(fixture$expected$stratified_split$train_label))
  expect_equal(unname(catboost.pool.get_features(split$train)), fixture$expected$stratified_split$train_features)
  expect_equal(unname(catboost.pool.get_label(split$eval)), as.double(fixture$expected$stratified_split$eval_label))
  expect_equal(unname(catboost.pool.get_features(split$eval)), fixture$expected$stratified_split$eval_features)
})

test_that("pool train_eval_split: save_eval_pool=FALSE returns eval=NULL, train matches Python oracle", {
  pool <- build_pool(fixture)
  split <- catboost.pool.train_eval_split(
    pool,
    has_time = TRUE,
    is_classification = FALSE,
    eval_fraction = fixture$inputs$eval_fraction,
    save_eval_pool = FALSE
  )

  expect_null(split$eval)
  expect_equal(unname(catboost.pool.get_label(split$train)), as.double(fixture$expected$save_eval_pool_false$train_label))
})

test_that("pool save: saving an unquantized pool raises the same error as Python's Pool.save()", {
  pool <- build_pool(fixture)
  expect_true(fixture$expected$save_on_unquantized_raises)
  expect_error(catboost.pool.save(pool, tempfile()), regexp = "Pool is not quantized")
})

run_oracle <- function(repo_root, args) {
  system2(
    "uv",
    c("run", "--frozen", "--project", file.path(repo_root, "tools", "oracle"), "python3", args),
    stdout = TRUE, stderr = TRUE
  )
}

exit_status <- function(result) {
  status <- attr(result, "status")
  if (is.null(status)) 0L else as.integer(status)
}

test_that("pool save: quantized-pool binary output round-trips through the pinned Python oracle", {
  # Environment availability is decided ONCE, up front, by a probe that only
  # exercises uv plus the pinned catboost import and never touches the saved
  # pool. Any failure of the actual read below is therefore a real defect in
  # what catboost.pool.save() wrote, and fails the test instead of skipping.
  testthat::skip_if(Sys.which("uv") == "", "uv not available for round-trip check")
  repo_root <- testthat::test_path("..", "..")
  probe <- run_oracle(repo_root, c("-c", shQuote("import catboost")))
  testthat::skip_if(
    exit_status(probe) != 0,
    paste("pinned Python oracle environment unavailable:", paste(probe, collapse = "\n"))
  )

  pool <- build_pool(fixture)
  catboost.pool.quantize(pool, params = list(border_count = fixture$inputs$border_count))
  out <- tempfile(fileext = ".bin")
  on.exit(unlink(out))
  catboost.pool.save(pool, out)

  result <- run_oracle(
    repo_root,
    c(file.path(repo_root, "tools", "oracle", "read_quantized_pool.py"), out)
  )
  expect_equal(
    exit_status(result), 0L,
    info = paste("Python oracle could not read the saved pool:", paste(result, collapse = "\n"))
  )

  parsed <- jsonlite::fromJSON(paste(result, collapse = "\n"))
  expect_equal(parsed$num_row, fixture$expected$save_roundtrip$num_row)
  expect_equal(parsed$num_col, fixture$expected$save_roundtrip$num_col)
  expect_equal(as.double(parsed$label), as.double(fixture$expected$save_roundtrip$label))
})
