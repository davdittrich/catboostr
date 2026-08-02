context("test_pool_structural.R")

# P3.4 differential test: build the R equivalent of tools/oracle's pinned
# Python catboost==1.2.10 Pool (tools/oracle/gen_pool_structural_fixture.py)
# and assert the observable catboost.pool.slice/catboost.pool.train_eval_split
# output matches what the fixture oracle script recorded
# (tests/fixtures/oracle/pool_structural.json), plus catboost.pool.save's
# binary output is readable back by the same pinned Python oracle (R's own
# catboost.load_pool has no "quantized://" scheme support, so that leg
# shells out to tools/oracle/read_quantized_pool.py via uv at test time).
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

test_that("pool slice: categorical feature present raises the native CB_ENSURE error", {
  pool <- build_categorical_pool(fixture)
  expect_error(
    catboost.pool.slice(pool, fixture$inputs$slice_offset, fixture$inputs$slice_size),
    regexp = "slicing datasets with categorical, text or embedding features is not supported",
    fixed = TRUE
  )
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

test_that("pool save: quantized-pool binary output round-trips through the pinned Python oracle", {
  testthat::skip_if(Sys.which("uv") == "", "uv not available for round-trip check")

  pool <- build_pool(fixture)
  catboost.pool.quantize(pool, params = list(border_count = fixture$inputs$border_count))
  out <- tempfile(fileext = ".bin")
  on.exit(unlink(out))
  catboost.pool.save(pool, out)

  repo_root <- testthat::test_path("..", "..")
  result <- system2(
    "uv",
    c("run", "--frozen", "--project", file.path(repo_root, "tools", "oracle"),
      "python3", file.path(repo_root, "tools", "oracle", "read_quantized_pool.py"), out),
    stdout = TRUE, stderr = TRUE
  )
  status <- attr(result, "status")
  if (!is.null(status) && status != 0) {
    testthat::skip(paste("Python oracle round-trip check failed to run:", paste(result, collapse = "\n")))
  }

  parsed <- jsonlite::fromJSON(paste(result, collapse = "\n"))
  expect_equal(parsed$num_row, fixture$expected$save_roundtrip$num_row)
  expect_equal(parsed$num_col, fixture$expected$save_roundtrip$num_col)
  expect_equal(as.double(parsed$label), as.double(fixture$expected$save_roundtrip$label))
})
