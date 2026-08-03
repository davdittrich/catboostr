context("test_pool_quantization.R")

# P3.3 differential test: build the R equivalent of tools/oracle's pinned
# Python catboost==1.2.10 Pool (tools/oracle/gen_pool_quantization_fixture.py)
# and assert the observable catboost.pool.is_quantized/catboost.pool.quantize/
# catboost.pool.save_quantization_borders output matches what the fixture
# oracle script recorded (tests/fixtures/oracle/pool_quantization.json).
#
# quantize() itself has no getter that returns the quantized bin data as an
# R-visible structure, so the only byte-for-byte comparable oracle output is
# the borders file save_quantization_borders() writes -- compared verbatim
# against the fixture's recorded borders_text. is_quantized() before/after
# quantize() and quantize()-on-an-already-quantized-Pool erroring are the
# other two observable behaviors.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_quantization_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "pool_quantization.json"),
  simplifyVector = TRUE
)

build_pool <- function(fixture) {
  inputs <- fixture$inputs
  catboost.load_pool(
    data.frame(num1 = inputs$num1, num2 = inputs$num2),
    label = as.double(inputs$label)
  )
}

test_that("pool quantization: is_quantized is FALSE before quantize, matches Python oracle", {
  pool <- build_pool(fixture)
  expect_equal(catboost.pool.is_quantized(pool), fixture$expected$is_quantized_before)
})

test_that("pool quantization: is_quantized is TRUE after quantize, matches Python oracle", {
  pool <- build_pool(fixture)
  catboost.pool.quantize(pool, params = list(border_count = fixture$inputs$border_count))
  expect_equal(catboost.pool.is_quantized(pool), fixture$expected$is_quantized_after)
})

test_that("pool quantization: save_quantization_borders output matches Python oracle byte-for-byte", {
  pool <- build_pool(fixture)
  catboost.pool.quantize(pool, params = list(border_count = fixture$inputs$border_count))
  out <- tempfile()
  on.exit(unlink(out))
  catboost.pool.save_quantization_borders(pool, out)
  borders_text <- readChar(out, file.info(out)$size)
  expect_equal(borders_text, fixture$expected$borders_text)
})

test_that("pool quantization: quantizing an already-quantized pool errors, matches Python oracle", {
  pool <- build_pool(fixture)
  catboost.pool.quantize(pool, params = list(border_count = fixture$inputs$border_count))
  if (fixture$expected$already_quantized_raises) {
    expect_error(catboost.pool.quantize(pool, params = list(border_count = fixture$inputs$border_count)))
  } else {
    expect_silent(catboost.pool.quantize(pool, params = list(border_count = fixture$inputs$border_count)))
  }
})

test_that("pool quantization: save_quantization_borders on an unquantized pool errors, matches Python oracle", {
  pool <- build_pool(fixture)
  out <- tempfile()
  on.exit(unlink(out))
  if (fixture$expected$save_on_unquantized_raises) {
    expect_error(catboost.pool.save_quantization_borders(pool, out))
  } else {
    expect_silent(catboost.pool.save_quantization_borders(pool, out))
  }
})
