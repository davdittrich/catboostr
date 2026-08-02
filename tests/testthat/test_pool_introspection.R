context("test_pool_introspection.R")

# P3.2 differential test: build the R equivalent of tools/oracle's pinned
# Python catboost==1.2.10 Pool (tools/oracle/gen_pool_introspection_fixture.py)
# and assert the observable catboost.pool.get_cat_feature_indices/
# get_text_feature_indices/get_embedding_feature_indices/get_feature_names/
# set_feature_names/get_features/shape/num_col/num_row/is_empty output
# matches what the fixture oracle script recorded
# (tests/fixtures/oracle/pool_introspection.json).
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_introspection_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "pool_introspection.json"),
  simplifyVector = TRUE
)

build_mixed_pool <- function(fixture) {
  inputs <- fixture$inputs
  catboost.load_pool(
    data.frame(num1 = inputs$num1, cat1 = as.factor(inputs$cat1)),
    label = as.double(inputs$label)
  )
}

test_that("pool introspection: get_cat_feature_indices matches Python oracle", {
  pool <- build_mixed_pool(fixture)
  expect_equal(
    catboost.pool.get_cat_feature_indices(pool),
    as.integer(fixture$expected$before$cat_feature_indices)
  )
})

test_that("pool introspection: get_text_feature_indices matches Python oracle (always empty)", {
  pool <- build_mixed_pool(fixture)
  expect_equal(catboost.pool.get_text_feature_indices(pool), integer(0))
})

test_that("pool introspection: get_embedding_feature_indices matches Python oracle (always empty)", {
  pool <- build_mixed_pool(fixture)
  expect_equal(catboost.pool.get_embedding_feature_indices(pool), integer(0))
})

test_that("pool introspection: get_feature_names matches Python oracle", {
  pool <- build_mixed_pool(fixture)
  expect_equal(catboost.pool.get_feature_names(pool), fixture$expected$before$feature_names)
})

test_that("pool introspection: set_feature_names then get_feature_names matches Python oracle", {
  pool <- build_mixed_pool(fixture)
  catboost.pool.set_feature_names(pool, fixture$inputs$new_feature_names)
  expect_equal(catboost.pool.get_feature_names(pool), fixture$expected$after_rename$feature_names)
})

test_that("pool introspection: num_row/num_col/shape match Python oracle", {
  pool <- build_mixed_pool(fixture)
  expect_equal(catboost.pool.num_row(pool), fixture$expected$before$num_row)
  expect_equal(catboost.pool.num_col(pool), fixture$expected$before$num_col)
  expect_equal(catboost.pool.shape(pool), as.integer(fixture$expected$before$shape))
})

test_that("pool introspection: is_empty matches Python oracle on a non-empty pool", {
  pool <- build_mixed_pool(fixture)
  expect_equal(catboost.pool.is_empty(pool), fixture$expected$before$is_empty_)
})

test_that("pool introspection: is_empty/num_row match Python oracle on a 0-row pool", {
  pool <- catboost.load_pool(matrix(numeric(0), nrow = 0, ncol = 1))
  expect_equal(catboost.pool.is_empty(pool), fixture$expected$empty_is_empty_)
  expect_equal(catboost.pool.num_row(pool), fixture$expected$empty_num_row)
})

test_that("pool introspection: get_features matches Python oracle on an all-numeric pool", {
  inputs <- fixture$inputs
  n <- length(inputs$num1)
  numeric_only <- matrix(unlist(inputs$numeric_only), nrow = n, byrow = TRUE)
  pool <- catboost.load_pool(numeric_only, label = as.double(inputs$label),
                              feature_names = list("n1", "n2", "n3"))
  expected_features <- matrix(unlist(fixture$expected$numeric_features), nrow = n, byrow = TRUE)
  expect_equal(catboost.pool.get_features(pool), expected_features, tolerance = 1e-6, check.attributes = FALSE)
  expect_equal(catboost.pool.shape(pool), as.integer(fixture$expected$numeric_shape))
})

test_that("pool introspection: get_features errors on a Pool with categorical features", {
  pool <- build_mixed_pool(fixture)
  expect_error(catboost.pool.get_features(pool))
})
