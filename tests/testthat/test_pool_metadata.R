context("test_pool_metadata.R")

# P3.1 differential test: build the R equivalent of tools/oracle's pinned
# Python catboost==1.2.10 Pool (tools/oracle/gen_pool_metadata_fixture.py),
# apply the same sequence of catboost.pool.set_* mutations, and compare every
# observable catboost.pool.get_*/has_label/num_pairs output against the
# fixture the oracle script recorded (tests/fixtures/oracle/pool_metadata.json).
#
# Regenerate the fixture with:
#   uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_metadata_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "pool_metadata.json"),
  simplifyVector = TRUE
)

build_metadata_pool <- function(fixture) {
  inputs <- fixture$inputs
  n <- length(inputs$num1)

  pool <- catboost.load_pool(
    data = data.frame(num1 = inputs$num1, cat1 = as.factor(inputs$cat1)),
    label = as.double(inputs$label)
  )

  catboost.pool.set_weight(pool, inputs$weight)
  catboost.pool.set_baseline(pool, matrix(unlist(inputs$baseline), nrow = n, byrow = TRUE))
  catboost.pool.set_group_id(pool, as.integer(inputs$group_id))
  catboost.pool.set_group_weight(pool, inputs$group_weight)
  catboost.pool.set_subgroup_id(pool, as.integer(inputs$subgroup_id))
  catboost.pool.set_pairs(pool, matrix(unlist(inputs$pairs), ncol = 2, byrow = TRUE))
  catboost.pool.set_pairs_weight(pool, inputs$pairs_weight)
  catboost.pool.set_timestamp(pool, inputs$timestamp)

  return(pool)
}

test_that("pool metadata: has_label matches Python oracle", {
  pool <- build_metadata_pool(fixture)
  expect_equal(catboost.pool.has_label(pool), fixture$expected$has_label)
})

test_that("pool metadata: get_label matches Python oracle", {
  pool <- build_metadata_pool(fixture)
  expect_equal(as.double(catboost.pool.get_label(pool)), fixture$expected$label, tolerance = 1e-6)
})

test_that("pool metadata: get_weight matches Python oracle after set_weight", {
  pool <- build_metadata_pool(fixture)
  expect_equal(catboost.pool.get_weight(pool), fixture$expected$weight, tolerance = 1e-6)
})

test_that("pool metadata: get_baseline matches Python oracle after set_baseline", {
  pool <- build_metadata_pool(fixture)
  expected_baseline <- matrix(unlist(fixture$expected$baseline), ncol = 1, byrow = TRUE)
  expect_equal(catboost.pool.get_baseline(pool), expected_baseline, tolerance = 1e-6, check.attributes = FALSE)
})

test_that("pool metadata: get_group_id_hash matches Python oracle after set_group_id", {
  pool <- build_metadata_pool(fixture)
  expect_equal(catboost.pool.get_group_id_hash(pool), fixture$expected$group_id_hash)
})

test_that("pool metadata: num_pairs matches Python oracle after set_pairs/set_pairs_weight", {
  pool <- build_metadata_pool(fixture)
  expect_equal(catboost.pool.num_pairs(pool), fixture$expected$num_pairs)
})

test_that("pool metadata: has_label is FALSE and get_group_id_hash is NULL on an unlabeled, ungrouped pool", {
  pool <- catboost.load_pool(data = data.frame(num1 = c(1, 2, 3)))
  expect_false(catboost.pool.has_label(pool))
  expect_null(catboost.pool.get_group_id_hash(pool))
  expect_equal(catboost.pool.num_pairs(pool), 0)
})
