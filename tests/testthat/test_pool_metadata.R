context("test_pool_metadata.R")

# P3.1 differential test: build the R equivalent of tools/oracle's pinned
# Python catboost==1.2.10 Pool (tools/oracle/gen_pool_metadata_fixture.py),
# apply the same sequence of catboost.pool.set_* mutations, and assert the
# observable catboost.pool.get_*/has_label/num_pairs output matches what the
# fixture oracle script recorded (tests/fixtures/oracle/pool_metadata.json).
#
# set_group_weight/set_subgroup_id/set_pairs_weight/set_timestamp have no
# matching getter, so they are instead probed through an observable,
# comparable side effect that only that field can produce (see the fixture
# script's docstring for the vendor/catboost source references):
#   - group_weight -> perturbs trained-model predictions (predict_full).
#   - subgroup_id  -> perturbs the PFound metric (pfound).
#   - pairs_weight -> perturbs a PairLogit fit's predictions (predict_pairlogit).
#   - timestamp    -> perturbs a has_time=TRUE fit's predictions (predict_timestamp).
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_metadata_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "pool_metadata.json"),
  simplifyVector = TRUE
)

build_metadata_pool <- function(fixture) {
  inputs <- fixture$inputs
  n <- length(inputs$num1)

  pool <- catboost.load_pool(
    data.frame(num1 = inputs$num1, cat1 = as.factor(inputs$cat1)),
    label = as.double(inputs$label)
  )

  catboost.pool.set_weight(pool, inputs$weight)
  catboost.pool.set_baseline(pool, matrix(unlist(inputs$baseline), nrow = n, byrow = TRUE))
  catboost.pool.set_group_id(pool, as.integer(inputs$group_id))
  catboost.pool.set_group_weight(pool, inputs$group_weight)
  catboost.pool.set_subgroup_id(pool, as.integer(inputs$subgroup_id))
  # inputs$pairs is already an (N x 2) matrix -- jsonlite::fromJSON with
  # simplifyVector = TRUE auto-simplifies the JSON array-of-pairs into one.
  # (unlist()-then-reshape here was a bug: unlist() flattens a matrix
  # column-major, and re-filling byrow = TRUE then reads those column-major
  # values back out row-major, silently transposing/scrambling pair
  # identities -- e.g. (0,1),(1,2),(2,3) became (0,1),(2,1),(2,3).)
  catboost.pool.set_pairs(pool, inputs$pairs)
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

test_that("pool metadata: set_group_weight/set_baseline/set_weight/set_group_id are observable through trained predictions matching Python oracle", {
  pool <- build_metadata_pool(fixture)
  model <- catboost.train(pool, params = list(
    loss_function = "Logloss", iterations = 5, depth = 2,
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))
  prediction <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(as.double(prediction), fixture$expected$predict_full, tolerance = 1e-6)
})

test_that("pool metadata: set_subgroup_id is observable through the PFound metric matching Python oracle", {
  pool <- build_metadata_pool(fixture)
  model <- catboost.train(pool, params = list(
    loss_function = "Logloss", iterations = 5, depth = 2,
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))
  pfound <- catboost.eval_metrics(model, pool, "PFound")[["PFound"]]
  expect_equal(pfound[length(pfound)], fixture$expected$pfound, tolerance = 1e-6)
})

test_that("pool metadata: set_pairs_weight is observable through a PairLogit fit's predictions matching Python oracle", {
  pool <- build_metadata_pool(fixture)
  model <- catboost.train(pool, params = list(
    loss_function = "PairLogit", iterations = 5, depth = 2,
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))
  prediction <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(as.double(prediction), fixture$expected$predict_pairlogit, tolerance = 1e-6)
})

test_that("pool metadata: set_timestamp is observable through a has_time fit's predictions matching Python oracle", {
  inputs <- fixture$inputs
  timestamp_pool <- catboost.load_pool(
    data.frame(num1 = inputs$num1),
    label = as.double(inputs$label)
  )
  catboost.pool.set_timestamp(timestamp_pool, inputs$timestamp)
  model <- catboost.train(timestamp_pool, params = list(
    loss_function = "Logloss", iterations = 5, depth = 2, has_time = TRUE,
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))
  prediction <- catboost.predict(model, timestamp_pool, prediction_type = "RawFormulaVal")
  expect_equal(as.double(prediction), fixture$expected$predict_timestamp, tolerance = 1e-6)
})

# catboost-8z4.49: timestamp= constructor argument parity vs. the
# already-tested catboost.pool.set_timestamp() setter (catboost-8z4.38).
test_that("load_pool: timestamp= constructor argument matches catboost.pool.set_timestamp (catboost-8z4.49)", {
  inputs <- fixture$inputs
  pool_via_constructor <- catboostr:::catboost.load_pool(
    data.frame(num1 = inputs$num1), label = as.double(inputs$label),
    timestamp = inputs$timestamp
  )
  pool_via_setter <- catboostr:::catboost.load_pool(
    data.frame(num1 = inputs$num1), label = as.double(inputs$label)
  )
  catboostr:::catboost.pool.set_timestamp(pool_via_setter, inputs$timestamp)
  params <- list(
    loss_function = "Logloss", iterations = 5, depth = 2, has_time = TRUE,
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  )
  model_a <- catboostr:::catboost.train(pool_via_constructor, params = params)
  model_b <- catboostr:::catboost.train(pool_via_setter, params = params)
  expect_equal(
    catboostr:::catboost.predict(model_a, pool_via_constructor),
    catboostr:::catboost.predict(model_b, pool_via_setter)
  )
})

test_that("load_pool: timestamp= length must match object count (catboost-8z4.49)", {
  expect_error(
    catboostr:::catboost.load_pool(data = matrix(1:20, ncol = 2), label = 1:10, timestamp = 1:5),
    regexp = "timestamp"
  )
})

test_that("load_pool: timestamp= must be numeric, not character/logical (catboost-8z4.49)", {
  expect_error(
    catboostr:::catboost.load_pool(data = matrix(1:20, ncol = 2), label = 1:10, timestamp = letters[1:10]),
    regexp = "timestamp"
  )
  expect_error(
    catboostr:::catboost.load_pool(data = matrix(1:20, ncol = 2), label = 1:10, timestamp = rep(TRUE, 10)),
    regexp = "timestamp"
  )
})

test_that("from_matrix: feature_tags= is rejected, not silently dropped (catboost-8z4.49)", {
  expect_error(
    catboostr:::catboost.load_pool(data = matrix(1:20, ncol = 2), label = 1:10, feature_tags = list(a = 1)),
    regexp = "feature_tags"
  )
  expect_error(
    catboostr:::catboost.from_matrix(matrix(1:20, ncol = 2), label = 1:10, feature_tags = list(a = 1)),
    regexp = "feature_tags"
  )
})
