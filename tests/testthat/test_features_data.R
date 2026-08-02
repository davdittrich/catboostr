context("test_features_data.R")

# P3.5 differential test: build the R equivalent of tools/oracle's pinned
# Python catboost==1.2.10 FeaturesData (tools/oracle/gen_features_data_fixture.py)
# and assert the observable catboost.features_data.get_object_count/
# get_num_feature_count/get_cat_feature_count/get_feature_count/
# get_feature_names output -- plus a Pool built via
# catboost.load_pool(data = catboost.FeaturesData(...)) -- matches what the
# fixture oracle script recorded (tests/fixtures/oracle/features_data.json).
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_features_data_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "features_data.json"),
  simplifyVector = TRUE
)

build_num_matrix <- function(inputs) {
  matrix(c(inputs$num1, inputs$num2), nrow = length(inputs$num1), ncol = 2)
}

build_cat_matrix <- function(inputs) {
  matrix(c(inputs$cat1, inputs$cat2), nrow = length(inputs$cat1), ncol = 2)
}

test_that("FeaturesData: mixed num+cat matches Python oracle", {
  inputs <- fixture$inputs
  fd <- catboost.FeaturesData(
    num_feature_data = build_num_matrix(inputs),
    cat_feature_data = build_cat_matrix(inputs),
    num_feature_names = as.list(inputs$num_feature_names),
    cat_feature_names = as.list(inputs$cat_feature_names)
  )
  expected <- fixture$expected$mixed
  expect_equal(catboost.features_data.get_object_count(fd), expected$get_object_count)
  expect_equal(catboost.features_data.get_num_feature_count(fd), expected$get_num_feature_count)
  expect_equal(catboost.features_data.get_cat_feature_count(fd), expected$get_cat_feature_count)
  expect_equal(catboost.features_data.get_feature_count(fd), expected$get_feature_count)
  expect_equal(catboost.features_data.get_feature_names(fd), expected$get_feature_names)
})

test_that("FeaturesData: num-only with default names matches Python oracle", {
  inputs <- fixture$inputs
  fd <- catboost.FeaturesData(num_feature_data = build_num_matrix(inputs))
  expected <- fixture$expected$num_only
  expect_equal(catboost.features_data.get_object_count(fd), expected$get_object_count)
  expect_equal(catboost.features_data.get_num_feature_count(fd), expected$get_num_feature_count)
  expect_equal(catboost.features_data.get_cat_feature_count(fd), expected$get_cat_feature_count)
  expect_equal(catboost.features_data.get_feature_count(fd), expected$get_feature_count)
  expect_equal(catboost.features_data.get_feature_names(fd), expected$get_feature_names)
})

test_that("FeaturesData: cat-only with default names matches Python oracle", {
  inputs <- fixture$inputs
  fd <- catboost.FeaturesData(cat_feature_data = build_cat_matrix(inputs))
  expected <- fixture$expected$cat_only
  expect_equal(catboost.features_data.get_object_count(fd), expected$get_object_count)
  expect_equal(catboost.features_data.get_num_feature_count(fd), expected$get_num_feature_count)
  expect_equal(catboost.features_data.get_cat_feature_count(fd), expected$get_cat_feature_count)
  expect_equal(catboost.features_data.get_feature_count(fd), expected$get_feature_count)
  expect_equal(catboost.features_data.get_feature_names(fd), expected$get_feature_names)
})

test_that("FeaturesData: constructor rejects both parts NULL", {
  expect_error(catboost.FeaturesData())
})

test_that("FeaturesData: constructor rejects feature_names without matching feature_data", {
  inputs <- fixture$inputs
  expect_error(catboost.FeaturesData(cat_feature_data = build_cat_matrix(inputs),
                                     num_feature_names = as.list(inputs$num_feature_names)))
})

test_that("FeaturesData: constructor rejects mismatched object counts", {
  inputs <- fixture$inputs
  num_data <- build_num_matrix(inputs)
  cat_data <- build_cat_matrix(inputs)[1:4, , drop = FALSE]
  expect_error(catboost.FeaturesData(num_feature_data = num_data, cat_feature_data = cat_data))
})

test_that("FeaturesData wired into catboost.load_pool matches Python oracle Pool", {
  inputs <- fixture$inputs
  fd <- catboost.FeaturesData(
    num_feature_data = build_num_matrix(inputs),
    cat_feature_data = build_cat_matrix(inputs),
    num_feature_names = as.list(inputs$num_feature_names),
    cat_feature_names = as.list(inputs$cat_feature_names)
  )
  pool <- catboost.load_pool(fd, label = as.double(inputs$label))
  expected <- fixture$expected$pool_from_features_data

  expect_equal(catboost.pool.get_cat_feature_indices(pool), as.integer(expected$cat_feature_indices))
  expect_equal(catboost.pool.get_feature_names(pool), expected$feature_names)
  expect_equal(catboost.pool.num_row(pool), expected$num_row)
  expect_equal(catboost.pool.num_col(pool), expected$num_col)
  expect_equal(as.double(catboost.pool.get_label(pool)), expected$get_label, tolerance = 1e-6)
})

test_that("FeaturesData wired into catboost.from_matrix matches Python oracle Pool", {
  inputs <- fixture$inputs
  fd <- catboost.FeaturesData(
    num_feature_data = build_num_matrix(inputs),
    cat_feature_data = build_cat_matrix(inputs),
    num_feature_names = as.list(inputs$num_feature_names),
    cat_feature_names = as.list(inputs$cat_feature_names)
  )
  pool <- catboostr:::catboost.from_matrix(fd, label = as.double(inputs$label))
  expected <- fixture$expected$pool_from_features_data

  expect_equal(catboost.pool.get_cat_feature_indices(pool), as.integer(expected$cat_feature_indices))
  expect_equal(catboost.pool.get_feature_names(pool), expected$feature_names)
})

test_that("catboost.load_pool errors when cat_features passed alongside FeaturesData", {
  inputs <- fixture$inputs
  fd <- catboost.FeaturesData(num_feature_data = build_num_matrix(inputs))
  expect_error(catboost.load_pool(fd, cat_features = c(0L)))
})
