context("test_select_features_bytags.R")

# P10.G (catboost-8z4.121) -- catboost.select_features(grouping = "ByTags")
# and catboost.load_pool(feature_tags = ...).
#
# Closes the 3 red parity rows catboost-8z4.113/P9.1 filed as bucket-C
# ("genuine new-code gap"): catboost.EFeaturesSelectionGrouping (class) and
# its .ByTags/.Individual enum-value rows. .Individual was already exercised
# end-to-end by test_select_features.R; this file is what actually proves
# .ByTags (and the class as a whole, since it now dispatches both members)
# against the Python oracle, bit-for-bit like its sibling file, because
# grouping = "ByTags" reaches the exact same native NCB::SelectFeatures entry
# point as grouping = "Individual" -- only TFeaturesSelectOptions.Grouping
# differs -- once the R-side Pool actually carries feature tags
# (catboost.load_pool's feature_tags argument, new in this ticket).
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_select_features_bytags_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "select_features_bytags.json"),
  simplifyVector = TRUE
)

feature_tags <- lapply(fixture$feature_tags, function(tag) {
  list(features = as.integer(tag$features), cost = tag$cost)
})

learn_pool <- catboost.load_pool(
  fixture$inputs$learn_features, label = fixture$inputs$learn_label, feature_tags = feature_tags
)
test_pool <- catboost.load_pool(
  fixture$inputs$test_features, label = fixture$inputs$test_label, feature_tags = feature_tags
)

base_params <- function() {
  p <- fixture$base_params
  p$verbose <- NULL
  p$logging_level <- "Silent"
  return(p)
}

expect_summary_matches <- function(actual, expected) {
  expect_equal(actual$selected_features, expected$selected_features, check.attributes = FALSE)
  expect_equal(actual$eliminated_features, expected$eliminated_features, check.attributes = FALSE)
  expect_equal(actual$selected_features_tags, expected$selected_features_tags, check.attributes = FALSE)
  expect_equal(actual$eliminated_features_tags, expected$eliminated_features_tags, check.attributes = FALSE)
  expect_equal(
    actual$features_tags_loss_graph$removed_features_tags_count,
    expected$features_tags_loss_graph$removed_features_tags_count,
    check.attributes = FALSE
  )
  expect_equal(
    actual$features_tags_loss_graph$main_indices,
    expected$features_tags_loss_graph$main_indices,
    check.attributes = FALSE
  )
  expect_equal(
    actual$features_tags_loss_graph$loss_values,
    expected$features_tags_loss_graph$loss_values,
    tolerance = 1e-12, check.attributes = FALSE
  )
}

test_that("select_features: grouping = 'ByTags' + RecursiveByShapValues matches the Python oracle", {
  expected <- fixture$expected$recursive_by_shap_values

  result <- catboost.select_features(
    learn_pool,
    test_pool = test_pool,
    params = base_params(),
    grouping = "ByTags",
    features_tags_for_select = fixture$features_tags_for_select,
    num_features_tags_to_select = fixture$num_features_tags_to_select,
    algorithm = "RecursiveByShapValues",
    steps = fixture$steps,
    train_final_model = TRUE
  )

  expect_summary_matches(result, expected)
  expect_equal(result$model$tree_count, expected$final_tree_count)
  expect_false(is.null(result$model))
  expect_equal(
    as.numeric(catboost.predict(result$model, test_pool, prediction_type = "RawFormulaVal")),
    expected$final_predict,
    tolerance = 1e-12, check.attributes = FALSE
  )
})

test_that("select_features: grouping = 'ByTags' + RecursiveByLossFunctionChange, train_final_model = FALSE matches the Python oracle", {
  expected <- fixture$expected$recursive_by_loss_function_change

  result <- catboost.select_features(
    learn_pool,
    test_pool = test_pool,
    params = base_params(),
    grouping = "ByTags",
    features_tags_for_select = fixture$features_tags_for_select,
    num_features_tags_to_select = fixture$num_features_tags_to_select,
    algorithm = "RecursiveByLossFunctionChange",
    steps = fixture$steps,
    train_final_model = FALSE
  )

  expect_summary_matches(result, expected)
  expect_null(result$model)
})

test_that("select_features: grouping = 'ByTags' rejects Individual-only arguments", {
  expect_error(
    catboost.select_features(
      learn_pool, grouping = "ByTags", num_features_tags_to_select = 1,
      params = base_params()
    ),
    "You should specify features_tags_for_select"
  )
  expect_error(
    catboost.select_features(
      learn_pool, grouping = "ByTags", features_tags_for_select = c("signal"),
      params = base_params()
    ),
    "You should specify num_features_tags_to_select"
  )
  expect_error(
    catboost.select_features(
      learn_pool, grouping = "ByTags", features_for_select = c(0, 1),
      features_tags_for_select = c("signal"), num_features_tags_to_select = 1,
      params = base_params()
    ),
    "You should not specify features_for_select"
  )
  expect_error(
    catboost.select_features(
      learn_pool, features_for_select = c(0, 1), num_features_to_select = 1,
      features_tags_for_select = c("signal"),
      params = base_params()
    ),
    "You should not specify features_tags_for_select"
  )
  expect_error(
    catboost.select_features(learn_pool, grouping = "Bogus", params = base_params()),
    "Unsupported grouping"
  )
})

test_that("catboost.load_pool: feature_tags validates structure and index ranges", {
  data <- fixture$inputs$learn_features

  expect_error(
    catboost.load_pool(data, feature_tags = list(unnamed = list(features = c(0, 1)), list(features = c(2)))),
    "feature_tags must be a named list"
  )
  expect_error(
    catboost.load_pool(data, feature_tags = list(a = list(features = c(0, 1)), a = list(features = c(2)))),
    "feature_tags must be a named list"
  )
  expect_error(
    catboost.load_pool(data, feature_tags = list(a = list(cost = 1))),
    "must be a list with a 'features' element"
  )
  expect_error(
    catboost.load_pool(data, feature_tags = list(a = list(features = c(0, 99)))),
    "0-based indices in"
  )
  expect_error(
    catboost.load_pool(data, feature_tags = list(a = list(features = c("no_such_feature")))),
    "no feature_names were supplied"
  )

  named_pool <- catboost.load_pool(
    data, feature_names = as.list(paste0("f", seq_len(ncol(data)) - 1L)),
    feature_tags = list(a = list(features = c("f0", "f3")))
  )
  expect_true(inherits(named_pool, "catboost.Pool"))
})

test_that("catboost.load_pool: feature_tags is rejected for file and data.frame pools", {
  csv_path <- tempfile(fileext = ".csv")
  write.table(
    cbind(fixture$inputs$learn_label, fixture$inputs$learn_features),
    file = csv_path, sep = "\t", row.names = FALSE, col.names = FALSE
  )
  cd_path <- tempfile(fileext = ".cd")
  writeLines("0\tLabel", cd_path)
  expect_error(
    catboost.load_pool(csv_path, column_description = cd_path, feature_tags = list(a = list(features = c(0, 1)))),
    "should be NULL when the pool is read from file"
  )

  df <- as.data.frame(fixture$inputs$learn_features)
  expect_error(
    catboost.load_pool(df, label = fixture$inputs$learn_label, feature_tags = list(a = list(features = c(0, 1)))),
    "should be NULL when the pool is constructed from data.frame"
  )
})
