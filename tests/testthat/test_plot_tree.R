context("test_plot_tree.R")

# P4.3 (catboost-8z4.52) structural differential test: catboost.plot_tree
# (R/catboost.R). Python's plot_tree(tree_idx, pool=None) returns a
# graphviz.Digraph built from the model's own splits/leaf_values (never
# rendered unless .render()/.view() is explicitly called) -- spec Sec 4.3's
# "Structural" row targets exactly that node/edge structure, not a rendered
# image. The oracle fixture stores, for each graphviz.Digraph.body entry
# (one per node()/edge() call on the real Python object -- not a
# reimplementation of the tree-print algorithm), the parsed id/label/
# color/shape (nodes) or from/to/label (edges) fields as canonical JSON.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_plot_tree_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "plot_tree.json"),
  simplifyVector = TRUE
)

sort_rows <- function(df, key_col) {
  df[order(as.integer(df[[key_col]])), , drop = FALSE]
}

# catboost.load_pool()/catboost.from_matrix() pre-hash categorical columns into
# floats via CatBoostHashStrings_R before the vendor pool is ever built, so (same
# root cause documented in test_calc_feature_statistics.R for
# CatBoostGetCatFeatureValues_R) the R-built Pool has no hash-to-string
# dictionary to recover the original category strings from. catboost.plot_tree()
# falls back to a "<hash:...>" label for the one-hot split value in that case
# (see R/catboost.R's catboost.plot_tree @param pool docs); normalize that one
# token away before comparing against the Python oracle, which always has real
# strings available.
normalize_cat_values <- function(df) {
  df$label <- sub("(?<=, value=).*$", "<CAT>", df$label, perl = TRUE)
  df
}

expect_nodes_edges_match <- function(result, expected, normalize_cat = FALSE) {
  expect_s3_class(result, "catboost.plot_tree")
  expect_type(result$dot, "character")
  expect_true(grepl("^digraph \\{", result$dot))

  got_nodes <- sort_rows(result$nodes, "id")
  exp_nodes <- sort_rows(expected$nodes, "id")
  rownames(got_nodes) <- NULL
  rownames(exp_nodes) <- NULL
  if (normalize_cat) {
    got_nodes <- normalize_cat_values(got_nodes)
    exp_nodes <- normalize_cat_values(exp_nodes)
  }
  expect_equal(got_nodes, exp_nodes)

  got_edges <- sort_rows(result$edges, "to")
  exp_edges <- sort_rows(expected$edges, "to")
  rownames(got_edges) <- NULL
  rownames(exp_edges) <- NULL
  expect_equal(got_edges, exp_edges)
}

build_float_only_pool_and_model <- function() {
  inputs <- fixture$float_only$inputs
  data <- data.frame(num1 = inputs$num1, num2 = inputs$num2)
  pool <- catboost.load_pool(data, label = inputs$label)
  catboost.pool.set_feature_names(pool, inputs$feature_names)

  model <- catboost.train(pool, params = list(
    iterations = 3, depth = 3, loss_function = "Logloss",
    random_seed = 1, thread_count = 1, logging_level = "Silent"
  ))
  list(pool = pool, model = model, tree_idx = inputs$tree_idx)
}

build_cat_pool_and_model <- function() {
  inputs <- fixture$with_cat$inputs
  data <- data.frame(num1 = inputs$num1, cat1 = factor(inputs$cat1), stringsAsFactors = FALSE)
  pool <- catboost.load_pool(data, label = inputs$label)
  catboost.pool.set_feature_names(pool, inputs$feature_names)

  model <- catboost.train(pool, params = list(
    iterations = 3, depth = 2, loss_function = "Logloss",
    random_seed = 2, thread_count = 1, logging_level = "Silent",
    one_hot_max_size = 10
  ))
  list(pool = pool, model = model, tree_idx = inputs$tree_idx)
}

test_that("plot_tree: float-only tree, with pool, matches Python oracle", {
  built <- build_float_only_pool_and_model()
  result <- catboost.plot_tree(built$model, built$tree_idx, built$pool)
  expect_nodes_edges_match(result, fixture$float_only$expected$with_pool)
})

test_that("plot_tree: float-only tree, without pool, matches Python oracle", {
  built <- build_float_only_pool_and_model()
  result <- catboost.plot_tree(built$model, built$tree_idx)
  expect_nodes_edges_match(result, fixture$float_only$expected$without_pool)
})

test_that("plot_tree: categorical (one-hot) tree, with pool, matches Python oracle structure", {
  built <- build_cat_pool_and_model()
  result <- catboost.plot_tree(built$model, built$tree_idx, built$pool)
  expect_nodes_edges_match(result, fixture$with_cat$expected$with_pool, normalize_cat = TRUE)
  # The categorical split value itself falls back to a "<hash:...>" label (see
  # normalize_cat_values() above): assert that fallback shape explicitly, so a
  # regression that silently drops the value entirely is still caught.
  cat_nodes <- result$nodes[grepl(", value=", result$nodes$label), ]
  expect_true(nrow(cat_nodes) > 0)
  expect_true(all(grepl(", value=<hash:-?[0-9]+>$", cat_nodes$label)))
})

test_that("plot_tree: rejects a non-model", {
  expect_error(
    catboost.plot_tree("not-a-model", 0),
    "Expected catboost.Model"
  )
})

test_that("plot_tree: rejects a non-pool", {
  built <- build_float_only_pool_and_model()
  expect_error(
    catboost.plot_tree(built$model, built$tree_idx, "not-a-pool"),
    "Expected catboost.Pool"
  )
})

test_that("plot_tree: rejects an out-of-range tree_idx", {
  built <- build_float_only_pool_and_model()
  expect_error(
    catboost.plot_tree(built$model, 999L),
    "tree_idx out of range"
  )
})

test_that("plot_tree: categorical split without pool raises the parity error", {
  built <- build_cat_pool_and_model()
  expect_error(
    catboost.plot_tree(built$model, built$tree_idx),
    "training dataset is required if categorical features are present"
  )
})
