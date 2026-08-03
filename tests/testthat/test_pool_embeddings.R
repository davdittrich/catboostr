context("test_pool_embeddings.R")

# P3.7 (catboost-8z4.44) part b: embedding-feature Pool construction.
#
# Python holds embedding features inside the data frame (a column whose cells
# are arrays) and names their positions with Pool(embedding_features = [...]).
# An R matrix cell cannot hold a vector, so R takes them beside the matrix as
# a named list of (object x dimension) numeric matrices, appended after the
# matrix columns:
#   catboost.load_pool(m, label, embedding_features = list(emb = emb_matrix))
#   catboost.from_matrix(m, label, embedding_features_data = list(emb = ...))
# Both must build the same Pool as the oracle's
# Pool(data = df, embedding_features = [2]).
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_embeddings_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "pool_embeddings.json"),
  simplifyVector = TRUE
)

embedding_matrix <- function() fixture$inputs$embedding

feature_matrix <- function() {
  cbind(f0 = fixture$inputs$f0, f1 = fixture$inputs$f1)
}

test_that("pool embeddings: load_pool builds the same Pool as the Python oracle", {
  pool <- catboost.load_pool(
    feature_matrix(),
    label = as.double(fixture$inputs$label),
    embedding_features = list(emb = embedding_matrix())
  )

  expect_equal(catboost.pool.num_row(pool), fixture$expected$num_row)
  expect_equal(catboost.pool.num_col(pool), fixture$expected$num_col)
  expect_equal(catboost.pool.get_feature_names(pool), fixture$expected$feature_names)
  expect_equal(catboost.pool.get_embedding_feature_indices(pool),
               fixture$expected$embedding_feature_indices)
  # The oracle reports no cat/text features for this Pool (empty JSON arrays).
  expect_length(fixture$expected$cat_feature_indices, 0)
  expect_length(catboost.pool.get_cat_feature_indices(pool), 0)
  expect_length(fixture$expected$text_feature_indices, 0)
  expect_length(catboost.pool.get_text_feature_indices(pool), 0)
  expect_equal(as.double(catboost.pool.get_label(pool)), fixture$expected$label,
               tolerance = 1e-6)
})

test_that("pool embeddings: from_matrix with explicit indices matches load_pool", {
  # catboost.from_matrix is internal (not in NAMESPACE); load_pool delegates to it.
  catboost.from_matrix <- catboostr:::catboost.from_matrix
  pool <- catboost.from_matrix(
    feature_matrix(),
    label = as.double(fixture$inputs$label),
    feature_names = as.list(fixture$expected$feature_names),
    embedding_features_data = list(embedding_matrix()),
    embedding_features_indices = fixture$expected$embedding_feature_indices
  )
  expect_equal(catboost.pool.num_col(pool), fixture$expected$num_col)
  expect_equal(catboost.pool.get_feature_names(pool), fixture$expected$feature_names)
  expect_equal(catboost.pool.get_embedding_feature_indices(pool),
               fixture$expected$embedding_feature_indices)
})

# Embedding processing is pinned to KNN by the fixture. The default
# ("LDA", "KNN") is not a usable differential probe: this fork's binary and
# the pinned Python wheel disagree on the LDA calcer even when both load the
# identical dsv file through the identical file loader, so the disagreement
# predates any Pool-construction path (see the P3.7 report's control run).
test_that("pool embeddings: fit/predict on an embedding Pool matches Python oracle", {
  pool <- catboost.load_pool(
    feature_matrix(),
    label = as.double(fixture$inputs$label),
    embedding_features = list(emb = embedding_matrix())
  )
  params <- fixture$params
  params$verbose <- NULL
  params$logging_level <- "Silent"
  model <- catboost.train(pool, params = params)
  prediction <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(as.double(prediction), fixture$expected$predict, tolerance = 1e-6)
})

test_that("pool embeddings: malformed embedding_features input is rejected", {
  expect_error(
    catboost.load_pool(feature_matrix(), label = as.double(fixture$inputs$label),
                       embedding_features = list(emb = embedding_matrix()[1:2, ])),
    "embedding_features_data\\[\\[1\\]\\] has 2 rows"
  )
  expect_error(
    catboost.load_pool(data.frame(a = c(1, 2)), label = c(0, 1),
                       embedding_features = list(emb = matrix(0, 2, 2))),
    "only supported when 'data' is a matrix"
  )
})
