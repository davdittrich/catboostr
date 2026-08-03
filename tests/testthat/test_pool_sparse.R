context("test_pool_sparse.R")

# P3.7 (catboost-8z4.44) part c: sparse (Matrix::dgCMatrix / CSR) input.
#
# catboost.load_pool/catboost.from_matrix accept any Matrix sparseMatrix and
# densify it before handing it to the native Pool builder: CatBoost's sparse
# column format is a memory optimisation whose unstored cells are ordinary
# zeros, so the resulting Pool must equal the one the Python oracle builds
# from the same data as a scipy.sparse.csr_matrix.
#
# The fixture also records the oracle's dense-path predictions and their
# maximum delta against the sparse path (0 on this dataset), so the claim
# above is auditable rather than asserted.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_sparse_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "pool_sparse.json"),
  simplifyVector = TRUE
)

sparse_input <- function() {
  Matrix::Matrix(fixture$inputs$dense, sparse = TRUE)
}

test_that("pool sparse: dgCMatrix input builds the same Pool as the Python oracle's CSR path", {
  skip_if_not_installed("Matrix")
  data <- sparse_input()
  expect_s4_class(data, "dgCMatrix")
  expect_equal(length(data@x), fixture$inputs$nnz)

  pool <- catboost.load_pool(data, label = as.double(fixture$inputs$label))
  expect_equal(catboost.pool.num_row(pool), fixture$expected$num_row)
  expect_equal(catboost.pool.num_col(pool), fixture$expected$num_col)
  expect_equal(catboost.pool.get_feature_names(pool), fixture$expected$feature_names)
  expect_equal(as.double(catboost.pool.get_label(pool)), fixture$expected$label,
               tolerance = 1e-6)
  expect_equal(catboost.pool.get_features(pool), fixture$expected$features,
               tolerance = 1e-6, check.attributes = FALSE)
})

test_that("pool sparse: from_matrix accepts a dgCMatrix directly", {
  skip_if_not_installed("Matrix")
  # catboost.from_matrix is internal (not in NAMESPACE); load_pool delegates to it.
  catboost.from_matrix <- catboostr:::catboost.from_matrix
  pool <- catboost.from_matrix(sparse_input(), label = as.double(fixture$inputs$label))
  expect_equal(catboost.pool.num_row(pool), fixture$expected$num_row)
  expect_equal(catboost.pool.num_col(pool), fixture$expected$num_col)
  expect_equal(catboost.pool.get_features(pool), fixture$expected$features,
               tolerance = 1e-6, check.attributes = FALSE)
})

test_that("pool sparse: fit/predict on a dgCMatrix Pool matches the oracle's CSR path", {
  skip_if_not_installed("Matrix")
  pool <- catboost.load_pool(sparse_input(), label = as.double(fixture$inputs$label))
  params <- fixture$params
  params$verbose <- NULL
  params$logging_level <- "Silent"
  model <- catboost.train(pool, params = params)
  prediction <- catboost.predict(model, pool, prediction_type = "RawFormulaVal")
  expect_equal(as.double(prediction), fixture$expected$predict, tolerance = 1e-6)
  # The oracle's own sparse and dense training paths agree on this dataset,
  # which is what makes densifying in R observationally equivalent here.
  expect_equal(fixture$expected$max_sparse_dense_delta, 0)
})

test_that("pool sparse: a row-major sparse matrix (dgRMatrix) is accepted too", {
  skip_if_not_installed("Matrix")
  data <- methods::as(sparse_input(), "RsparseMatrix")
  expect_s4_class(data, "dgRMatrix")
  pool <- catboost.load_pool(data, label = as.double(fixture$inputs$label))
  expect_equal(catboost.pool.get_features(pool), fixture$expected$features,
               tolerance = 1e-6, check.attributes = FALSE)
})
