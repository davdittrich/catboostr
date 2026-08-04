context("test_virtual_ensembles.R")

# P5.4 (catboost-8z4.61) -- catboost.virtual_ensembles_predict differential
# test against the Python oracle.
#
# No matrix row in tests/fixtures/parity/matrix.dispositioned.json names this
# capability (grep for "virtual" only turns up flag:--virtual-ensembles-count,
# the unrelated CLI calc-mode flag). capability_diff.json *does* carry
# CatBoost.virtual_ensembles_predict matched to catboost.virtual_ensembles_predict,
# so the Phase 2 inventory found and matched the capability but it never made
# it into the dispositioned matrix -- a genuine Phase 2 inventory gap, filed
# as a follow-up ticket rather than hand-added here.
#
# R's catboost.virtual_ensembles_predict (R/catboost.R) calls the native
# CatBoostPredictVirtualEnsembles_R entry point (src/catboostr.cpp), which
# wraps the same ApplyUncertaintyPredictions used by Python's
# _base_virtual_ensembles_predict. Both sides then reshape the flat
# (objects x virtual_ensembles*dim) buffer independently -- R via
# aperm(array(..., dim = c(D, V, N)), perm = c(2, 1, 3)), Python via
# predictions.reshape(N, V, D) -- so this test is exercising that the two
# reshapes agree, not merely that the native call succeeds.
#
# Regenerate fixture with:
# uv run --frozen --project tools/oracle python3 tools/oracle/gen_virtual_ensembles_fixture.py

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "virtual_ensembles.json"),
  simplifyVector = TRUE
)

params <- function() {
  p <- fixture$base_params
  p$verbose <- NULL
  p$logging_level <- "Silent"
  return(p)
}

learn_pool <- catboost.load_pool(fixture$inputs$learn_features, label = fixture$inputs$learn_label)
test_pool <- catboost.load_pool(fixture$inputs$test_features)
model <- catboost.train(learn_pool, params = params())

test_that("virtual_ensembles_predict: VirtEnsembles shape and values match the Python oracle", {
  expected_shape <- fixture$expected$virt_ensembles_shape # (N, V, D) numpy order
  expected <- fixture$expected$virt_ensembles

  actual <- catboost.virtual_ensembles_predict(
    model,
    test_pool,
    prediction_type = "VirtEnsembles",
    virtual_ensembles_count = fixture$virtual_ensembles_count,
    thread_count = 1
  )

  # R array comes out of aperm(array(flat, dim=c(D,V,N)), perm=c(2,1,3)) as
  # (V, D, N): dims 1 and 2 of the pre-aperm (D,V,N) array are swapped, then
  # N stays last. Python/numpy's array is (N, V, D). So actual[v,d,n] ==
  # expected[n,v,d] -- aperm(actual, c(3,1,2)) reorders R's (V,D,N) into
  # (N,V,D) to compare directly against the oracle's nested list.
  expected_v <- fixture$virtual_ensembles_count
  expected_d <- expected_shape[3]
  expect_equal(dim(actual), c(expected_v, expected_d, expected_shape[1]))
  actual_nvd <- aperm(actual, perm = c(3, 1, 2))
  expect_equal(dim(actual_nvd), expected_shape)
  expect_equal(actual_nvd, expected, tolerance = 1e-12, check.attributes = FALSE)
})

test_that("virtual_ensembles_predict: TotalUncertainty shape and values match the Python oracle", {
  expected_shape <- fixture$expected$total_uncertainty_shape
  expected <- fixture$expected$total_uncertainty

  actual <- catboost.virtual_ensembles_predict(
    model,
    test_pool,
    prediction_type = "TotalUncertainty",
    virtual_ensembles_count = fixture$virtual_ensembles_count,
    thread_count = 1
  )

  expect_equal(dim(actual), expected_shape)
  expect_equal(actual, expected, tolerance = 1e-12, check.attributes = FALSE)
})
