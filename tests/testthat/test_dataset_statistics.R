context("test_dataset_statistics.R")

# P3.3 differential test: run catboost.dataset_statistics() (R equivalent of
# CatBoost CLI's `dataset-statistics` mode -- calls the same core library
# entry point the CLI mode itself calls,
# NCB::CalculateDatasetStatisticsSingleHost, in-process) on the pinned smoke
# fixture dataset (tests/fixtures/oracle-cli/smoke_data.csv +
# smoke.cd, same fixture used by the training/predict CLI oracle) and assert
# its output matches the pinned CatBoost CLI v1.2.10 oracle
# (tools/oracle/cli/bin/catboost-v1.2.10 dataset-statistics) byte-for-byte
# after JSON parsing.
#
# Regenerate the committed oracle fixture with:
#   tools/oracle/cli/acquire.sh
#   ./tools/oracle/cli/bin/catboost-v1.2.10 dataset-statistics \
#     --input-path tests/fixtures/oracle-cli/smoke_data.csv \
#     --cd tests/fixtures/oracle-cli/smoke.cd \
#     --delimiter , --has-header --border-counts 32 -T 1 \
#     -o tests/fixtures/oracle-cli/dataset_statistics_stats.json \
#     --histograms-path tests/fixtures/oracle-cli/dataset_statistics_histograms.json

POOL_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv")
CD_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd")
BORDER_COUNT <- 32

oracle_stats <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle-cli", "dataset_statistics_stats.json"),
  simplifyVector = TRUE
)
oracle_histograms <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle-cli", "dataset_statistics_histograms.json"),
  simplifyVector = TRUE
)

test_that("dataset_statistics: statistics match the CatBoost CLI oracle", {
  res <- catboost.dataset_statistics(
    POOL_PATH,
    cd_path = CD_PATH,
    delimiter = ",",
    has_header = TRUE,
    border_count = BORDER_COUNT,
    thread_count = 1
  )
  expect_equal(res$statistics, oracle_stats)
})

test_that("dataset_statistics: histograms match the CatBoost CLI oracle", {
  res <- catboost.dataset_statistics(
    POOL_PATH,
    cd_path = CD_PATH,
    delimiter = ",",
    has_header = TRUE,
    border_count = BORDER_COUNT,
    thread_count = 1
  )
  expect_equal(res$histograms, oracle_histograms)
})

test_that("dataset_statistics: only_light_statistics skips histograms, matches CLI oracle behavior", {
  res <- catboost.dataset_statistics(
    POOL_PATH,
    cd_path = CD_PATH,
    delimiter = ",",
    has_header = TRUE,
    border_count = BORDER_COUNT,
    thread_count = 1,
    only_light_statistics = TRUE
  )
  expect_null(res$histograms)
  expect_equal(res$statistics, oracle_stats)
})
