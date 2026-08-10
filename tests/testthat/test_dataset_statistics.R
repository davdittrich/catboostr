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

# catboost-hpk.4: differential tests for the 4 CLI-only sub-flags
# (--not-convert-string-targets, --custom-feature-limits, --spot-size,
# --spot-count) added to the 9-argument R wrapper. Each maps directly onto an
# existing TCalculateStatisticsParams field (mode_dataset_statistics_helpers.h)
# that CalculateDatasetStatisticsSingleHost already consumes.

# --not-convert-string-targets needs a string-typed Label column to have any
# observable effect (the smoke fixture's target is already numeric), so this
# uses its own tiny fixture.
STR_TARGET_POOL_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "dataset_statistics_string_target.csv")
STR_TARGET_CD_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "dataset_statistics_string_target.cd")

oracle_str_target_stats <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle-cli", "dataset_statistics_string_target_stats.json"),
  simplifyVector = TRUE
)
oracle_str_target_converted_stats <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle-cli", "dataset_statistics_string_target_converted_stats.json"),
  simplifyVector = TRUE
)

# Regenerate with:
#   ./tools/oracle/cli/bin/catboost-v1.2.10 dataset-statistics \
#     --input-path tests/fixtures/oracle-cli/dataset_statistics_string_target.csv \
#     --cd tests/fixtures/oracle-cli/dataset_statistics_string_target.cd \
#     --delimiter , --has-header --border-counts 32 -T 1 \
#     -o tests/fixtures/oracle-cli/dataset_statistics_string_target_stats.json \
#     --histograms-path /tmp/histograms.json
#   ./tools/oracle/cli/bin/catboost-v1.2.10 dataset-statistics \
#     --input-path tests/fixtures/oracle-cli/dataset_statistics_string_target.csv \
#     --cd tests/fixtures/oracle-cli/dataset_statistics_string_target.cd \
#     --delimiter , --has-header --border-counts 32 -T 1 \
#     --not-convert-string-targets false \
#     -o tests/fixtures/oracle-cli/dataset_statistics_string_target_converted_stats.json \
#     --histograms-path /tmp/histograms.json
test_that("dataset_statistics: not_convert_string_targets defaults to keeping string targets as-is", {
  res <- catboost.dataset_statistics(
    STR_TARGET_POOL_PATH,
    cd_path = STR_TARGET_CD_PATH,
    delimiter = ",",
    has_header = TRUE,
    border_count = BORDER_COUNT,
    thread_count = 1,
    only_light_statistics = TRUE
  )
  expect_equal(res$statistics, oracle_str_target_stats)
})

test_that("dataset_statistics: not_convert_string_targets = FALSE converts a numeric-string target to float", {
  res <- catboost.dataset_statistics(
    STR_TARGET_POOL_PATH,
    cd_path = STR_TARGET_CD_PATH,
    delimiter = ",",
    has_header = TRUE,
    border_count = BORDER_COUNT,
    thread_count = 1,
    only_light_statistics = TRUE,
    not_convert_string_targets = FALSE
  )
  expect_equal(res$statistics, oracle_str_target_converted_stats)
})

oracle_cfl_stats <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle-cli", "dataset_statistics_custom_feature_limits_stats.json"),
  simplifyVector = TRUE
)

# Regenerate with:
#   ./tools/oracle/cli/bin/catboost-v1.2.10 dataset-statistics \
#     --input-path tests/fixtures/oracle-cli/smoke_data.csv \
#     --cd tests/fixtures/oracle-cli/smoke.cd \
#     --delimiter , --has-header --border-counts 32 -T 1 \
#     --custom-feature-limits "0:0:1" \
#     -o tests/fixtures/oracle-cli/dataset_statistics_custom_feature_limits_stats.json \
#     --histograms-path /tmp/histograms.json
test_that("dataset_statistics: custom_feature_limits clips a float feature and reports over/underflow", {
  res <- catboost.dataset_statistics(
    POOL_PATH,
    cd_path = CD_PATH,
    delimiter = ",",
    has_header = TRUE,
    border_count = BORDER_COUNT,
    thread_count = 1,
    only_light_statistics = TRUE,
    custom_feature_limits = "0:0:1"
  )
  expect_equal(res$statistics, oracle_cfl_stats)
})

oracle_spot_stats <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle-cli", "dataset_statistics_spot_stats.json"),
  simplifyVector = TRUE
)

# Regenerate with:
#   ./tools/oracle/cli/bin/catboost-v1.2.10 dataset-statistics \
#     --input-path tests/fixtures/oracle-cli/smoke_data.csv \
#     --cd tests/fixtures/oracle-cli/smoke.cd \
#     --delimiter , --has-header --border-counts 32 -T 1 \
#     --spot-size 10 --spot-count 2 \
#     -o tests/fixtures/oracle-cli/dataset_statistics_spot_stats.json \
#     --histograms-path /tmp/histograms.json
test_that("dataset_statistics: spot_size/spot_count subsample the dataset instead of reading it whole", {
  res <- catboost.dataset_statistics(
    POOL_PATH,
    cd_path = CD_PATH,
    delimiter = ",",
    has_header = TRUE,
    border_count = BORDER_COUNT,
    thread_count = 1,
    only_light_statistics = TRUE,
    spot_size = 10,
    spot_count = 2
  )
  expect_equal(res$statistics, oracle_spot_stats)
})

test_that("dataset_statistics: spot_size and spot_count must be given together", {
  expect_error(
    catboost.dataset_statistics(
      POOL_PATH,
      cd_path = CD_PATH,
      delimiter = ",",
      has_header = TRUE,
      thread_count = 1,
      spot_size = 10
    ),
    "spot_size and spot_count"
  )
})
