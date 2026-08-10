context("test_grouped_data_parity.R")

# catboost-inm (followup to catboost-8z4.115/P10.A): closes the 6-row
# grouped-data flag family (flag:--learn-pairs, flag:--test-pairs,
# flag:--input-pairs, flag:--learn-graph, flag:--test-graph,
# flag:--input-graph) that P10.A fix-round-1 left red for lack of any
# pairs/graph fixture. All six load the same dsv-flat WinnerId<TAB>LoserId
# format (catboost/libs/data/pairs_data_loaders.cpp) into the same
# TRawPairsData structure (catboost/libs/data/graph.cpp) -- --learn-graph/
# --test-graph/--input-graph feed the identical loader as --learn-pairs/
# --test-pairs/--input-pairs, just consumed for aggregation features
# instead of pairwise-ranking loss -- so one CLI-oracle fixture (relative
# 1e-6 tolerance, matching test_cli_hyperparam_coverage.R's CLI-oracle
# convention) covers the family.
#
# Two models, not one: a model trained with a graph unconditionally
# requires a graph at predict time too (MeanAggregated/MinAggregated/
# MaxAggregated features have nowhere else to come from), and
# catboost/libs/data/proceed_pool_in_blocks.h:95 refuses pairs+graph
# together in calc's block-streaming loader -- so pairs (PairLogit) and
# graph (RMSE) each get their own fit+calc run.
#
# Regenerate fixtures (after tools/oracle/cli/acquire.sh) with:
#   tools/oracle/cli/gen_grouped_data_fixture.sh

CLI_TOLERANCE <- 1e-6

DATA_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "grouped_data.csv")
CD_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "grouped_data.cd")
PAIRS_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "grouped_pairs.tsv")
GRAPH_PATH <- testthat::test_path("..", "fixtures", "oracle-cli", "grouped_graph.tsv")

read_grouped_data <- function() {
  df <- read.csv(DATA_PATH)
  list(x = as.matrix(df[, c("x1", "x2")]), y = df$target, group_id = df$group_id)
}

read_oracle_predictions <- function(name) {
  path <- testthat::test_path("..", "fixtures", "oracle-cli", name)
  raw <- jsonlite::fromJSON(path, simplifyDataFrame = TRUE)
  raw$RawFormulaVal[order(raw$SampleId)]
}

test_that("flag:--learn-pairs/--test-pairs/--input-pairs: PairLogit fit+calc match the CLI oracle", {
  d <- read_grouped_data()
  pairs <- as.matrix(read.delim(PAIRS_PATH, header = FALSE))
  storage.mode(pairs) <- "integer"

  pool <- catboost.load_pool(d$x, label = d$y, group_id = d$group_id, pairs = pairs)

  model <- catboost.train(pool, params = list(
    loss_function = "PairLogit", iterations = 20, depth = 4, learning_rate = 0.1,
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))

  # --input-pairs: reload a predict-time pool carrying the same pairs (the
  # model doesn't need them to score under PairLogit, but this is exactly
  # what --input-pairs loads at calc time).
  predict_pool <- catboost.load_pool(d$x, pairs = pairs)
  preds <- catboost.predict(model, predict_pool, prediction_type = "RawFormulaVal")

  oracle <- read_oracle_predictions("grouped_data_predictions_pairs.json")
  expect_equal(as.numeric(preds), as.numeric(oracle), tolerance = CLI_TOLERANCE)
})

test_that("flag:--learn-graph/--test-graph/--input-graph: RMSE fit+calc match the CLI oracle", {
  # catboost.from_matrix's in-memory builder has graph support genuinely
  # unimplemented natively (catboost/libs/data/data_provider_builders.cpp:
  # TRawObjectsDataProviderBuilder::SetGraph() is CB_ENSURE_INTERNAL(false,
  # "Unimplemented") -- verified empirically, not just by reading; filed as
  # catboost-hpk), so this exercises the file-based catboost.from_file()
  # path (graph_path argument, CatBoostCreateFromFile_R), which is the one
  # path the CLI's own --learn-graph/--test-graph/--input-graph loaders
  # match.
  pool <- catboost.load_pool(DATA_PATH, column_description = CD_PATH, delimiter = ",",
                              has_header = TRUE, thread_count = 1, graph = GRAPH_PATH)

  model <- catboost.train(pool, params = list(
    loss_function = "RMSE", iterations = 20, depth = 4, learning_rate = 0.1,
    random_seed = 42, thread_count = 1, logging_level = "Silent"
  ))

  # --input-graph: the graph-aggregated features are structural, so the
  # predict-time pool must carry the same graph or scoring fails outright.
  predict_pool <- catboost.load_pool(DATA_PATH, column_description = CD_PATH, delimiter = ",",
                                      has_header = TRUE, thread_count = 1, graph = GRAPH_PATH)
  preds <- catboost.predict(model, predict_pool, prediction_type = "RawFormulaVal")

  oracle <- read_oracle_predictions("grouped_data_predictions_graph.json")
  expect_equal(as.numeric(preds), as.numeric(oracle), tolerance = CLI_TOLERANCE)
})
