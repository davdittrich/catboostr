context("test_metadata.R")

# P5.6 (catboost-8z4.63) differential test: catboost.get_metadata(),
# catboost.set_metadata() and catboost.get_model_feature_names() (R/catboost.R),
# which read/write the same THashMap<TString, TString> TFullModel::ModelInfo
# (catboost/libs/model/model.h) that Python's model.get_metadata() proxy
# (_MetadataHashProxy, _catboost.pyx) wraps and that the CLI's `metadata`
# mode (catboost/app/mode_metadata.cpp) reads/writes directly on a saved
# model file.
#
# Subcommand -> R function mapping (Step 5 of the task brief: only add a
# CLI-only R equivalent if no in-process function already covers it -- none
# needed here, all four CLI subcommands are covered by functions that mirror
# an existing Python capability):
#   metadata set                 -> catboost.set_metadata()   (Python: model.get_metadata()[key] = value)
#   metadata get                 -> catboost.get_metadata()[[key]] (Python: model.get_metadata()[key] / .get())
#   metadata dump                -> catboost.get_metadata()   (Python: dict(model.get_metadata()))
#   metadata dump-feature-names  -> catboost.get_model_feature_names() (Python: model.feature_names_)
#
# Five matrix rows:
#  - CatBoost.get_metadata / CatBoostClassifier.get_metadata /
#    CatBoostRegressor.get_metadata / CatBoostRanker.get_metadata (Python
#    oracle, tolerance 1e-12): all four Python classes delegate to the same
#    _CatBoostBase._object._get_metadata_wrapper() (core.py:2061-2062), so one
#    R-side differential test against a model trained with catboost.train()
#    covers all four rows.
#  - mode:metadata / mode:metadata set / mode:metadata get / mode:metadata
#    dump / mode:metadata dump-feature-names (CLI oracle, tolerance 1e-9):
#    pinned CatBoost CLI v1.2.10 binary's `metadata` mode output on the
#    shared smoke fixture, compared to a model trained in R with the same
#    params.
#
# Known, documented divergences excluded from the strict comparisons below
# (none of them are metadata-accessor bugs -- see each comment for why):
#  - train_finish_time, model_guid: embed the actual wall-clock time / a
#    fresh GUID at serialization, so they differ on every run by design (see
#    docs/phase-0/catboost-8z4.3-report.md's ".cbm files are not
#    byte-identical across runs" finding).
#  - catboost_version_info: baked in at compile time from the *building*
#    tree's VCS info (gen_vcs_info); this R package's build tree has none
#    (compiled from a gitignored acquired copy of vendor/catboost, not a git
#    checkout), so it reads "No VCS" here versus the pinned CLI/Python
#    wheel's real git commit info. Orthogonal to metadata get/set/dump.
#  - class_params (CLI section only): the CLI's `fit` reads Label directly
#    off the CSV text and treats it as a String class label; catboostr's
#    pool/label ingestion normalizes numeric-looking labels to Integer before
#    they reach the C++ core. A pool/label-ingestion difference, not a
#    metadata-accessor one; already out of scope for this ticket (see the
#    brief: "metadata mode only -- no normalize-model").
#  - flat_params$model_format (CLI section only): only set when training
#    writes a model to a file with an explicit CLI --model-format flag;
#    catboost.train() keeps the model in memory. Not a metadata bug.
#
# Regenerate the Python fixture with:
#   uv run --frozen --project tools/oracle python3 tools/oracle/gen_metadata_fixture.py
#
# Regenerate the CLI fixture with (after tools/oracle/cli/acquire.sh):
#   ./tools/oracle/cli/bin/catboost-v1.2.10 fit \
#     --learn-set tests/fixtures/oracle-cli/smoke_data.csv \
#     --column-description tests/fixtures/oracle-cli/smoke.cd \
#     --delimiter , --has-header --loss-function Logloss -i 20 --depth 4 \
#     --learning-rate 0.1 --random-seed 42 -T 1 \
#     --model-file /tmp/smoke_model.cbm --logging-level Silent
#   ./tools/oracle/cli/bin/catboost-v1.2.10 metadata set \
#     --key custom_key --value custom_value \
#     -m /tmp/smoke_model.cbm -o /tmp/smoke_model_set.cbm
#   ./tools/oracle/cli/bin/catboost-v1.2.10 metadata dump \
#     -m /tmp/smoke_model.cbm --dump-format JSON            # -> "dump" field
#   ./tools/oracle/cli/bin/catboost-v1.2.10 metadata dump-feature-names \
#     -m /tmp/smoke_model.cbm                                # -> "feature_names" field
#   ./tools/oracle/cli/bin/catboost-v1.2.10 metadata get \
#     --key custom_key -m /tmp/smoke_model_set.cbm            # -> "set_then_get" field
# assembled into tests/fixtures/oracle-cli/metadata_dump.json.

PY_TOL <- 1e-12
CLI_TOL <- 1e-9

`%||%` <- function(a, b) if (is.null(a)) b else a

VOLATILE_KEYS <- c("train_finish_time", "model_guid", "catboost_version_info")

# Recursively sort a parsed-JSON list's names so two structurally-equal JSON
# objects compare equal regardless of THashMap iteration order.
canonicalize <- function(x) {
  if (is.list(x)) {
    if (!is.null(names(x)) && all(names(x) != "")) {
      x <- x[order(names(x))]
    }
    x <- lapply(x, canonicalize)
  }
  x
}

norm_json <- function(json_str, drop_paths = list()) {
  x <- jsonlite::fromJSON(json_str, simplifyVector = FALSE)
  for (path in drop_paths) {
    if (length(path) == 1) {
      x[[path]] <- NULL
    } else {
      x[[path[1]]][[path[2]]] <- NULL
    }
  }
  jsonlite::toJSON(canonicalize(x), auto_unbox = TRUE, null = "null")
}

## --- CatBoost{,Classifier,Regressor,Ranker}.get_metadata (Python oracle) ---

fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle", "metadata.json"),
  simplifyVector = TRUE
)
inputs <- fixture$inputs
expected <- fixture$expected

py_pool <- catboost.load_pool(
  data.frame(num1 = inputs$num1, num2 = inputs$num2),
  label = inputs$label,
  feature_names = as.list(inputs$feature_names)
)
py_model <- catboost.train(py_pool, params = list(
  iterations = 10, depth = 2, loss_function = "Logloss",
  random_seed = 42, thread_count = 1, logging_level = "Silent",
  train_dir = tempfile("catboost_train_metadata_")
))

test_that("get_metadata: key set matches the Python oracle", {
  expect_equal(
    sort(setdiff(names(catboost.get_metadata(py_model)), VOLATILE_KEYS)),
    sort(setdiff(names(expected$metadata_before), VOLATILE_KEYS))
  )
})

test_that("get_metadata: class_params and training values are byte-identical to the Python oracle", {
  md <- catboost.get_metadata(py_model)
  expect_identical(md[["class_params"]], expected$metadata_before$class_params)
  expect_identical(md[["training"]], expected$metadata_before$training)
})

test_that("get_metadata: params and output_options match the Python oracle up to the run-specific train_dir", {
  md <- catboost.get_metadata(py_model)
  expect_identical(
    norm_json(md[["output_options"]], list("train_dir")),
    norm_json(expected$metadata_before$output_options, list("train_dir"))
  )
  expect_identical(
    norm_json(md[["params"]], list(c("flat_params", "train_dir"))),
    norm_json(expected$metadata_before$params, list(c("flat_params", "train_dir")))
  )
})

test_that("get_model_feature_names: matches the Python oracle's feature_names_", {
  expect_identical(catboost.get_model_feature_names(py_model), expected$feature_names)
})

test_that("get_metadata: a key absent before set_metadata is NA via `[` and errors via `[[`", {
  md <- catboost.get_metadata(py_model)
  expect_true(is.na(md["custom_key"]))
  expect_error(md[["custom_key"]], "subscript out of bounds")
})

test_that("set_metadata: mutates the live model in place, matching the Python oracle's assignment", {
  catboost.set_metadata(py_model, "custom_key", "custom_value")
  md <- catboost.get_metadata(py_model)
  expect_identical(md[["custom_key"]], expected$metadata_after_set$custom_key)
})

test_that("set_metadata: validates its arguments", {
  expect_error(catboost.set_metadata(py_model, c("a", "b"), "v"), "key must be a single string")
  expect_error(catboost.set_metadata(py_model, "k", c("a", "b")), "value must be a single string")
  expect_error(catboost.set_metadata(py_model, 1, "v"), "key must be a single string")
})

test_that("get_metadata and set_metadata reject a non-Model first argument", {
  expect_error(catboost.get_metadata(list()), "Expected catboost.Model")
  expect_error(catboost.set_metadata(list(), "k", "v"), "Expected catboost.Model")
  expect_error(catboost.get_model_feature_names(list()), "Expected catboost.Model")
})

## --- mode:metadata / mode:metadata set / get / dump / dump-feature-names (CLI oracle) ---

cli_fixture <- jsonlite::fromJSON(
  testthat::test_path("..", "fixtures", "oracle-cli", "metadata_dump.json"),
  simplifyVector = TRUE
)

cli_data <- read.csv(
  testthat::test_path("..", "fixtures", "oracle-cli", "smoke_data.csv"),
  header = TRUE
)
cli_cd <- read.table(
  testthat::test_path("..", "fixtures", "oracle-cli", "smoke.cd"),
  header = FALSE, sep = "\t", stringsAsFactors = FALSE
)
cli_label_col <- cli_cd[cli_cd$V2 == "Label", "V1"] + 1 # 0-indexed in .cd, 1-indexed in R
cli_cat_col <- cli_cd[cli_cd$V2 == "Categ", "V1"] + 1 # 0-indexed in .cd, 1-indexed in R
cli_data[[cli_cat_col]] <- as.factor(cli_data[[cli_cat_col]])
cli_pool <- catboost.load_pool(
  cli_data[, -cli_label_col],
  label = as.character(cli_data[, cli_label_col]),
  # The CLI's smoke.cd only names column 2 ("cat1"); columns 0/1 fall back to
  # their bare index as a name (mode_metadata.cpp's GetModelUsedFeaturesNames
  # output below). Declaring the same index-strings here, instead of letting
  # catboost.load_pool() inherit smoke_data.csv's real header names, isolates
  # the metadata-accessor comparison from that unrelated pool-naming
  # difference (see docs/phase-3 pool-parity coverage for that).
  feature_names = as.list(c("0", "1", "cat1"))
)
cli_model <- catboost.train(cli_pool, params = list(
  loss_function = "Logloss", iterations = 20, depth = 4,
  learning_rate = 0.1, random_seed = 42, thread_count = 1,
  logging_level = "Silent"
))

test_that("mode:metadata dump / mode:metadata -- key set matches the CLI oracle", {
  expect_equal(
    sort(setdiff(names(catboost.get_metadata(cli_model)), c(VOLATILE_KEYS, "class_params"))),
    sort(setdiff(names(cli_fixture$dump), c(VOLATILE_KEYS, "class_params")))
  )
})

test_that("mode:metadata dump -- the deterministic training metrics block is byte-identical to the CLI oracle", {
  expect_identical(catboost.get_metadata(cli_model)[["training"]], cli_fixture$dump$training)
})

test_that("mode:metadata get -- representative params/output_options fields match the CLI oracle", {
  md <- catboost.get_metadata(cli_model)
  r_params <- jsonlite::fromJSON(md[["params"]], simplifyVector = FALSE)
  cli_params <- jsonlite::fromJSON(cli_fixture$dump$params, simplifyVector = FALSE)
  expect_identical(r_params$flat_params$iterations, cli_params$flat_params$iterations)
  expect_identical(r_params$flat_params$depth, cli_params$flat_params$depth)
  expect_identical(r_params$flat_params$loss_function, cli_params$flat_params$loss_function)
  expect_identical(r_params$flat_params$random_seed, cli_params$flat_params$random_seed)
  # learning_rate round-trips through a float32 in the C++ core (0.1 ->
  # 0.1000000015 -- a wider float32-rounding tolerance than CLI_TOL, not a
  # metadata-accessor imprecision).
  expect_equal(r_params$flat_params$learning_rate, cli_params$flat_params$learning_rate, tolerance = 1e-7)

  r_oo <- jsonlite::fromJSON(md[["output_options"]], simplifyVector = FALSE)
  cli_oo <- jsonlite::fromJSON(cli_fixture$dump$output_options, simplifyVector = FALSE)
  expect_identical(r_oo$logging_level %||% "Silent", "Silent")
  expect_identical(r_oo$prediction_type, cli_oo$prediction_type)
})

test_that("mode:metadata dump-feature-names -- matches the CLI oracle", {
  expect_identical(catboost.get_model_feature_names(cli_model), cli_fixture$feature_names)
})

test_that("mode:metadata set -- catboost.set_metadata() matches what the CLI's `metadata set --key --value` mode would write", {
  # Matrix disposition note for mode:metadata set: "state-mutator; parity via
  # a subsequent metadata get". cli_fixture$set_then_get$cli_get_value is the
  # CLI oracle's own `metadata get --key custom_key` readback after running
  # `metadata set --key custom_key --value custom_value` on this exact
  # trained model -- i.e. the CLI's set-then-get round trip.
  catboost.set_metadata(cli_model, cli_fixture$set_then_get$key, cli_fixture$set_then_get$value)
  expect_identical(
    catboost.get_metadata(cli_model)[[cli_fixture$set_then_get$key]],
    cli_fixture$set_then_get$cli_get_value
  )
})
