#!/usr/bin/env bash
# Regenerate the CLI oracle fixture exercising the pairs/graph sidecar
# columns (catboost-inm, followup to catboost-8z4.115/P10.A): closes
# flag:--learn-pairs, flag:--test-pairs, flag:--input-pairs (via a PairLogit
# fit+calc) and flag:--learn-graph, flag:--test-graph, flag:--input-graph
# (via a separate RMSE fit+calc), since all six load the same dsv-flat
# WinnerId<TAB>LoserId format (catboost/libs/data/pairs_data_loaders.cpp)
# into the same TRawPairsData structure (catboost/libs/data/graph.cpp:
# --learn-graph/--test-graph feed the exact same loader as --learn-pairs/
# --test-pairs, just consumed for aggregation features instead of
# pairwise-ranking loss).
#
# Two separate fit+calc runs, not one: catboost/libs/data/
# proceed_pool_in_blocks.h:95 refuses pairs+graph together in the
# block-streaming loader calc/fstr/dataset-statistics/eval-metrics use, and
# a model trained with a graph unconditionally requires a graph at predict
# time too (its MeanAggregated/MinAggregated/MaxAggregated features have
# nowhere else to come from) -- so pairs and graph each need their own
# model to stay a clean, real parity signal instead of an artifact of
# reusing one model for both.
#
# Dataset: 20 rows, 2 groups of 10 (GroupId 0/1), 2 numeric features. Pairs:
# 5 winner/loser pairs per group (drives PairLogit, so pairs actually affect
# the fitted model). Graph: a per-group chain (row i -> row i+1) so
# graph-aggregation features are non-trivial and change predictions vs. a
# graph-less fit.
#
# Usage: tools/oracle/cli/gen_grouped_data_fixture.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN="${SCRIPT_DIR}/bin/catboost-v1.2.10"
OUT_DIR="${REPO_ROOT}/tests/fixtures/oracle-cli"

if [[ ! -x "${BIN}" ]]; then
  echo "FATAL: ${BIN} not found. Run tools/oracle/cli/acquire.sh first." >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
cd "${WORK_DIR}"

python3 - <<'PYEOF'
import random
random.seed(42)
rows = []
for g in range(2):
    for i in range(10):
        x1 = round(random.uniform(-3, 3), 6)
        x2 = round(random.uniform(-3, 3), 6)
        y = round(random.uniform(-1, 1), 6)
        rows.append((x1, x2, y, g))

with open("grouped_data.csv", "w") as f:
    f.write("x1,x2,target,group_id\n")
    for x1, x2, y, g in rows:
        f.write(f"{x1},{x2},{y},{g}\n")

with open("grouped_data.cd", "w") as f:
    f.write("2\tLabel\ttarget\n")
    f.write("3\tGroupId\tgroup_id\n")

# 5 winner/loser pairs per group of 10 (flat absolute row indices).
with open("grouped_pairs.tsv", "w") as f:
    for g in range(2):
        base = g * 10
        for i in range(0, 10, 2):
            f.write(f"{base + i}\t{base + i + 1}\n")

# Per-group chain graph (row i -> row i+1), used for aggregation features.
with open("grouped_graph.tsv", "w") as f:
    for g in range(2):
        base = g * 10
        for i in range(9):
            f.write(f"{base + i}\t{base + i + 1}\n")
PYEOF

# --- pairs model (PairLogit) --------------------------------------------
"${BIN}" fit \
  --learn-set grouped_data.csv --test-set grouped_data.csv \
  --column-description grouped_data.cd --delimiter , --has-header \
  --learn-pairs grouped_pairs.tsv --test-pairs grouped_pairs.tsv \
  --loss-function PairLogit -i 20 --depth 4 --learning-rate 0.1 \
  --random-seed 42 -T 1 --model-file grouped_pairs_model.cbm --logging-level Silent

"${BIN}" calc \
  --input-path grouped_data.csv --column-description grouped_data.cd \
  --delimiter , --has-header \
  --input-pairs grouped_pairs.tsv \
  -m grouped_pairs_model.cbm --output-path grouped_predictions_pairs.tsv \
  --prediction-type RawFormulaVal

# --- graph model (RMSE) -------------------------------------------------
"${BIN}" fit \
  --learn-set grouped_data.csv --test-set grouped_data.csv \
  --column-description grouped_data.cd --delimiter , --has-header \
  --learn-graph grouped_graph.tsv --test-graph grouped_graph.tsv \
  --loss-function RMSE -i 20 --depth 4 --learning-rate 0.1 \
  --random-seed 42 -T 1 --model-file grouped_graph_model.cbm --logging-level Silent

"${BIN}" calc \
  --input-path grouped_data.csv --column-description grouped_data.cd \
  --delimiter , --has-header \
  --input-graph grouped_graph.tsv \
  -m grouped_graph_model.cbm --output-path grouped_predictions_graph.tsv \
  --prediction-type RawFormulaVal

mkdir -p "${OUT_DIR}"
cp grouped_data.csv grouped_data.cd grouped_pairs.tsv grouped_graph.tsv "${OUT_DIR}/"
python3 "${SCRIPT_DIR}/tsv_to_json.py" grouped_predictions_pairs.tsv "${OUT_DIR}/grouped_data_predictions_pairs.json"
python3 "${SCRIPT_DIR}/tsv_to_json.py" grouped_predictions_graph.tsv "${OUT_DIR}/grouped_data_predictions_graph.json"

echo "Wrote ${OUT_DIR}/{grouped_data.csv,grouped_data.cd,grouped_pairs.tsv,grouped_graph.tsv,grouped_data_predictions_pairs.json,grouped_data_predictions_graph.json}"
