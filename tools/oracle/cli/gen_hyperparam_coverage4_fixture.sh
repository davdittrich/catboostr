#!/usr/bin/env bash
# P10.A (catboost-8z4.115) -- fourth CLI-oracle batch: flag:--classes-count,
# flag:--class-names, flag:--class-weights (MultiClass-only; incompatible
# with the Logloss loss function used by the other 3 batches).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BIN=tools/oracle/cli/bin/catboost-v1.2.10
FIX=tests/fixtures/oracle-cli
MODEL=$(mktemp)
EVALOUT=$(mktemp)

"${BIN}" fit \
  --learn-set "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header --loss-function MultiClass -i 20 \
  --model-file "${MODEL}" --logging-level Silent -T 1 --random-seed 42 \
  --classes-count 2 --class-names "0,1" --class-weights "1,1"

"${BIN}" calc \
  --input-path "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header -m "${MODEL}" -T 1 \
  --output-path "${EVALOUT}" --output-columns RawFormulaVal --prediction-type RawFormulaVal

tail -n +2 "${EVALOUT}" > "${FIX}/hyperparam_coverage4_predictions.tsv"
rm -f "${MODEL}" "${EVALOUT}"
echo "Wrote ${FIX}/hyperparam_coverage4_predictions.tsv"
