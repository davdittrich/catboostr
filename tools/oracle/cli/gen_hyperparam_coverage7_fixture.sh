#!/usr/bin/env bash
# P10.A (catboost-8z4.115) -- seventh CLI-oracle batch: flag:--store-all-simple-ctr,
# flag:--ignore-features/-I, flag:--has-time, flag:--allow-const-label.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BIN=tools/oracle/cli/bin/catboost-v1.2.10
FIX=tests/fixtures/oracle-cli
MODEL=$(mktemp)
EVALOUT=$(mktemp)

"${BIN}" fit \
  --learn-set "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header --loss-function Logloss -i 20 \
  --model-file "${MODEL}" --logging-level Silent -T 1 --random-seed 42 \
  --store-all-simple-ctr -I 1 --has-time --allow-const-label

"${BIN}" calc \
  --input-path "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header -m "${MODEL}" -T 1 \
  --output-path "${EVALOUT}" --output-columns RawFormulaVal --prediction-type RawFormulaVal

tail -n +2 "${EVALOUT}" > "${FIX}/hyperparam_coverage7_predictions.tsv"
rm -f "${MODEL}" "${EVALOUT}"
echo "Wrote ${FIX}/hyperparam_coverage7_predictions.tsv"
