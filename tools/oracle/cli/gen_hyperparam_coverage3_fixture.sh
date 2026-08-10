#!/usr/bin/env bash
# P10.A (catboost-8z4.115) -- third CLI-oracle batch: flag:--grow-policy +
# flag:--max-leaves (Lossguide trees, incompatible with the Ordered boosting
# used by gen_hyperparam_coverage2_fixture.sh's approx-on-full-history row).
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
  --grow-policy Lossguide --max-leaves 8

"${BIN}" calc \
  --input-path "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header -m "${MODEL}" -T 1 \
  --output-path "${EVALOUT}" --output-columns RawFormulaVal --prediction-type RawFormulaVal

tail -n +2 "${EVALOUT}" > "${FIX}/hyperparam_coverage3_predictions.tsv"
rm -f "${MODEL}" "${EVALOUT}"
echo "Wrote ${FIX}/hyperparam_coverage3_predictions.tsv"
