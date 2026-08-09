#!/usr/bin/env bash
# P10.A (catboost-8z4.115) -- fifth CLI-oracle batch: flag:--tokenizers,
# flag:--dictionaries, flag:--feature-calcers, flag:--embedding-processing,
# flag:--text-processing. On the smoke fixture (no text/embedding feature
# columns) these options are accepted as configuration with no feature data
# to act on, so predictions are identical to a plain run -- the differential
# assertion is that R accepts and passes through the same option shapes the
# CLI does (bit-identical predictions), not a claim that text/embedding
# processing changes this particular model. --text-processing is a separate,
# mutually-exclusive-with-the-others run (native rejects combining
# text_processing with tokenizers/dictionaries/feature_calcers directly).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BIN=tools/oracle/cli/bin/catboost-v1.2.10
FIX=tests/fixtures/oracle-cli
MODEL=$(mktemp)
MODEL2=$(mktemp)
EVALOUT=$(mktemp)
EVALOUT2=$(mktemp)

"${BIN}" fit \
  --learn-set "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header --loss-function Logloss -i 20 \
  --model-file "${MODEL}" --logging-level Silent -T 1 --random-seed 42 \
  --tokenizers "Space:delimiter= " \
  --dictionaries "Word:token_level_type=Word" \
  --feature-calcers BoW \
  --embedding-processing "{}"

"${BIN}" calc \
  --input-path "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header -m "${MODEL}" -T 1 \
  --output-path "${EVALOUT}" --output-columns RawFormulaVal --prediction-type RawFormulaVal
tail -n +2 "${EVALOUT}" > "${FIX}/hyperparam_coverage5_predictions.tsv"

"${BIN}" fit \
  --learn-set "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header --loss-function Logloss -i 20 \
  --model-file "${MODEL2}" --logging-level Silent -T 1 --random-seed 42 \
  --text-processing "{}"

"${BIN}" calc \
  --input-path "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header -m "${MODEL2}" -T 1 \
  --output-path "${EVALOUT2}" --output-columns RawFormulaVal --prediction-type RawFormulaVal
tail -n +2 "${EVALOUT2}" > "${FIX}/hyperparam_coverage6_predictions.tsv"

rm -f "${MODEL}" "${MODEL2}" "${EVALOUT}" "${EVALOUT2}"
echo "Wrote ${FIX}/hyperparam_coverage5_predictions.tsv and hyperparam_coverage6_predictions.tsv"
