#!/usr/bin/env bash
# P10.A (catboost-8z4.115) -- second, smaller CLI-oracle batch for the
# fit/eval-feature/model-based-eval/select-features hyperparameter flags that
# are NOT mutually compatible with the main batch
# (gen_hyperparam_coverage_fixture.sh): flag:--approx-on-full-history
# (requires Ordered boosting, conflicting with the main batch's
# monotone-constraints), flag:--bootstrap-type MVS + flag:--subsample +
# flag:--mvs-reg (Bayesian bootstrap used in the main batch is mutually
# exclusive with MVS/subsample/mvs_reg), flag:--posterior-sampling
# (rejects an explicit --diffusion-temperature, which the main batch sets
# for flag:--langevin), and flag:--target-border.
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
  --boosting-type Ordered --approx-on-full-history \
  --bootstrap-type MVS --subsample 0.8 --mvs-reg 0.2 \
  --posterior-sampling true \
  --target-border 0.5

"${BIN}" calc \
  --input-path "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header -m "${MODEL}" -T 1 \
  --output-path "${EVALOUT}" --output-columns RawFormulaVal --prediction-type RawFormulaVal

tail -n +2 "${EVALOUT}" > "${FIX}/hyperparam_coverage2_predictions.tsv"
rm -f "${MODEL}" "${EVALOUT}"
echo "Wrote ${FIX}/hyperparam_coverage2_predictions.tsv"
