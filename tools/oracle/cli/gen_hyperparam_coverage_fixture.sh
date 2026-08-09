#!/usr/bin/env bash
# P10.A (catboost-8z4.115): CLI-oracle fixture for the shared fit/eval-feature/
# model-based-eval/select-features hyperparameter flag family (matrix rows
# flag:--fold-len-multiplier, --fold-permutation-block, --boost-from-average,
# --boosting-type, --od-pval/--od-wait/--od-type, --model-shrink-rate/-mode,
# --langevin/--diffusion-temperature, --rsm, --leaf-estimation-iterations/
# -backtracking/-method, --depth/-n, --min-data-in-leaf, --l2-leaf-reg,
# --bayesian-matrix-reg, --model-size-reg, --sparse-features-conflict-fraction,
# --random-strength, --score-function, --bootstrap-type, --sampling-unit,
# --bagging-temperature/--tmp, --sampling-frequency, --monotone-constraints,
# --feature-weights, --penalties-coefficient, --first-feature-use-penalties,
# --per-object-feature-penalties, --max-ctr-complexity, --simple-ctr,
# --combinations-ctr, --feature-ctr/--per-feature-ctr, --ctr-target-border-count,
# --counter-calc-method, --ctr-leaf-count-limit, --ctr-history-unit,
# --store-all-simple-ctr, --one-hot-max-size, --ignore-features/-I,
# --auto-class-weights, --force-unit-auto-pair-weights, --border-count/-x,
# --per-float-feature-quantization, --feature-border-type/--grid,
# --nan-mode, --used-ram-limit, --random-seed/--seed/-r, --task-type,
# --detailed-profile, --final-ctr-computation-mode, --allow-writing-files).
#
# All these flags are mutually-compatible (verified empirically via
# tmp/probe_combined.R during P10.A development) on CPU against the shared
# 40-row smoke fixture (tests/fixtures/oracle-cli/smoke_data.csv +
# smoke.cd), so one fit+calc run closes the whole family in one shot,
# mirroring test_param_family_coverage.R's Python-oracle batching approach
# but against the CLI binary.
#
# Usage: tools/oracle/cli/acquire.sh && tools/oracle/cli/gen_hyperparam_coverage_fixture.sh
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
  --fold-len-multiplier 1.5 --fold-permutation-block 2 \
  --boost-from-average true --boosting-type Ordered \
  --od-pval 0.01 --od-wait 2 --od-type IncToDec \
  --model-shrink-rate 0.1 --model-shrink-mode Constant \
  --langevin true --diffusion-temperature 100 \
  --rsm 0.9 \
  --leaf-estimation-iterations 2 --leaf-estimation-backtracking AnyImprovement \
  --depth 3 --min-data-in-leaf 1 --l2-leaf-reg 3 \
  --bayesian-matrix-reg 0.1 --model-size-reg 0.5 \
  --sparse-features-conflict-fraction 0.0 --random-strength 1.0 \
  --leaf-estimation-method Newton --score-function Cosine \
  --bootstrap-type Bayesian --sampling-unit Object --bagging-temperature 0.5 \
  --sampling-frequency PerTree \
  --monotone-constraints "(1,0,0)" --feature-weights "(1,1,1)" \
  --penalties-coefficient 1.0 --first-feature-use-penalties "(0,0,0)" \
  --per-object-feature-penalties "(0,0,0)" \
  --max-ctr-complexity 2 --simple-ctr Borders --combinations-ctr Borders \
  --per-feature-ctr "2:Borders:Prior=0/1" \
  --ctr-target-border-count 1 --counter-calc-method Full \
  --ctr-leaf-count-limit 100 --ctr-history-unit Sample \
  --one-hot-max-size 2 \
  --auto-class-weights Balanced --force-unit-auto-pair-weights \
  --border-count 32 --per-float-feature-quantization "0:border_count=32" \
  --feature-border-type GreedyLogSum --nan-mode Min \
  --used-ram-limit 1gb --task-type CPU --detailed-profile \
  --final-ctr-computation-mode Default --allow-writing-files true

"${BIN}" calc \
  --input-path "${FIX}/smoke_data.csv" --column-description "${FIX}/smoke.cd" \
  --delimiter , --has-header -m "${MODEL}" -T 1 \
  --output-path "${EVALOUT}" --output-columns RawFormulaVal --prediction-type RawFormulaVal

tail -n +2 "${EVALOUT}" > "${FIX}/hyperparam_coverage_predictions.tsv"
rm -f "${MODEL}" "${EVALOUT}"
echo "Wrote ${FIX}/hyperparam_coverage_predictions.tsv"
