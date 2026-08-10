#!/usr/bin/env bash
# Regenerate the CLI oracle fixture exercising a non-default --shap-calc-type
# (catboost-inm, followup to catboost-8z4.115/P10.A): closes the
# select-features half of flag:--shap-calc-type. The existing
# test_select_features.R fixture (tools/oracle/gen_select_features_fixture.py)
# is Python-oracle-generated and only ever ran shap_calc_type = "Regular"
# (the default), so nothing in the suite differentially tested this argument
# at a non-default value. The Python oracle binary is unavailable in this
# worktree, but select-features is also a full CLI mode (RecursiveByShapValues
# is the algorithm that actually consumes shap_calc_type), so this uses the
# CLI oracle instead -- spec 4.3's CLI path, relative 1e-6 tolerance (matching
# test_cli_hyperparam_coverage.R's CLI-oracle convention).
#
# Reuses the smoke dataset (tests/fixtures/oracle-cli/smoke_data.csv/.cd).
#
# Usage: tools/oracle/cli/gen_select_features_shap_calc_type_fixture.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN="${SCRIPT_DIR}/bin/catboost-v1.2.10"
OUT_DIR="${REPO_ROOT}/tests/fixtures/oracle-cli"

if [[ ! -x "${BIN}" ]]; then
  echo "FATAL: ${BIN} not found. Run tools/oracle/cli/acquire.sh first." >&2
  exit 1
fi
if [[ ! -f "${OUT_DIR}/smoke_data.csv" ]]; then
  echo "FATAL: ${OUT_DIR}/smoke_data.csv not found. Run" >&2
  echo "  tools/oracle/cli/gen_smoke_fixture.sh" >&2
  echo "first -- this fixture reuses that dataset." >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
cd "${WORK_DIR}"

cp "${OUT_DIR}/smoke_data.csv" smoke_data.csv
cp "${OUT_DIR}/smoke.cd" smoke.cd

run() {
  local out_file="$1"
  local result_file="$2"
  "${BIN}" select-features \
    --learn-set smoke_data.csv --test-set smoke_data.csv \
    --column-description smoke.cd --delimiter , --has-header \
    --loss-function Logloss -i 20 --depth 4 --learning-rate 0.1 \
    --random-seed 42 -T 1 --logging-level Silent \
    --features-for-select 0-1 --num-features-to-select 1 \
    --features-selection-algorithm RecursiveByShapValues \
    --shap-calc-type Exact \
    --features-selection-steps 2 \
    --features-selection-result-path "${result_file}" \
    --model-file "${out_file}"
}

run model.cbm select_features_shap_calc_type.json
run model_rerun.cbm select_features_shap_calc_type_rerun.json

if ! diff -q select_features_shap_calc_type.json select_features_shap_calc_type_rerun.json > /dev/null; then
  echo "FATAL: select-features --shap-calc-type Approximate is not deterministic across runs; fixture rejected." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cp select_features_shap_calc_type.json "${OUT_DIR}/select_features_shap_calc_type.json"

echo "Wrote ${OUT_DIR}/select_features_shap_calc_type.json (determinism re-run: identical)"
