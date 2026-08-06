#!/usr/bin/env bash
# Regenerate the CLI oracle fixture for the `eval-metrics` mode
# (mode:eval-metrics row, catboost-8z4.77) from the pinned catboost binary.
#
# `eval-metrics` shares R's catboost.eval_metrics() core entry point
# (CatBoostEvalMetrics_R -> plot.h's TMetricsPlotCalcer, the same code
# CLI's mode_eval_metrics.cpp drives), so this fixture pins the CLI's own
# printed TSV as the oracle -- spec §4.3's CLI path, relative 1e-9
# tolerance (CLI prints ~10 significant digits, no --precision flag).
#
# The dataset is *reused* from the Python oracle's generator, exactly as
# gen_smoke_fixture.sh does, so every CLI fixture in this directory describes
# the same data.
#
# Usage: tools/oracle/cli/gen_eval_metrics_fixture.sh
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

cp "${OUT_DIR}/smoke_data.csv" "${WORK_DIR}/smoke_data.csv"
cp "${OUT_DIR}/smoke.cd" "${WORK_DIR}/smoke.cd"

(
  cd "${WORK_DIR}"
  "${BIN}" fit \
    --learn-set smoke_data.csv --column-description smoke.cd --delimiter , \
    --has-header --loss-function Logloss -i 20 --depth 4 --learning-rate 0.1 \
    --random-seed 42 -T 1 --logging-level Silent --model-file model.bin
)

run_eval_metrics() {
  local out_file="$1"
  (
    cd "${WORK_DIR}"
    "${BIN}" eval-metrics \
      --input-path smoke_data.csv --column-description smoke.cd --delimiter , \
      --has-header -m model.bin --metrics 'Logloss,AUC' -T 1 --eval-period 5 \
      --output-path "${out_file}"
  )
}

run_eval_metrics eval_metrics.tsv
run_eval_metrics eval_metrics_rerun.tsv

if ! diff -q "${WORK_DIR}/eval_metrics.tsv" "${WORK_DIR}/eval_metrics_rerun.tsv" > /dev/null; then
  echo "FATAL: eval-metrics is not deterministic across runs; fixture rejected." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cp "${WORK_DIR}/eval_metrics.tsv" "${OUT_DIR}/eval_metrics.tsv"

echo "Wrote ${OUT_DIR}/eval_metrics.tsv (determinism re-run: identical)"
