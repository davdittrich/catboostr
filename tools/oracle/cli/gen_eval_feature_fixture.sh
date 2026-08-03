#!/usr/bin/env bash
# Regenerate the CLI oracle fixture for the `eval-feature` mode (P4.7,
# catboost-8z4.56) from the pinned catboost binary.
#
# `eval-feature` has no Python-package counterpart (the Python surface only
# offers CatBoost.select_features, a different algorithm and Phase 5 scope), so
# the CLI is the sole oracle for it -- §4.3's CLI path, relative 1e-9
# tolerance.
#
# The 40-row dataset is *reused* from the Python oracle's generator, exactly as
# gen_smoke_fixture.sh does, so every CLI fixture in this directory describes
# the same data.
#
# Unlike gen_smoke_fixture.sh this script does NOT pipe its output through
# tsv_to_json.py: the eval-feature summary's "best iteration in each fold" and
# "feature set" columns are comma-joined integer lists, which that converter
# would turn into strings for multi-element values and ints for single-element
# ones. The raw TSV is committed instead -- lossless, and byte-for-byte what
# the CLI emitted.
#
# Usage: tools/oracle/cli/gen_eval_feature_fixture.sh
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

run_eval_feature() {
  local out_file="$1"
  local train_dir="$2"
  (
    cd "${WORK_DIR}"
    "${BIN}" eval-feature \
      --learn-set smoke_data.csv --column-description smoke.cd --delimiter , \
      --has-header --loss-function Logloss -i 20 --depth 4 --learning-rate 0.1 \
      --random-seed 42 -T 1 --logging-level Silent \
      --features-to-evaluate '0;1' --feature-eval-mode OneVsAll \
      --offset 0 --fold-count 2 --fold-size-unit Object --fold-size 10 \
      --feature-eval-output-file "${out_file}" --train-dir "${train_dir}"
  )
}

run_eval_feature feature_eval.tsv tdir1
run_eval_feature feature_eval_rerun.tsv tdir2

if ! diff -q "${WORK_DIR}/feature_eval.tsv" "${WORK_DIR}/feature_eval_rerun.tsv" > /dev/null; then
  echo "FATAL: eval-feature is not deterministic across runs; fixture rejected." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cp "${WORK_DIR}/feature_eval.tsv" "${OUT_DIR}/eval_feature_summary.tsv"

echo "Wrote ${OUT_DIR}/eval_feature_summary.tsv (determinism re-run: identical)"
echo "eval_feature_metadata.json records the exact command and is maintained by"
echo "hand -- update it if the command above changes."
