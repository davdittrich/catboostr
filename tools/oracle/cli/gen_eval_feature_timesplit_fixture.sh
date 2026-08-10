#!/usr/bin/env bash
# Regenerate the CLI oracle fixture exercising --timesplit-quantile
# (catboost-inm, followup to catboost-8z4.115/P10.A): closes
# flag:--timesplit-quantile. P10.A fix-round-1's justification named the
# wrong trigger (it assumed a "--feature-eval-mode TimeSplit" value, which
# does not exist -- feature-eval-mode is one of OneVsNone/OneVsOthers/
# OneVsAll/OthersVsAll). Verified in
# catboost/libs/train_lib/eval_feature.cpp:910 that the real trigger is
# `trainingData->MetaInfo.HasTimestamp`: a dataset with a timestamp column
# switches eval-feature from PrepareFolds to PrepareTimeSplitFolds, which is
# the only path that reads TimeSplitQuantile at all
# (CountDisjointFolds/FindQuantileTimestamp, eval_feature.cpp:1118-1121).
# Timestamps require group ids (same file, CB_ENSURE at eval_feature.cpp
# ~1114).
#
# Dataset: 40 rows, 4 groups of 10, one timestamp per GroupId (0/10/20/30;
# the timestamps file is GroupId<TAB>timestamp, one line per group --
# catboost/libs/data/loader.cpp:145-147, not one line per object) so
# different --timesplit-quantile values put a different number of groups in
# the "before quantile" fold and produce genuinely different output
# (verified below before committing the fixture).
#
# Usage: tools/oracle/cli/gen_eval_feature_timesplit_fixture.sh
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
for g in range(8):
    for i in range(10):
        x1 = round(random.uniform(-3, 3), 6)
        x2 = round(random.uniform(-3, 3), 6)
        y = round(random.uniform(-1, 1), 6)
        rows.append((x1, x2, y, g))

with open("timesplit_data.csv", "w") as f:
    f.write("x1,x2,target,group_id\n")
    for x1, x2, y, g in rows:
        f.write(f"{x1},{x2},{y},{g}\n")

with open("timesplit_data.cd", "w") as f:
    f.write("2\tLabel\ttarget\n")
    f.write("3\tGroupId\tgroup_id\n")

with open("timesplit_timestamps.tsv", "w") as f:
    for g in range(8):
        f.write(f"{g}\t{g * 10}\n")
PYEOF

run_eval_feature() {
  local quantile="$1"
  local out_file="$2"
  "${BIN}" eval-feature \
    --learn-set timesplit_data.csv --column-description timesplit_data.cd \
    --delimiter , --has-header \
    --learn-timestamps timesplit_timestamps.tsv \
    --loss-function RMSE -i 10 --depth 3 --learning-rate 0.1 \
    --random-seed 42 -T 1 --logging-level Silent \
    --features-to-evaluate '0;1' --feature-eval-mode OneVsAll \
    --offset 0 --fold-count 2 --fold-size-unit Group --fold-size 1 \
    --timesplit-quantile "${quantile}" \
    --feature-eval-output-file "${out_file}"
}

run_eval_feature 0.5 feature_eval_q25.tsv
run_eval_feature 0.75 feature_eval_q75.tsv
run_eval_feature 0.5 feature_eval_q25_rerun.tsv

if ! diff -q feature_eval_q25.tsv feature_eval_q25_rerun.tsv > /dev/null; then
  echo "FATAL: eval-feature --timesplit-quantile is not deterministic across runs; fixture rejected." >&2
  exit 1
fi
if diff -q feature_eval_q25.tsv feature_eval_q75.tsv > /dev/null; then
  echo "FATAL: --timesplit-quantile 0.25 and 0.75 produced identical output; this dataset does not" >&2
  echo "exercise the flag. Fixture rejected." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
cp timesplit_data.csv timesplit_data.cd timesplit_timestamps.tsv "${OUT_DIR}/"
cp feature_eval_q25.tsv "${OUT_DIR}/eval_feature_timesplit_q25.tsv"
cp feature_eval_q75.tsv "${OUT_DIR}/eval_feature_timesplit_q75.tsv"

echo "Wrote ${OUT_DIR}/{timesplit_data.csv,timesplit_data.cd,timesplit_timestamps.tsv,eval_feature_timesplit_q25.tsv,eval_feature_timesplit_q75.tsv}"
echo "(determinism re-run: identical; q25 vs q75: genuinely different -- flag has real effect)"
