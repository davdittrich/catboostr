#!/usr/bin/env python3
"""Generate the pool-structural-operations oracle fixture (catboost-8z4.41 /
P3.4): deterministic Pool inputs + the pinned Python catboost==1.2.10 Pool's
observable outputs for slice(), train_eval_split(), and save().

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_structural_fixture.py

slice():
  Python's Pool.slice(rindex) accepts an arbitrary row-index array. The R
  equivalent (catboost.pool.slice) wraps the pre-existing native
  CatBoostPoolSlice_R entry point (src/catboostr.cpp), which only supports
  contiguous [offset, offset + size) ranges -- so the fixture's rindex is
  always a contiguous range, comparable to both. Also records the
  CatBoostError raised when slicing a Pool with a categorical feature
  (Pool.slice's own non-numeric-feature restriction, mirrored by
  CatBoostPoolSlice_R's CB_ENSURE).

train_eval_split():
  Recorded for two configurations: has_time=True (no shuffle -- a pure,
  RNG-free contiguous split, the strongest possible cross-language
  comparison) and is_classification=True (stratified split, has_time=False,
  fixed default PartitionRandSeed=0 shuffle -- comparable bit-for-bit only
  because R's CatBoostPoolTrainEvalSplit_R (src/catboostr.cpp) reimplements
  TrainEvalSplit() against the exact same core NCB entry points, compiled
  from the same pinned vendor/catboost snapshot).

save():
  Python's Pool.save() requires a quantized Pool and writes CatBoost's own
  binary quantized-pool format (_catboost.pyx _save() ->
  SaveQuantizedPool(TDataProviderPtr, fname), catboost/private/libs/
  quantized_pool/serialization.h). The fixture quantizes and saves a Pool,
  then re-loads the saved file (Pool(data="quantized://...")) and records
  its observable shape/label -- the only way to compare R's writer against
  Python's reader (R's own loader can't read "quantized://" paths; see the
  ticket report for why byte-identity was not asserted).
"""
import json
import os
import sys

from catboost import CatBoostError, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "pool_structural.json")

N_ROWS = 8

NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0]
NUM2 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
LABEL = [0, 1, 0, 1, 0, 1, 1, 0]

DATA = [[NUM1[i], NUM2[i]] for i in range(N_ROWS)]

SLICE_OFFSET = 2
SLICE_SIZE = 4

EVAL_FRACTION = 0.25
BORDER_COUNT = 4


def main():
    import catboost

    os.makedirs(FIXTURE_DIR, exist_ok=True)

    # --- slice(): numeric-only pool, contiguous range, happy path ---
    pool = Pool(DATA, LABEL, feature_names=["num1", "num2"])
    rindex = list(range(SLICE_OFFSET, SLICE_OFFSET + SLICE_SIZE))
    sliced = pool.slice(rindex)
    slice_features = sliced.get_features()
    slice_label = sliced.get_label()

    # --- slice(): categorical feature present -> CatBoostError ---
    cat_pool = Pool(
        [[NUM1[i], "a" if i % 2 == 0 else "b"] for i in range(N_ROWS)],
        LABEL,
        cat_features=[1],
        feature_names=["num1", "cat1"],
    )
    slice_on_categorical_raises = False
    slice_on_categorical_message = None
    try:
        cat_pool.slice(rindex)
    except CatBoostError as exc:
        slice_on_categorical_raises = True
        slice_on_categorical_message = str(exc)

    # --- train_eval_split(): has_time=True, no shuffle, pure index split ---
    pool_for_split = Pool(DATA, LABEL, feature_names=["num1", "num2"])
    train_pool, eval_pool = pool_for_split.train_eval_split(
        has_time=True,
        is_classification=False,
        eval_fraction=EVAL_FRACTION,
        save_eval_pool=True,
    )
    has_time_split = {
        "train_label": train_pool.get_label(),
        "train_features": train_pool.get_features(),
        "eval_label": eval_pool.get_label(),
        "eval_features": eval_pool.get_features(),
    }

    # --- train_eval_split(): is_classification=True, stratified, shuffled ---
    pool_for_stratified = Pool(DATA, LABEL, feature_names=["num1", "num2"])
    strat_train_pool, strat_eval_pool = pool_for_stratified.train_eval_split(
        has_time=False,
        is_classification=True,
        eval_fraction=EVAL_FRACTION,
        save_eval_pool=True,
    )
    stratified_split = {
        "train_label": strat_train_pool.get_label(),
        "train_features": strat_train_pool.get_features(),
        "eval_label": strat_eval_pool.get_label(),
        "eval_features": strat_eval_pool.get_features(),
    }

    # --- train_eval_split(): save_eval_pool=False ---
    pool_for_no_eval = Pool(DATA, LABEL, feature_names=["num1", "num2"])
    train_only_pool, eval_none = pool_for_no_eval.train_eval_split(
        has_time=True,
        is_classification=False,
        eval_fraction=EVAL_FRACTION,
        save_eval_pool=False,
    )
    save_eval_pool_false = {
        "train_label": train_only_pool.get_label(),
        "eval_is_none": eval_none is None,
    }

    # --- save(): quantize, save, reload, compare shape/label ---
    save_pool = Pool(DATA, LABEL, feature_names=["num1", "num2"])
    save_on_unquantized_raises = False
    try:
        save_pool.save(os.path.join(FIXTURE_DIR, "pool_structural_save.unused"))
    except CatBoostError:
        save_on_unquantized_raises = True

    save_pool.quantize(border_count=BORDER_COUNT)
    quantized_path = os.path.join(FIXTURE_DIR, "pool_structural_quantized.bin")
    save_pool.save(quantized_path)

    reloaded = Pool(data="quantized://" + quantized_path)
    save_roundtrip = {
        "num_row": reloaded.num_row(),
        "num_col": reloaded.num_col(),
        "label": reloaded.get_label(),
    }
    # The reference file itself is regenerated by this script when re-run and
    # is not needed after generating the fixture -- R's own gen step writes
    # its own quantized file and re-derives comparable numbers independently.
    os.remove(quantized_path)

    expected = {
        "slice": {
            "features": slice_features.tolist(),
            "label": slice_label.tolist(),
        },
        "slice_on_categorical": {
            "raises": slice_on_categorical_raises,
            "message": slice_on_categorical_message,
        },
        "has_time_split": {
            "train_label": has_time_split["train_label"].tolist(),
            "train_features": has_time_split["train_features"].tolist(),
            "eval_label": has_time_split["eval_label"].tolist(),
            "eval_features": has_time_split["eval_features"].tolist(),
        },
        "stratified_split": {
            "train_label": stratified_split["train_label"].tolist(),
            "train_features": stratified_split["train_features"].tolist(),
            "eval_label": stratified_split["eval_label"].tolist(),
            "eval_features": stratified_split["eval_features"].tolist(),
        },
        "save_eval_pool_false": {
            "train_label": save_eval_pool_false["train_label"].tolist(),
            "eval_is_none": save_eval_pool_false["eval_is_none"],
        },
        "save_on_unquantized_raises": save_on_unquantized_raises,
        "save_roundtrip": {
            "num_row": save_roundtrip["num_row"],
            "num_col": save_roundtrip["num_col"],
            "label": save_roundtrip["label"].tolist(),
        },
    }

    fixture = {
        "catboost_version": catboost.__version__,
        "inputs": {
            "num1": NUM1,
            "num2": NUM2,
            "label": LABEL,
            "slice_offset": SLICE_OFFSET,
            "slice_size": SLICE_SIZE,
            "eval_fraction": EVAL_FRACTION,
            "border_count": BORDER_COUNT,
        },
        "expected": expected,
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"slice_on_categorical_message={slice_on_categorical_message!r}", file=sys.stderr)
    print(f"has_time_split train/eval sizes: {len(has_time_split['train_label'])}/{len(has_time_split['eval_label'])}", file=sys.stderr)
    print(f"stratified_split train/eval sizes: {len(stratified_split['train_label'])}/{len(stratified_split['eval_label'])}", file=sys.stderr)
    print(f"save_roundtrip={save_roundtrip}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
