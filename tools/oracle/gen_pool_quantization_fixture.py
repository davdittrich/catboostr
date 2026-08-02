#!/usr/bin/env python3
"""Generate the pool-quantization oracle fixture (catboost-8z4.40 / P3.3):
deterministic Pool inputs + the pinned Python catboost==1.2.10 Pool's
observable outputs for is_quantized(), quantize(), and
save_quantization_borders().

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_pool_quantization_fixture.py

quantize() itself has no getter that returns the quantized bin data as a
Python-visible structure, so the only byte-for-byte comparable oracle output
is the borders file save_quantization_borders() writes (matrixnet-format
text: "<flat_feature_idx>\t<border_value>" lines) -- recorded verbatim here
and compared verbatim against the R equivalent's output file. is_quantized()
before/after quantize() and quantize()-on-an-already-quantized-Pool raising
CatBoostError are recorded as the other two observable behaviors.
"""
import json
import os
import sys

from catboost import CatBoostError, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "pool_quantization.json")
BORDERS_PATH = os.path.join(FIXTURE_DIR, "pool_quantization_borders.tsv")

N_ROWS = 8

NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0]
NUM2 = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
LABEL = [0, 1, 0, 1, 0, 1, 1, 0]

DATA = [[NUM1[i], NUM2[i]] for i in range(N_ROWS)]
BORDER_COUNT = 4


def main():
    import catboost

    os.makedirs(FIXTURE_DIR, exist_ok=True)

    pool = Pool(DATA, LABEL, feature_names=["num1", "num2"])
    is_quantized_before = bool(pool.is_quantized())

    pool.quantize(border_count=BORDER_COUNT)
    is_quantized_after = bool(pool.is_quantized())

    pool.save_quantization_borders(BORDERS_PATH)
    with open(BORDERS_PATH) as f:
        borders_text = f.read()

    already_quantized_raises = False
    try:
        pool.quantize(border_count=BORDER_COUNT)
    except CatBoostError:
        already_quantized_raises = True

    unquantized_pool = Pool(DATA, LABEL, feature_names=["num1", "num2"])
    save_on_unquantized_raises = False
    try:
        unquantized_pool.save_quantization_borders(BORDERS_PATH + ".unused")
    except CatBoostError:
        save_on_unquantized_raises = True

    expected = {
        "is_quantized_before": is_quantized_before,
        "is_quantized_after": is_quantized_after,
        "already_quantized_raises": already_quantized_raises,
        "save_on_unquantized_raises": save_on_unquantized_raises,
        "borders_text": borders_text,
    }

    fixture = {
        "catboost_version": catboost.__version__,
        "inputs": {
            "num1": NUM1,
            "num2": NUM2,
            "label": LABEL,
            "border_count": BORDER_COUNT,
        },
        "expected": expected,
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")

    print(f"catboost.__version__={catboost.__version__}", file=sys.stderr)
    print(f"is_quantized_before={is_quantized_before} is_quantized_after={is_quantized_after}", file=sys.stderr)
    print(f"borders_text={borders_text!r}", file=sys.stderr)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()
