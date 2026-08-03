#!/usr/bin/env python3
"""Read back a quantized-pool binary file (as written by either Python's
Pool.save() or R's catboost.pool.save(), both of which call the same core
SaveQuantizedPool entry point) and print its observable shape/label as JSON
on stdout. Used by tests/testthat/test_pool_structural.R (catboost-8z4.41 /
P3.4) to prove catboost.pool.save()'s output is actually readable by the
pinned Python oracle.

R could read the file back itself -- catboost.load_pool("quantized://" + path,
column_description = "") does work -- but that would only prove the in-tree
build can re-read bytes it just wrote. Reading it with the independently
built, pinned catboost==1.2.10 wheel is what actually pins the on-disk format,
so this helper stays.

Usage: uv run --frozen --project tools/oracle python3 tools/oracle/read_quantized_pool.py <path>
"""
import json
import sys

from catboost import Pool


def main():
    path = sys.argv[1]
    pool = Pool(data="quantized://" + path)
    print(json.dumps({
        "num_row": pool.num_row(),
        "num_col": pool.num_col(),
        "label": pool.get_label().tolist(),
    }))


if __name__ == "__main__":
    main()
