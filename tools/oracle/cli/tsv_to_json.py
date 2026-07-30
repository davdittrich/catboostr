#!/usr/bin/env python3
"""Convert a `catboost calc` TSV output file to the JSON array format
committed as tests/fixtures/oracle-cli/smoke_predictions.json: one object
per row, keyed by the TSV header, numeric columns parsed as float/int with
no added truncation beyond what the CLI itself already printed (see the
precision note in smoke_metadata.json -- the CLI's ~10-significant-digit
output is the precision ceiling, not this converter).

Usage: tsv_to_json.py <input.tsv> <output.json>
"""
import csv
import json
import sys


def convert(value):
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [{k: convert(v) for k, v in row.items()} for row in reader]
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
