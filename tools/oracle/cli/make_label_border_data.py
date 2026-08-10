#!/usr/bin/env python3
"""Build tests/fixtures/oracle-cli/label_border_data.csv: same num1/num2/cat1
feature columns as smoke_data.csv, but with a continuous (soft-label)
target spread evenly over [0, 1) (i / n) instead of smoke's hard {0, 1}, so
a --label-border of 0.5 (the pre-existing default) and 0.3 produce
genuinely different binarizations for the rows with target in (0.3, 0.5] --
proving catboost.get_roc_curve()'s new label_border argument actually
changes the result, not just accepts it. Values are all distinct (i / n)
rather than repeating a handful of levels, since CatBoost's target-type
autodetection treats a target with few unique values as a *classification*
label set and requires it to have exactly 2 classes.
"""
import csv
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "oracle-cli"


def main() -> None:
    rows = list(csv.reader((FIXTURE_DIR / "smoke_data.csv").open()))
    header, data = rows[0], rows[1:]
    n = len(data)
    out = [header]
    for i, row in enumerate(data):
        out.append(row[:-1] + [str(i / n)])
    with (FIXTURE_DIR / "label_border_data.csv").open("w", newline="") as f:
        csv.writer(f).writerows(out)
    print(f"wrote {len(out) - 1} rows")


if __name__ == "__main__":
    main()
