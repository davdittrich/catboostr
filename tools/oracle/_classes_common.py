"""Shared dataset constants/helpers for tools/oracle/gen_*_classes_fixture.py.

Every `*_classes` fixture generator pins one model per each of the four
estimator classes (CatBoost, CatBoostClassifier, CatBoostRegressor,
CatBoostRanker) against the *same* underlying dataset shape, so that the
only thing varying row-to-row is the method under test. This module is the
single source of truth for that shared dataset -- extracted verbatim from
the generator scripts (no value changes) so the "same dataset across all
`*_classes` fixtures" invariant cannot silently drift in one script without
the others being updated.

`gen_drop_unused_features_classes_fixture.py` uses its own noise-feature
dataset (NUM1_FOR_BINARY/NOISE1/NOISE2/etc., needed to make
DropUnusedFeatures() deterministically prune specific features) and so only
imports the label/group constants below, not NUM1/NUM2/FEATURE_NAMES/x().
"""
N_ROWS = 20
NUM1 = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25,
        -10.0, 11.5, 1.5, -2.5, 3.25, 4.0, -5.75, 6.5, -7.25, 8.0]
NUM2 = [0.3 * i for i in range(N_ROWS)]
FEATURE_NAMES = ["num1", "num2"]

BINARY_LABEL = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
MULTICLASS_LABEL = [i % 3 for i in range(N_ROWS)]
REGRESSION_LABEL = [0.1 * i - 1.0 for i in range(N_ROWS)]
GROUP_ID = [i // 4 for i in range(N_ROWS)]  # 5 groups of 4


def x():
    return [[NUM1[i], NUM2[i]] for i in range(N_ROWS)]
