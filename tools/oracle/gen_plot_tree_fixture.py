#!/usr/bin/env python3
"""Generate the plot_tree oracle fixture (catboost-8z4.52 / P4.3): deterministic
train data + the pinned Python catboost==1.2.10 CatBoostClassifier.plot_tree()'s
observable structure, for the structural differential test against
catboost.plot_tree() (R/catboost.R).

plot_tree() returns a graphviz.Digraph built from the model's internal
per-tree splits/leaf_values (catboost/python-package/catboost/core.py
_plot_oblivious_tree / _get_tree_splits / _get_tree_leaf_values). Per spec
Sec 4.3's Structural row, the differential-test target is not the rendered
image but a canonical node/edge structure: for each graphviz.Digraph.body
entry (one per graph.node()/graph.edge() call -- a single source of truth,
not a reimplementation of the tree-print algorithm) this script regex-parses
the node id/label/color/shape or edge from/to/label fields into plain JSON.

Two fixture cases:
  * "float_only": two float features, no categorical -- exercises the
    pool=NULL path (feature labels fall back to flat_feature_index) and the
    pool!=NULL path (feature labels use feature names) on the SAME model.
  * "with_cat": one float + one one-hot categorical feature -- exercises the
    categorical-split label path (requires pool, matching the vendor
    CB_ENSURE "training dataset is required if categorical features are
    present" behavior).

Run via: uv run --frozen --project tools/oracle python3 tools/oracle/gen_plot_tree_fixture.py
"""
import json
import os
import re

from catboost import CatBoostClassifier, Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "oracle")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "plot_tree.json")

NODE_RE = re.compile(r'\t(\S+) \[label="(.*)" color=(\w+) shape=(\w+)\]\n', re.DOTALL)
EDGE_RE = re.compile(r'\t(\S+) -> (\S+) \[label=(\w+)\]\n')


def parse_digraph(graph):
    nodes = []
    edges = []
    for item in graph.body:
        m = NODE_RE.match(item)
        if m:
            nodes.append({
                "id": m.group(1),
                "label": m.group(2),
                "color": m.group(3),
                "shape": m.group(4),
            })
            continue
        m = EDGE_RE.match(item)
        if m:
            edges.append({"from": m.group(1), "to": m.group(2), "label": m.group(3)})
            continue
        raise ValueError("Unparsed graphviz body item: {!r}".format(item))
    return {"nodes": nodes, "edges": edges}


def main():
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    fixture = {}

    # --- float_only case ---
    num1 = [0.123456789, 1.987654321, 2.5, 3.333333, 4.1, 5.777777, 6.2, 7.9,
            -1.0, -2.5, 8.4, 9.9, 0.0001234, 3.14159, -4.2, 6.66]
    num2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    label = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    feature_names = ["num1", "num2"]

    X = [[num1[i], num2[i]] for i in range(len(label))]
    pool = Pool(X, label, feature_names=feature_names)
    model = CatBoostClassifier(
        iterations=3, depth=3, loss_function="Logloss",
        random_seed=1, thread_count=1, verbose=False,
        train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_plot_tree_float"),
    )
    model.fit(pool)

    tree_idx = 1
    graph_with_pool = model.plot_tree(tree_idx, pool)
    graph_without_pool = model.plot_tree(tree_idx)

    fixture["float_only"] = {
        "inputs": {
            "num1": num1, "num2": num2, "label": label,
            "feature_names": feature_names, "tree_idx": tree_idx,
        },
        "expected": {
            "with_pool": parse_digraph(graph_with_pool),
            "without_pool": parse_digraph(graph_without_pool),
        },
    }

    # --- with_cat case (one-hot categorical; requires pool) ---
    num1c = [0.5, -1.5, 2.25, 3.0, -4.75, 5.5, -6.25, 7.0, -8.5, 9.25, -10.0, 11.5]
    cat1 = ["a", "b", "a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]
    labelc = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    feature_names_c = ["num1", "cat1"]

    Xc = [[num1c[i], cat1[i]] for i in range(len(labelc))]
    poolc = Pool(Xc, labelc, cat_features=[1], feature_names=feature_names_c)
    modelc = CatBoostClassifier(
        iterations=3, depth=2, loss_function="Logloss",
        random_seed=2, thread_count=1, verbose=False, one_hot_max_size=10,
        train_dir=os.path.join(SCRIPT_DIR, ".catboost_train_plot_tree_cat"),
    )
    modelc.fit(poolc)

    tree_idx_c = 0
    graph_c = modelc.plot_tree(tree_idx_c, poolc)

    fixture["with_cat"] = {
        "inputs": {
            "num1": num1c, "cat1": cat1, "label": labelc,
            "feature_names": feature_names_c, "tree_idx": tree_idx_c,
            "one_hot_max_size": 10,
        },
        "expected": {
            "with_pool": parse_digraph(graph_c),
        },
    }

    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=2)
    print(FIXTURE_PATH)


if __name__ == "__main__":
    main()
