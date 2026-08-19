"""
Decision Tree — fully parameterised
=====================================
The building block behind Random Forest and all boosting methods.
A single tree partitions feature space using axis-aligned splits.

Run:
    python decision_tree_model.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def train_decision_tree(
    X_train, y_train, X_test, y_test,
    criterion: str = "gini",            # split quality: "gini" (impurity) or "entropy" (information gain)
    splitter: str = "best",             # "best" = best split, "random" = best random split (adds randomness)
    max_depth: int = None,              # max tree depth; None = expand until pure or min_samples_leaf
    min_samples_split: int = 2,         # min samples required to split a node
    min_samples_leaf: int = 1,          # min samples in a leaf node
    min_weight_fraction_leaf: float = 0.0,  # min weighted fraction of total samples in a leaf
    max_features: str = None,           # features to consider per split: None=all, "sqrt", "log2", int, float
    max_leaf_nodes: int = None,         # cap on leaf nodes; None = unlimited
    min_impurity_decrease: float = 0.0, # split only if impurity decreases by at least this
    class_weight: str = "balanced",     # handle class imbalance
    ccp_alpha: float = 0.0,             # complexity-cost pruning parameter; higher = more pruning
    random_state: int = 42,
):
    model = DecisionTreeClassifier(
        criterion=criterion, splitter=splitter, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features, max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease, class_weight=class_weight,
        ccp_alpha=ccp_alpha, random_state=random_state,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": "Decision Tree",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
        "y_pred": y_pred, "y_proba": y_proba,
    }

    lines = [
        "=" * 50, "  DECISION TREE  —  Results", "=" * 50,
        f"  Criterion    : {criterion}",
        f"  Max depth    : {model.get_depth()}",
        f"  Leaf count   : {model.get_n_leaves()}",
        f"  Accuracy     : {results['accuracy']:.4f}",
        f"  Precision    : {results['precision']:.4f}",
        f"  Recall       : {results['recall']:.4f}",
        f"  F1 Score     : {results['f1']:.4f}",
        f"  ROC AUC      : {results['roc_auc']:.4f}",
        f"  Train time   : {train_time:.4f}s",
        "=" * 50, "", "Classification Report:",
        classification_report(y_test, y_pred),
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # visualise the tree (capped at depth 4 for readability)
    fig, ax = plt.subplots(figsize=(24, 10))
    plot_tree(model, max_depth=4, filled=True, rounded=True,
              feature_names=[f"f{i}" for i in range(X_train.shape[1])],
              class_names=["0", "1"], ax=ax, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "decision_tree.png"), dpi=150)
    plt.close()
    print(f"  [saved] tree plot → {PLOT_DIR}")

    return model, results


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from data_generators.classification_data import generate_synthetic_data
    X_tr, X_te, y_tr, y_te, _ = generate_synthetic_data()
    train_decision_tree(X_tr, y_tr, X_te, y_te)
