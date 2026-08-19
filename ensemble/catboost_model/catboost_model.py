"""
CatBoost — fully parameterised
================================
Yandex's gradient boosting framework. Handles categorical features natively
(no one-hot encoding needed) and uses ordered boosting to reduce overfitting.

Run:
    python catboost_model.py
"""

import os, sys, time
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_catboost(
    X_train, y_train, X_test, y_test,
    iterations: int = 300,              # number of boosting rounds (= n_estimators)
    learning_rate: float = 0.05,        # step size shrinkage
    depth: int = 6,                     # max tree depth; CatBoost uses symmetric (oblivious) trees
    l2_leaf_reg: float = 3.0,           # L2 regularisation on leaf weights
    border_count: int = 254,            # number of splits for numerical features (synonym: max_bin)
    bagging_temperature: float = 1.0,   # controls intensity of Bayesian bootstrap; 0 = no bagging
    subsample: float = 0.8,            # fraction of training data used per round
    colsample_bylevel: float = 1.0,     # fraction of features per split level
    min_data_in_leaf: int = 1,          # min samples in a leaf
    grow_policy: str = "SymmetricTree", # "SymmetricTree" (default), "Depthwise", "Lossguide"
    boosting_type: str = "Ordered",     # "Ordered" (reduces overfitting) or "Plain" (classic)
    auto_class_weights: str = "Balanced",  # handle class imbalance: "Balanced", "SqrtBalanced", None
    random_state: int = 42,
    verbose: int = 0,                   # 0 = silent
):
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth,
        l2_leaf_reg=l2_leaf_reg, border_count=border_count,
        bagging_temperature=bagging_temperature, subsample=subsample,
        colsample_bylevel=colsample_bylevel, min_data_in_leaf=min_data_in_leaf,
        grow_policy=grow_policy, boosting_type=boosting_type,
        auto_class_weights=auto_class_weights,
        random_seed=random_state, verbose=verbose,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test).astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": "CatBoost",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
        "y_pred": y_pred, "y_proba": y_proba,
    }

    lines = [
        "=" * 50, "  CATBOOST  —  Results", "=" * 50,
        f"  Iterations    : {iterations}",
        f"  learning_rate : {learning_rate}",
        f"  depth         : {depth}",
        f"  boosting      : {boosting_type}",
        f"  Accuracy      : {results['accuracy']:.4f}",
        f"  Precision     : {results['precision']:.4f}",
        f"  Recall        : {results['recall']:.4f}",
        f"  F1 Score      : {results['f1']:.4f}",
        f"  ROC AUC       : {results['roc_auc']:.4f}",
        f"  Train time    : {train_time:.4f}s",
        "=" * 50, "", "Classification Report:",
        classification_report(y_test, y_pred),
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)
    return model, results


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from data_generators.classification_data import generate_synthetic_data
    X_tr, X_te, y_tr, y_te, _ = generate_synthetic_data()
    train_catboost(X_tr, y_tr, X_te, y_te)
