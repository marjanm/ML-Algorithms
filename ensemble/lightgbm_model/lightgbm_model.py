"""
LightGBM — fully parameterised
================================
Microsoft's gradient boosting framework. Faster than XGBoost on large data
because it uses histogram-based splitting and leaf-wise growth.

Run:
    python lightgbm_model.py
"""

import os, sys, time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_lightgbm(
    X_train, y_train, X_test, y_test,
    n_estimators: int = 300,            # number of boosting rounds
    learning_rate: float = 0.05,        # step size shrinkage; smaller = more rounds needed but better generalisation
    max_depth: int = -1,                # max tree depth; -1 = no limit (controlled by num_leaves instead)
    num_leaves: int = 31,               # max number of leaves per tree; main control for model complexity
    min_child_samples: int = 20,        # min samples in a leaf; prevents overfitting on noisy data
    min_child_weight: float = 1e-3,     # min sum of instance weight in a leaf
    subsample: float = 0.8,            # fraction of training data used per round (row subsampling)
    subsample_freq: int = 1,            # frequency of subsampling; 0 = disabled
    colsample_bytree: float = 0.8,      # fraction of features used per tree
    reg_alpha: float = 0.0,             # L1 regularisation on leaf weights
    reg_lambda: float = 1.0,            # L2 regularisation on leaf weights
    min_split_gain: float = 0.0,        # min gain to make a split
    max_bin: int = 255,                 # max number of bins for histogram; more = finer splits
    boosting_type: str = "gbdt",        # "gbdt" (traditional), "dart" (dropout), "goss" (one-side sampling)
    objective: str = "binary",          # loss function
    class_weight: str = "balanced",     # handle class imbalance
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = -1,                  # -1 = silent
):
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, num_leaves=num_leaves,
        min_child_samples=min_child_samples, min_child_weight=min_child_weight,
        subsample=subsample, subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        min_split_gain=min_split_gain, max_bin=max_bin,
        boosting_type=boosting_type, objective=objective,
        class_weight=class_weight, n_jobs=n_jobs,
        random_state=random_state, verbose=verbose,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": "LightGBM",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
        "y_pred": y_pred, "y_proba": y_proba,
    }

    lines = [
        "=" * 50, "  LIGHTGBM  —  Results", "=" * 50,
        f"  n_estimators  : {n_estimators}",
        f"  learning_rate : {learning_rate}",
        f"  num_leaves    : {num_leaves}",
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
    train_lightgbm(X_tr, y_tr, X_te, y_te)
