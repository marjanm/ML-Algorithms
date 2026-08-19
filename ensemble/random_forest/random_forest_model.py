"""
Random Forest Classifier — fully parameterised
================================================
Exposes every scikit-learn RandomForestClassifier hyper-parameter so you can
experiment with each one.  Returns a trained model plus a results dict.
"""

import os
from typing import Optional

import time
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    # --- core hyper-parameters ---
    n_estimators: int = 300,              # number of trees in the forest; more = better but slower
    criterion: str = "gini",              # split quality measure: "gini" (impurity), "entropy" (information gain), "log_loss"
    max_depth: Optional[int] = 15,        # max depth per tree; None = expand until pure leaves; lower = less overfit
    min_samples_split: int = 5,           # min samples needed to split a node; higher = more conservative
    min_samples_leaf: int = 2,            # min samples required in a leaf; prevents tiny noisy leaves
    min_weight_fraction_leaf: float = 0.0,  # min weighted fraction of total samples in a leaf (0.0 = no constraint)
    max_features: str = "sqrt",           # features considered per split: "sqrt", "log2", None (all), int, or float fraction
    max_leaf_nodes: Optional[int] = None, # max leaf nodes per tree; None = unlimited; limits tree complexity
    min_impurity_decrease: float = 0.0,   # split only if impurity drops by at least this much (0.0 = no threshold)
    # --- bootstrap & sampling ---
    bootstrap: bool = True,               # whether to sample rows with replacement; True = bagging (core RF idea)
    oob_score: bool = True,               # use out-of-bag (unsampled) rows as free validation set
    max_samples: Optional[float] = 0.8,   # fraction of rows to sample per tree (only when bootstrap=True)
    # --- behaviour ---
    n_jobs: int = -1,                     # CPU cores to use; -1 = all available cores
    random_state: int = 42,               # seed for reproducibility
    verbose: int = 0,                     # verbosity level; 0 = silent
    warm_start: bool = False,             # if True, reuse previous fit and add more trees incrementally
    class_weight: Optional[str] = "balanced",  # adjust weights inversely proportional to class frequencies; None = equal weights
    ccp_alpha: float = 0.0,              # cost-complexity pruning; higher = more aggressive pruning (0.0 = no pruning)
    monotonic_cst: None = None,           # per-feature monotonic constraints (None = no constraints)
):
    """Train a Random Forest and return (model, results_dict)."""

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        oob_score=oob_score if bootstrap else False,
        max_samples=max_samples if bootstrap else None,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        warm_start=warm_start,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        monotonic_cst=monotonic_cst,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    predict_time = time.perf_counter() - start

    results = {
        "model_name": "Random Forest",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_importances": model.feature_importances_,
    }

    if bootstrap and oob_score:
        results["oob_score"] = model.oob_score_

    lines = [
        "=" * 50,
        "  RANDOM FOREST  —  Results",
        "=" * 50,
        f"  Accuracy   : {results['accuracy']:.4f}",
        f"  Precision  : {results['precision']:.4f}",
        f"  Recall     : {results['recall']:.4f}",
        f"  F1 Score   : {results['f1']:.4f}",
        f"  ROC AUC    : {results['roc_auc']:.4f}",
        f"  Train time : {train_time:.3f}s",
        f"  Predict    : {predict_time:.4f}s",
    ]
    if "oob_score" in results:
        lines.append(f"  OOB Score  : {results['oob_score']:.4f}")
    lines += ["=" * 50, "", "Classification Report:", results["classification_report"]]

    output_text = "\n".join(lines)
    print("\n" + output_text)

    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    return model, results
