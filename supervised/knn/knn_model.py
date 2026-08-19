"""
K-Nearest Neighbours Classifier — fully parameterised
=======================================================
Exposes every scikit-learn KNeighborsClassifier hyper-parameter so you can
experiment with each one.  Returns a trained model plus a results dict.
"""

import os
from typing import Optional

import time
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    # --- core hyper-parameters ---
    n_neighbors: int = 7,                # number of nearest neighbours to vote; odd avoids ties; lower = more sensitive to noise
    weights: str = "distance",           # "uniform" = all neighbours vote equally; "distance" = closer neighbours count more
    algorithm: str = "auto",             # search algorithm: "ball_tree", "kd_tree" (fast for low dims), "brute" (exact), "auto" (picks best)
    leaf_size: int = 30,                 # leaf size for ball_tree/kd_tree; affects build speed and query speed
    p: int = 2,                          # power parameter for Minkowski distance: 1 = manhattan (city-block), 2 = euclidean
    metric: str = "minkowski",           # distance metric: "minkowski", "euclidean", "manhattan", "chebyshev", etc.
    metric_params: Optional[dict] = None,  # extra keyword args for the metric function (e.g. weights for weighted minkowski)
    # --- behaviour ---
    n_jobs: int = -1,                    # CPU cores to use; -1 = all available cores
):
    """Train a KNN classifier and return (model, results_dict)."""

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    predict_time = time.perf_counter() - start

    results = {
        "model_name": "KNN",
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
    }

    lines = [
        "=" * 50,
        "  KNN  —  Results",
        "=" * 50,
        f"  Accuracy   : {results['accuracy']:.4f}",
        f"  Precision  : {results['precision']:.4f}",
        f"  Recall     : {results['recall']:.4f}",
        f"  F1 Score   : {results['f1']:.4f}",
        f"  ROC AUC    : {results['roc_auc']:.4f}",
        f"  Train time : {train_time:.3f}s",
        f"  Predict    : {predict_time:.4f}s",
        "=" * 50,
        "",
        "Classification Report:",
        results["classification_report"],
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)

    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    return model, results
