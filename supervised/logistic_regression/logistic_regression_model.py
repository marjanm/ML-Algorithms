"""
Logistic Regression — fully parameterised
===========================================
The "hello world" of classification. Learns a linear decision boundary
and outputs probabilities via the sigmoid function.

Run:
    python logistic_regression_model.py
"""

import os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_logistic_regression(
    X_train, y_train, X_test, y_test,
    penalty: str = "l2",                # regularisation type: "l1" (sparse weights), "l2" (small weights), "elasticnet", None
    C: float = 1.0,                     # inverse regularisation strength; smaller = stronger regularisation
    solver: str = "lbfgs",              # optimisation algorithm: "lbfgs", "liblinear", "saga", "newton-cg", "sag"
    max_iter: int = 1000,               # max iterations for the solver to converge
    tol: float = 1e-4,                  # convergence tolerance; stop when improvement < tol
    fit_intercept: bool = True,         # whether to add a bias term (almost always True)
    class_weight: str = "balanced",     # "balanced" adjusts for class imbalance; None = equal weight
    multi_class: str = "auto",          # "auto", "ovr" (one-vs-rest), "multinomial"
    l1_ratio: float = None,             # mixing ratio for elasticnet (0=L2, 1=L1); only used with penalty="elasticnet"
    warm_start: bool = False,           # reuse previous fit as starting point
    n_jobs: int = -1,                   # CPU cores; -1 = all
    random_state: int = 42,
):
    model = LogisticRegression(
        penalty=penalty, C=C, solver=solver, max_iter=max_iter, tol=tol,
        fit_intercept=fit_intercept, class_weight=class_weight,
        multi_class=multi_class, l1_ratio=l1_ratio, warm_start=warm_start,
        n_jobs=n_jobs, random_state=random_state,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": "Logistic Regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
        "y_pred": y_pred, "y_proba": y_proba,
    }

    lines = [
        "=" * 50, "  LOGISTIC REGRESSION  —  Results", "=" * 50,
        f"  Penalty    : {penalty}",
        f"  C          : {C}",
        f"  Solver     : {solver}",
        f"  Accuracy   : {results['accuracy']:.4f}",
        f"  Precision  : {results['precision']:.4f}",
        f"  Recall     : {results['recall']:.4f}",
        f"  F1 Score   : {results['f1']:.4f}",
        f"  ROC AUC    : {results['roc_auc']:.4f}",
        f"  Train time : {train_time:.3f}s",
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
    train_logistic_regression(X_tr, y_tr, X_te, y_te)
