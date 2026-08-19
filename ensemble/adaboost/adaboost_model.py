"""
AdaBoost — fully parameterised
================================
The original boosting algorithm. Sequentially trains weak learners
(stumps by default), giving more weight to misclassified samples each round.

Run:
    python adaboost_model.py
"""

import os, sys, time
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_adaboost(
    X_train, y_train, X_test, y_test,
    n_estimators: int = 200,            # number of weak learners to train sequentially
    learning_rate: float = 0.1,         # shrinks each learner's contribution; lower = more estimators needed
    algorithm: str = "SAMME",           # "SAMME" (discrete) or "SAMME.R" (real, uses probability estimates — generally better)
    base_max_depth: int = 1,            # depth of each decision stump; 1 = stump (single split)
    random_state: int = 42,
):
    base_estimator = DecisionTreeClassifier(max_depth=base_max_depth, random_state=random_state)

    model = AdaBoostClassifier(
        estimator=base_estimator,       # the weak learner template
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm,
        random_state=random_state,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": "AdaBoost",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
        "y_pred": y_pred, "y_proba": y_proba,
    }

    lines = [
        "=" * 50, "  ADABOOST  —  Results", "=" * 50,
        f"  n_estimators  : {n_estimators}",
        f"  learning_rate : {learning_rate}",
        f"  algorithm     : {algorithm}",
        f"  base depth    : {base_max_depth}",
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
    train_adaboost(X_tr, y_tr, X_te, y_te)
