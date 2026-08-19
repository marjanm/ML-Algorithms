"""
Naive Bayes — fully parameterised
====================================
Probabilistic classifier based on Bayes' theorem with the "naive"
assumption that features are conditionally independent given the class.
Fast, interpretable, and surprisingly effective for text and small data.

Run:
    python naive_bayes_model.py
"""

import os, sys, time
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_naive_bayes(
    X_train, y_train, X_test, y_test,
    variant: str = "gaussian",          # "gaussian" (continuous features) or "bernoulli" (binary features)
    # GaussianNB params
    var_smoothing: float = 1e-9,        # portion of largest variance added to all variances for stability
    # BernoulliNB params
    alpha: float = 1.0,                 # Laplace / Lidstone smoothing parameter (0 = no smoothing)
    binarize: float = 0.0,             # threshold for binarising features; None = already binary
    fit_prior: bool = True,             # learn class priors from data; False = uniform prior
):
    if variant == "gaussian":
        model = GaussianNB(var_smoothing=var_smoothing)
    else:
        model = BernoulliNB(alpha=alpha, binarize=binarize, fit_prior=fit_prior)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = {
        "model_name": f"Naive Bayes ({variant})",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
        "y_pred": y_pred, "y_proba": y_proba,
    }

    lines = [
        "=" * 50, f"  NAIVE BAYES ({variant.upper()})  —  Results", "=" * 50,
        f"  Variant     : {variant}",
        f"  Accuracy    : {results['accuracy']:.4f}",
        f"  Precision   : {results['precision']:.4f}",
        f"  Recall      : {results['recall']:.4f}",
        f"  F1 Score    : {results['f1']:.4f}",
        f"  ROC AUC     : {results['roc_auc']:.4f}",
        f"  Train time  : {train_time:.6f}s",
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
    train_naive_bayes(X_tr, y_tr, X_te, y_te)
