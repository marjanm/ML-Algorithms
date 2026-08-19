"""
Support Vector Machine — fully parameterised
===============================================
Finds the hyperplane that maximises the margin between classes.
Can learn non-linear boundaries via the kernel trick.

Run:
    python svm_model.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def train_svm(
    X_train, y_train, X_test, y_test,
    C: float = 1.0,                     # regularisation; larger = narrower margin, less misclassification
    kernel: str = "rbf",                # kernel function: "linear", "poly", "rbf" (gaussian), "sigmoid"
    degree: int = 3,                    # degree for "poly" kernel (ignored by others)
    gamma: str = "scale",               # kernel coefficient for rbf/poly/sigmoid: "scale"=1/(n_features*var), "auto"=1/n_features
    coef0: float = 0.0,                 # independent term in poly/sigmoid kernels
    shrinking: bool = True,             # whether to use shrinking heuristic (speeds up training)
    probability: bool = True,           # enable probability estimates (needed for predict_proba)
    tol: float = 1e-3,                  # stopping tolerance
    cache_size: float = 200,            # kernel cache in MB
    class_weight: str = "balanced",     # handle class imbalance; None = equal weight
    max_iter: int = -1,                 # -1 = no limit
    decision_function_shape: str = "ovr",  # "ovr" (one-vs-rest) or "ovo" (one-vs-one) for multiclass
    break_ties: bool = False,           # break ties according to decision_function values
    random_state: int = 42,
):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = SVC(
        C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
        shrinking=shrinking, probability=probability, tol=tol,
        cache_size=cache_size, class_weight=class_weight, max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        break_ties=break_ties, random_state=random_state,
    )

    start = time.perf_counter()
    model.fit(X_train_s, y_train)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    results = {
        "model_name": "SVM",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
        "y_pred": y_pred, "y_proba": y_proba,
    }

    lines = [
        "=" * 50, "  SUPPORT VECTOR MACHINE  —  Results", "=" * 50,
        f"  Kernel     : {kernel}",
        f"  C          : {C}",
        f"  Gamma      : {gamma}",
        f"  # SVs      : {sum(model.n_support_)}",
        f"  Accuracy   : {results['accuracy']:.4f}",
        f"  Precision  : {results['precision']:.4f}",
        f"  Recall     : {results['recall']:.4f}",
        f"  F1 Score   : {results['f1']:.4f}",
        f"  ROC AUC    : {results['roc_auc']:.4f}",
        f"  Train time : {train_time:.4f}s",
        "=" * 50, "", "Classification Report:",
        classification_report(y_test, y_pred),
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # decision boundary (only works for 2-feature data)
    if X_train.shape[1] == 2:
        h = 0.05
        x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
        y_min, y_max = X_train_s[:, 1].min() - 1, X_train_s[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.figure(figsize=(10, 7))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
        plt.scatter(X_train_s[:, 0], X_train_s[:, 1], c=y_train, cmap="RdBu", edgecolors="k", s=20, alpha=0.6)
        plt.title("SVM Decision Boundary (RBF kernel)")
        plt.xlabel("Feature 0 (scaled)")
        plt.ylabel("Feature 1 (scaled)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "svm_decision_boundary.png"), dpi=150)
        plt.close()
        print(f"  [saved] boundary plot → {PLOT_DIR}")

    return model, results


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from data_generators.classification_data import generate_synthetic_data
    X_tr, X_te, y_tr, y_te, _ = generate_synthetic_data()
    train_svm(X_tr, y_tr, X_te, y_te)
