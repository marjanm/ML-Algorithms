"""
kNN — FROM SCRATCH
====================
No sklearn. Hand-coded Euclidean distance, neighbour selection, and
majority voting.

The algorithm:
  1. For a new point, compute distance to every training point
  2. Pick the k closest neighbours
  3. Majority vote → predicted class

Run:
    python knn_scratch.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_generators.classification_data import generate_synthetic_data

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def euclidean_distance(a, b):
    """||a - b||₂ = sqrt(Σ (a_i - b_i)²)"""
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def manhattan_distance(a, b):
    """||a - b||₁ = Σ |a_i - b_i|"""
    return np.sum(np.abs(a - b), axis=1)


def predict_one(X_train, y_train, x_query, k=5, distance_fn=euclidean_distance):
    """Predict class for a single query point."""
    distances = distance_fn(X_train, x_query)
    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_idx]
    vote = Counter(nearest_labels).most_common(1)[0][0]
    return vote, distances[nearest_idx]


def predict(X_train, y_train, X_test, k=5, distance_fn=euclidean_distance):
    """Predict classes for all test points."""
    return np.array([predict_one(X_train, y_train, x, k, distance_fn)[0] for x in X_test])


def run_knn_demo():
    X_train, X_test, y_train, y_test, _ = generate_synthetic_data(
        n_samples=500, n_features=2, n_informative=2, n_redundant=0,
    )

    lines = [
        "=" * 60, "  kNN — FROM SCRATCH", "=" * 60,
        f"  Train size : {len(y_train)}",
        f"  Test size  : {len(y_test)}", "",
    ]

    # show a single prediction step-by-step
    query = X_test[0]
    pred, dists = predict_one(X_train, y_train, query, k=5)
    lines += [
        f"  Example prediction for X_test[0] = [{query[0]:.4f}, {query[1]:.4f}]:",
        f"    5 nearest distances : {[f'{d:.4f}' for d in dists]}",
        f"    Predicted class     : {pred}",
        f"    Actual class        : {y_test[0]}",
        "",
    ]

    # test different k values
    lines.append("  Accuracy for different k values:")
    k_values = [1, 3, 5, 7, 11, 15, 21, 31]
    k_accs = []
    for k in k_values:
        y_pred = predict(X_train, y_train, X_test, k=k)
        acc = np.mean(y_pred == y_test)
        k_accs.append(acc)
        lines.append(f"    k={k:3d} → accuracy = {acc:.4f}")

    # compare distance metrics
    lines += ["", "  Distance metric comparison (k=5):"]
    for name, fn in [("Euclidean", euclidean_distance), ("Manhattan", manhattan_distance)]:
        y_pred = predict(X_train, y_train, X_test, k=5, distance_fn=fn)
        acc = np.mean(y_pred == y_test)
        lines.append(f"    {name:12s} → accuracy = {acc:.4f}")

    lines += [
        "", "  Key formulas:",
        "    Euclidean : ||a-b||₂ = sqrt(Σ (a_i - b_i)²)",
        "    Manhattan : ||a-b||₁ = Σ |a_i - b_i|",
        "    Prediction: majority vote of k nearest neighbours",
        "    No training step — the entire training set IS the model",
        "=" * 60,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # decision boundary for k=5
    k = 5
    h = 0.1
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(X_train, y_train, grid, k=k).reshape(xx.shape)
    axes[0].contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="RdBu", edgecolors="k", s=15, alpha=0.5)
    best_acc = k_accs[k_values.index(5)]
    axes[0].set_title(f"kNN decision boundary (k=5, acc={best_acc:.3f})")
    axes[0].set_xlabel("Feature 0")
    axes[0].set_ylabel("Feature 1")

    # k vs accuracy
    axes[1].plot(k_values, k_accs, "o-", linewidth=2, markersize=8)
    axes[1].set_xlabel("k (number of neighbours)")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Effect of k on accuracy")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "knn_scratch.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_knn_demo()
