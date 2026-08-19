"""
Decision Tree Split — FROM SCRATCH
=====================================
No sklearn. Hand-coded Gini impurity, information gain, and recursive
tree building with a depth limit.

Shows the core algorithm:
  For each feature, for each threshold:
    1. Split data into left / right
    2. Compute weighted Gini (or info gain) of the split
    3. Pick the split with lowest impurity
  Recurse on each side.

Run:
    python decision_tree_split_scratch.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_generators.classification_data import generate_synthetic_data

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def gini_impurity(y):
    """Gini = 1 - Σ p_k² .  Pure node = 0, worst = 0.5 (binary)."""
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


def entropy(y):
    """Entropy = -Σ p_k * log2(p_k).  Pure node = 0."""
    if len(y) == 0:
        return 0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def find_best_split(X, y, criterion="gini"):
    """Try every feature × threshold, return the one with lowest impurity."""
    impurity_fn = gini_impurity if criterion == "gini" else entropy
    n_samples, n_features = X.shape
    best_score = float("inf")
    best_feature, best_threshold = None, None

    parent_impurity = impurity_fn(y)

    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        for thresh in thresholds:
            left_mask = X[:, feature_idx] <= thresh
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            left_imp = impurity_fn(y[left_mask])
            right_imp = impurity_fn(y[right_mask])
            n_left, n_right = left_mask.sum(), right_mask.sum()
            weighted_imp = (n_left * left_imp + n_right * right_imp) / n_samples

            if weighted_imp < best_score:
                best_score = weighted_imp
                best_feature = feature_idx
                best_threshold = thresh

    info_gain = parent_impurity - best_score
    return best_feature, best_threshold, best_score, info_gain


class ScratchNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


def build_tree(X, y, depth=0, max_depth=5, min_samples=2):
    """Recursively build a decision tree."""
    if depth >= max_depth or len(y) < min_samples or len(np.unique(y)) == 1:
        classes, counts = np.unique(y, return_counts=True)
        return ScratchNode(prediction=classes[np.argmax(counts)])

    feat, thresh, _, _ = find_best_split(X, y)
    if feat is None:
        classes, counts = np.unique(y, return_counts=True)
        return ScratchNode(prediction=classes[np.argmax(counts)])

    left_mask = X[:, feat] <= thresh
    left = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth, min_samples)
    right = build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth, min_samples)
    return ScratchNode(feature=feat, threshold=thresh, left=left, right=right)


def predict_one(node, x):
    if node.prediction is not None:
        return node.prediction
    if x[node.feature] <= node.threshold:
        return predict_one(node.left, x)
    return predict_one(node.right, x)


def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])


def print_tree(node, depth=0, prefix="ROOT"):
    """Return a string representation of the tree."""
    indent = "  " * depth
    if node.prediction is not None:
        return [f"{indent}{prefix} → predict class {node.prediction}"]
    lines = [f"{indent}{prefix} [feature_{node.feature} <= {node.threshold:.4f}]"]
    lines += print_tree(node.left, depth + 1, "L")
    lines += print_tree(node.right, depth + 1, "R")
    return lines


def run_decision_tree_demo():
    X_train, X_test, y_train, y_test, _ = generate_synthetic_data(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
    )

    lines = [
        "=" * 60, "  DECISION TREE SPLIT — FROM SCRATCH", "=" * 60, "",
    ]

    # show a single split step
    feat, thresh, score, gain = find_best_split(X_train, y_train)
    lines += [
        "  First split (root node):",
        f"    Best feature   : feature_{feat}",
        f"    Best threshold : {thresh:.4f}",
        f"    Weighted Gini  : {score:.4f}",
        f"    Information gain: {gain:.4f}",
        f"    Parent Gini    : {gini_impurity(y_train):.4f}",
        "",
    ]

    # build full tree
    tree = build_tree(X_train, y_train, max_depth=5)
    y_pred = predict(tree, X_test)
    accuracy = np.mean(y_pred == y_test)

    lines.append("  Full tree (max_depth=5):")
    lines += ["    " + l for l in print_tree(tree)]
    lines += [
        "", f"  Test accuracy: {accuracy:.4f}",
        "",
        "  Key formulas:",
        "    Gini(node)    = 1 - Σ p_k²",
        "    Entropy(node) = -Σ p_k · log2(p_k)",
        "    Info gain     = parent_impurity - weighted_children_impurity",
        "    Split rule    : pick feature + threshold that maximises info gain",
        "=" * 60,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- decision boundary plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    h = 0.05
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(tree, grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdBu", edgecolors="k", s=15, alpha=0.7)
    ax.set_title(f"From-scratch Decision Tree (depth=5, acc={accuracy:.3f})")
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "decision_tree_scratch.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_decision_tree_demo()
