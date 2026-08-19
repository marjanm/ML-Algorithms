"""
Feature Engineering — visual demo
====================================
Creating new features from raw data. Often matters more than model choice.

This demo:
  1. Starts with a non-linear dataset that a linear model can't separate
  2. Adds polynomial features (x1², x2², x1*x2) → linear model now works
  3. Adds binning, interaction terms → shows accuracy jumps
  4. Compares: raw features vs engineered features on the same model

Run:
    python feature_engineering_demo.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_decision_boundary(ax, model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=15, alpha=0.7)
    ax.set_title(title, fontsize=10)


def run_feature_engineering_demo():
    X, y = make_circles(n_samples=600, noise=0.1, factor=0.4, random_state=42)
    split = 400
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    strategies = {
        "Raw features (x1, x2)": {
            "pipeline": LogisticRegression(max_iter=1000),
            "transform": lambda X: X,
        },
        "Polynomial deg=2 (x1, x2, x1², x2², x1·x2)": {
            "pipeline": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LogisticRegression(max_iter=1000)),
            "transform": lambda X: X,
        },
        "Polynomial deg=3": {
            "pipeline": make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LogisticRegression(max_iter=1000)),
            "transform": lambda X: X,
        },
        "Manual: add r = sqrt(x1² + x2²)": {
            "pipeline": LogisticRegression(max_iter=1000),
            "transform": lambda X: np.column_stack([X, np.sqrt(X[:, 0]**2 + X[:, 1]**2)]),
        },
    }

    lines = [
        "=" * 65, "  FEATURE ENGINEERING  —  Demo", "=" * 65,
        "  Dataset: concentric circles (not linearly separable)",
        f"  Train: {split}, Test: {len(X) - split}", "",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    results = {}

    for ax, (name, cfg) in zip(axes, strategies.items()):
        Xtr = cfg["transform"](X_train)
        Xte = cfg["transform"](X_test)
        model = cfg["pipeline"]
        model.fit(Xtr, y_train)
        acc = accuracy_score(y_test, model.predict(Xte))
        results[name] = acc
        lines.append(f"  {name}:")
        lines.append(f"    Features: {Xtr.shape[1]},  Test accuracy: {acc:.4f}")
        lines.append("")

        # for plotting, we need the raw 2D boundary
        if isinstance(model, LogisticRegression) and Xtr.shape[1] > 2:
            ax.text(0.5, 0.5, f"Acc = {acc:.3f}\n({Xtr.shape[1]} features)\nCan't plot >2D boundary",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="RdBu", edgecolors="k", s=15, alpha=0.7)
            ax.set_title(name.split("(")[0].strip(), fontsize=9)
        else:
            plot_decision_boundary(ax, model, X_train, y_train, name.split("(")[0].strip())
        ax.set_xlabel(f"Acc = {acc:.3f}")

    lines += [
        "  Takeaway:",
        "    - Raw 2 features: logistic regression draws a straight line → ~50% (fails)",
        "    - Add polynomial features: now the model can learn curves → ~90%+",
        "    - Manual feature (radius): domain knowledge = 1 feature that solves it",
        "    - Feature engineering often matters MORE than choosing a fancier model",
        "=" * 65,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_engineering.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_feature_engineering_demo()
