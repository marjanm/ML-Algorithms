"""
Cross-Validation — visual demo
================================
Split data k ways, train on k-1, test on 1. Rotate. More reliable than
a single train/test split.

This demo:
  1. Runs a single 80/20 split 20 times (different random seeds) to show
     how much the score varies.
  2. Runs 5-fold and 10-fold CV to show how the variance shrinks.
  3. Compares different CV strategies (KFold, Stratified, Repeated).

Run:
    python cross_validation_demo.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold, RepeatedStratifiedKFold,
    train_test_split, LeaveOneOut,
)
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_generators.classification_data import generate_synthetic_data

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_cross_validation_demo():
    X_train, X_test, y_train, y_test, df = generate_synthetic_data(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
    )
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # --- 1. single split variability ---
    single_scores = []
    for seed in range(30):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        m = RandomForestClassifier(n_estimators=100, random_state=42)
        m.fit(Xtr, ytr)
        single_scores.append(accuracy_score(yte, m.predict(Xte)))

    # --- 2. k-fold variants ---
    cv_strategies = {
        "3-Fold": KFold(n_splits=3, shuffle=True, random_state=42),
        "5-Fold": KFold(n_splits=5, shuffle=True, random_state=42),
        "10-Fold": KFold(n_splits=10, shuffle=True, random_state=42),
        "5-Fold Stratified": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "5×2 Repeated Stratified": RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
    }

    cv_results = {}
    for name, cv in cv_strategies.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_results[name] = scores

    lines = [
        "=" * 65, "  CROSS-VALIDATION  —  Demo", "=" * 65, "",
        "  1. Single random 80/20 split (30 different seeds):",
        f"     Mean accuracy : {np.mean(single_scores):.4f}",
        f"     Std deviation : {np.std(single_scores):.4f}",
        f"     Range         : [{min(single_scores):.4f}, {max(single_scores):.4f}]",
        "",
        "  2. Cross-validation strategies:",
    ]
    for name, scores in cv_results.items():
        lines.append(f"     {name:30s} → {np.mean(scores):.4f} ± {np.std(scores):.4f}  (folds: {len(scores)})")
    lines += [
        "", "  Takeaway:",
        "    - Single split: high variance (score changes a lot with seed)",
        "    - More folds  : lower variance, more reliable estimate",
        "    - Stratified   : preserves class proportions in each fold",
        "    - Repeated     : even more stable (multiple shuffles)",
        "=" * 65,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # single split distribution
    axes[0].hist(single_scores, bins=12, edgecolor="black", alpha=0.7, color="tab:orange")
    axes[0].axvline(np.mean(single_scores), color="red", ls="--", label=f"Mean = {np.mean(single_scores):.4f}")
    axes[0].set_xlabel("Accuracy")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Single 80/20 split — 30 random seeds\n(see how much the score varies!)")
    axes[0].legend()

    # box plot of CV strategies
    labels = list(cv_results.keys())
    data = [cv_results[k] for k in labels]
    bp = axes[1].boxplot(data, labels=[l.replace(" ", "\n") for l in labels],
                         patch_artist=True, widths=0.5)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("CV strategies compared\n(more folds & repeats → tighter boxes)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cross_validation.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_cross_validation_demo()
