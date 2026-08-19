"""
Bias-Variance Tradeoff — visual demo
======================================
Simple model = high bias (underfits).  Complex model = high variance (overfits).
The sweet spot is in between.

This demo increases Decision Tree depth from 1 to 25 and plots training
accuracy vs test accuracy.  You'll see the classic divergence:
  - Training accuracy keeps climbing (eventually 100 %)
  - Test accuracy rises, peaks, then drops (overfitting)

Run:
    python bias_variance_demo.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_generators.classification_data import generate_synthetic_data

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_bias_variance_demo():
    X_train, X_test, y_train, y_test, _ = generate_synthetic_data(
        n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
    )

    depths = range(1, 26)
    train_accs, test_accs = [], []

    for d in depths:
        model = DecisionTreeClassifier(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, model.predict(X_train)))
        test_accs.append(accuracy_score(y_test, model.predict(X_test)))

    best_depth = depths[np.argmax(test_accs)]
    best_test = max(test_accs)

    lines = [
        "=" * 60, "  BIAS-VARIANCE TRADEOFF  —  Decision Tree depth sweep", "=" * 60,
        "",
        "  depth | train_acc | test_acc",
        "  ------|-----------|----------",
    ]
    for d, tr, te in zip(depths, train_accs, test_accs):
        marker = " <-- best" if d == best_depth else ""
        lines.append(f"   {d:4d} |   {tr:.4f}  |  {te:.4f}{marker}")
    lines += [
        "", f"  Best depth = {best_depth}  (test acc = {best_test:.4f})",
        "",
        "  Takeaway:",
        "    depth 1-3  → high bias (underfitting): train & test both low",
        "    depth 5-10 → sweet spot: test accuracy peaks",
        "    depth 15+  → high variance (overfitting): train→100%, test drops",
        "=" * 60,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(depths, train_accs, "o-", label="Training accuracy", color="tab:blue")
    axes[0].plot(depths, test_accs, "s-", label="Test accuracy", color="tab:orange")
    axes[0].axvline(best_depth, ls="--", color="grey", label=f"Best depth = {best_depth}")
    axes[0].fill_between([1, 4], 0.5, 1.02, alpha=0.08, color="blue", label="High bias zone")
    axes[0].fill_between([15, 25], 0.5, 1.02, alpha=0.08, color="red", label="High variance zone")
    axes[0].set_xlabel("Tree depth (model complexity →)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Bias-Variance Tradeoff")
    axes[0].set_ylim(0.5, 1.02)
    axes[0].legend(fontsize=8)

    gap = np.array(train_accs) - np.array(test_accs)
    axes[1].bar(depths, gap, color="tab:red", alpha=0.6)
    axes[1].set_xlabel("Tree depth")
    axes[1].set_ylabel("Train acc − Test acc (gap)")
    axes[1].set_title("Generalisation gap (overfitting indicator)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bias_variance.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_bias_variance_demo()
