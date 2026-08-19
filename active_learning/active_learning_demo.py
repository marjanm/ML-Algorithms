"""
Active Learning Demo
====================
Shows how to intelligently choose which data points to label, reducing the
labeling budget by 2-3x while achieving the same accuracy.

Strategies compared:
  1. Random sampling   — baseline, pick randomly
  2. Uncertainty        — pick samples the model is least confident about
  3. Margin sampling    — pick samples where top-2 class probabilities are closest
  4. Entropy sampling   — pick samples with highest prediction entropy

The loop:
  1. Start with a small labeled pool
  2. Train model on labeled pool
  3. Predict on unlabeled pool
  4. Select most informative samples using a query strategy
  5. "Label" them (reveal the true label) and add to labeled pool
  6. Repeat → plot accuracy vs number of labeled samples
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy as sp_entropy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def uncertainty_sampling(probs):
    """Select samples with lowest max probability (least confident)."""
    return np.argsort(probs.max(axis=1))


def margin_sampling(probs):
    """Select samples where top-2 probabilities are closest."""
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.argsort(margins)


def entropy_sampling(probs):
    """Select samples with highest prediction entropy."""
    ent = sp_entropy(probs.T)
    return np.argsort(-ent)  # descending entropy


def random_sampling(probs):
    """Random baseline."""
    return np.random.permutation(len(probs))


def active_learning_loop(X_train, y_train, X_test, y_test, query_fn, 
                         n_initial=20, n_query=20, n_rounds=25, seed=42):
    """Run the active learning loop and return accuracy curve."""
    np.random.seed(seed)

    pool_idx = np.arange(len(X_train))
    np.random.shuffle(pool_idx)
    labeled_idx = list(pool_idx[:n_initial])
    unlabeled_idx = list(pool_idx[n_initial:])

    accuracies = []
    n_labeled_list = []

    for rnd in range(n_rounds + 1):
        model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
        model.fit(X_train[labeled_idx], y_train[labeled_idx])
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies.append(acc)
        n_labeled_list.append(len(labeled_idx))

        if rnd == n_rounds or len(unlabeled_idx) < n_query:
            break

        # Query strategy
        probs = model.predict_proba(X_train[unlabeled_idx])
        ranked = query_fn(probs)
        selected = ranked[:n_query]

        for idx in sorted(selected, reverse=True):
            labeled_idx.append(unlabeled_idx.pop(idx))

    return n_labeled_list, accuracies


def run_active_learning_demo():
    log("ACTIVE LEARNING DEMO")
    log("=" * 60)

    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    log(f"\nDataset: {len(X)} digit images (10 classes)")
    log(f"Train: {len(X_train)}, Test: {len(X_test)}")
    log(f"Starting with 20 labeled, querying 20 per round, 25 rounds")

    strategies = [
        ("Random", random_sampling, "#95a5a6"),
        ("Uncertainty", uncertainty_sampling, "#e74c3c"),
        ("Margin", margin_sampling, "#3498db"),
        ("Entropy", entropy_sampling, "#2ecc71"),
    ]

    results = {}
    for name, fn, color in strategies:
        n_list, accs = active_learning_loop(
            X_train, y_train, X_test, y_test, fn, seed=42
        )
        results[name] = (n_list, accs, color)
        log(f"\n{name} sampling:")
        log(f"  After  20 labels: {accs[0]:.3f}")
        log(f"  After 120 labels: {accs[min(5, len(accs)-1)]:.3f}")
        log(f"  After {n_list[-1]} labels: {accs[-1]:.3f}")

    # Upper bound
    full_model = LogisticRegression(max_iter=1000, random_state=42)
    full_model.fit(X_train, y_train)
    full_acc = accuracy_score(y_test, full_model.predict(X_test))
    log(f"\nUpper bound (all {len(X_train)} labels): {full_acc:.3f}")

    # Key insight: how many labels does each strategy need to reach 90%?
    log(f"\nLabels needed to reach 90% accuracy:")
    for name, (n_list, accs, _) in results.items():
        reached = [n for n, a in zip(n_list, accs) if a >= 0.90]
        if reached:
            log(f"  {name:<15} {reached[0]:>5} labels")
        else:
            log(f"  {name:<15} did not reach 90%")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning curves
    ax = axes[0]
    for name, (n_list, accs, color) in results.items():
        ax.plot(n_list, accs, "o-", label=name, color=color, markersize=4, linewidth=2)
    ax.axhline(y=full_acc, color="black", linestyle="--", alpha=0.5,
               label=f"Full supervision ({full_acc:.3f})")
    ax.axhline(y=0.90, color="gray", linestyle=":", alpha=0.5, label="90% target")
    ax.set_xlabel("Number of Labeled Samples")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Active Learning: Accuracy vs Labels Used")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.02)

    # Savings: how many fewer labels needed?
    ax = axes[1]
    target_acc = 0.90
    labels_needed = {}
    for name, (n_list, accs, color) in results.items():
        reached = [n for n, a in zip(n_list, accs) if a >= target_acc]
        labels_needed[name] = reached[0] if reached else n_list[-1]

    names = list(labels_needed.keys())
    counts = list(labels_needed.values())
    colors = [results[n][2] for n in names]
    bars = ax.bar(names, counts, color=colors, width=0.5)
    ax.set_ylabel(f"Labels to reach {target_acc:.0%} accuracy")
    ax.set_title("Labeling Budget Comparison")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c + 5, str(c), ha="center", fontweight="bold")

    if labels_needed.get("Random", 0) > 0 and labels_needed.get("Entropy", 0) > 0:
        savings = 1 - labels_needed["Entropy"] / labels_needed["Random"]
        ax.set_xlabel(f"Entropy saves ~{savings:.0%} of labeling budget vs random")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "active_learning.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/active_learning.png")

    log(f"\n{'=' * 60}")
    log("KEY TAKEAWAYS")
    log(f"{'=' * 60}")
    log("""
Active learning reduces labeling cost by querying the MOST INFORMATIVE samples.

Query strategies (from simplest to most advanced):
  1. Uncertainty  — "I'm least sure about this one"
  2. Margin       — "My top two guesses are almost tied"
  3. Entropy      — "I'm confused across many classes"
  4. QBC          — committee of models disagrees (not shown)
  5. Expected gradient — which sample would change the model most

When to use active learning:
  • Labeling is expensive (medical images, legal docs)
  • You have lots of unlabeled data but limited annotation budget
  • The decision boundary is complex — random labeling wastes effort
    on "easy" samples far from the boundary
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_active_learning_demo()
