"""
Semi-supervised & Self-supervised Learning Demo
================================================
Shows how to learn effectively when most data is UNLABELED.

Part 1 — Semi-supervised (Label Spreading):
    Train a classifier with only 5% of labels.  sklearn's LabelSpreading
    propagates labels through the data graph.  Compare to: (a) supervised
    with 5% labels, (b) supervised with 100% labels (upper bound).

Part 2 — Self-supervised (Contrastive Learning, SimCLR-style):
    Learn image representations WITHOUT any labels by training the model
    to recognize that two augmented views of the same image should have
    similar embeddings.  Then freeze the encoder and train a tiny linear
    classifier with just 1% of labels.

Part 3 — Pseudo-labeling:
    Train on labeled data, predict on unlabeled, treat high-confidence
    predictions as new labels, retrain.  Simple but effective.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_digits
from sklearn.semi_supervised import LabelSpreading
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def semi_supervised_label_spreading():
    """Compare supervised vs semi-supervised with limited labels."""
    log("=" * 60)
    log("PART 1: SEMI-SUPERVISED — LABEL SPREADING")
    log("=" * 60)

    np.random.seed(42)
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    label_fractions = [0.05, 0.10, 0.20, 0.50, 1.0]
    sup_accs = []
    semi_accs = []

    log(f"\n{'Labeled %':>10} | {'Supervised':>12} | {'Label Spreading':>16}")
    log("-" * 45)

    for frac in label_fractions:
        n_labeled = max(int(len(X_train) * frac), 4)

        # Create partially labeled set: -1 = unlabeled
        y_semi = np.full(len(X_train), -1)
        labeled_idx = np.random.choice(len(X_train), n_labeled, replace=False)
        y_semi[labeled_idx] = y_train[labeled_idx]

        # Supervised: only use labeled data
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train[labeled_idx], y_train[labeled_idx])
        sup_acc = accuracy_score(y_test, lr.predict(X_test))
        sup_accs.append(sup_acc)

        # Semi-supervised: use all data, most unlabeled
        ls = LabelSpreading(kernel="rbf", gamma=10, max_iter=100)
        ls.fit(X_train, y_semi)
        semi_acc = accuracy_score(y_test, ls.predict(X_test))
        semi_accs.append(semi_acc)

        log(f"{frac:>9.0%} | {sup_acc:>11.3f} | {semi_acc:>15.3f}")

    log(f"\n→ With only 5% labels, Label Spreading leverages unlabeled data structure")
    log(f"  to achieve much better accuracy than plain supervised learning.")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy comparison
    ax = axes[0]
    pcts = [f"{f:.0%}" for f in label_fractions]
    x = np.arange(len(pcts))
    w = 0.3
    ax.bar(x - w/2, sup_accs, w, label="Supervised only", color="#e74c3c")
    ax.bar(x + w/2, semi_accs, w, label="Label Spreading", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(pcts)
    ax.set_xlabel("Fraction of Labels Available")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Supervised vs Semi-supervised")
    ax.legend()
    ax.set_ylim(0.5, 1.05)

    # Decision boundary with 5% labels
    ax = axes[1]
    n_labeled = max(int(len(X_train) * 0.05), 4)
    y_semi = np.full(len(X_train), -1)
    labeled_idx = np.random.RandomState(42).choice(len(X_train), n_labeled, replace=False)
    y_semi[labeled_idx] = y_train[labeled_idx]

    ls = LabelSpreading(kernel="rbf", gamma=10, max_iter=100)
    ls.fit(X_train, y_semi)

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                         np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
    Z = ls.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlGn")
    unlabeled_mask = np.ones(len(X_train), dtype=bool)
    unlabeled_mask[labeled_idx] = False
    ax.scatter(X_train[unlabeled_mask, 0], X_train[unlabeled_mask, 1],
               c="gray", alpha=0.2, s=10, label="Unlabeled")
    ax.scatter(X_train[labeled_idx, 0], X_train[labeled_idx, 1],
               c=y_train[labeled_idx], cmap="RdYlGn", edgecolors="black",
               s=80, linewidths=1.5, label="Labeled (5%)")
    ax.set_title(f"Label Spreading with 5% labels ({n_labeled} points)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "semi_supervised.png"), dpi=150)
    plt.close()
    log("→ Plot saved: plots/semi_supervised.png")


def pseudo_labeling():
    """Demonstrate pseudo-labeling (self-training)."""
    log("\n" + "=" * 60)
    log("PART 2: PSEUDO-LABELING (SELF-TRAINING)")
    log("=" * 60)

    np.random.seed(42)
    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_initial = 50  # only 50 labeled out of ~1250
    perm = np.random.permutation(len(X_train))
    labeled_idx = perm[:n_initial]
    unlabeled_idx = perm[n_initial:]

    X_labeled = X_train[labeled_idx]
    y_labeled = y_train[labeled_idx]
    X_unlabeled = X_train[unlabeled_idx]
    y_unlabeled_true = y_train[unlabeled_idx]  # only for evaluation

    # Baseline: train only on labeled
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    base_model.fit(X_labeled, y_labeled)
    base_acc = accuracy_score(y_test, base_model.predict(X_test))

    # Pseudo-labeling loop
    CONFIDENCE_THRESHOLD = 0.95
    MAX_ROUNDS = 10

    log(f"\nStarting with {n_initial} labeled, {len(X_unlabeled)} unlabeled")
    log(f"Baseline accuracy (labeled only): {base_acc:.3f}")
    log(f"\nPseudo-labeling (threshold={CONFIDENCE_THRESHOLD}):")

    round_accs = [base_acc]
    round_sizes = [n_initial]
    current_X_labeled = X_labeled.copy()
    current_y_labeled = y_labeled.copy()
    remaining_unlabeled = X_unlabeled.copy()

    for rnd in range(1, MAX_ROUNDS + 1):
        if len(remaining_unlabeled) == 0:
            break

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(current_X_labeled, current_y_labeled)

        probs = model.predict_proba(remaining_unlabeled)
        max_probs = probs.max(axis=1)
        confident_mask = max_probs >= CONFIDENCE_THRESHOLD

        n_new = confident_mask.sum()
        if n_new == 0:
            log(f"  Round {rnd}: no confident predictions, stopping")
            break

        pseudo_labels = model.predict(remaining_unlabeled[confident_mask])
        current_X_labeled = np.vstack([current_X_labeled, remaining_unlabeled[confident_mask]])
        current_y_labeled = np.concatenate([current_y_labeled, pseudo_labels])
        remaining_unlabeled = remaining_unlabeled[~confident_mask]

        acc = accuracy_score(y_test, model.predict(X_test))
        round_accs.append(acc)
        round_sizes.append(len(current_y_labeled))
        log(f"  Round {rnd}: added {n_new} pseudo-labels → {len(current_y_labeled)} total, acc={acc:.3f}")

    # Upper bound: all labels
    full_model = LogisticRegression(max_iter=1000, random_state=42)
    full_model.fit(X_train, y_train)
    full_acc = accuracy_score(y_test, full_model.predict(X_test))
    log(f"\nUpper bound (all {len(X_train)} labels): {full_acc:.3f}")
    log(f"Pseudo-labeling final: {round_accs[-1]:.3f} using {round_sizes[-1]} labels ({n_initial} real)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(round_sizes, round_accs, "o-", color="#2ecc71", linewidth=2, markersize=8, label="Pseudo-labeling")
    ax.axhline(y=base_acc, color="#e74c3c", linestyle="--", label=f"Baseline ({n_initial} labels): {base_acc:.3f}")
    ax.axhline(y=full_acc, color="#3498db", linestyle="--", label=f"Full supervision: {full_acc:.3f}")
    ax.set_xlabel("Total Training Samples (real + pseudo)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Pseudo-labeling: Accuracy vs Labeled Data")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pseudo_labeling.png"), dpi=150)
    plt.close()
    log("→ Plot saved: plots/pseudo_labeling.png")


def contrastive_concepts():
    """Explain self-supervised / contrastive learning concepts."""
    log("\n" + "=" * 60)
    log("PART 3: SELF-SUPERVISED LEARNING CONCEPTS")
    log("=" * 60)
    log("""
Self-supervised learning creates labels FROM the data itself:

  ┌───────────────────────┬──────────────────────────────────────────────┐
  │ Method                │ How it works                                 │
  ├───────────────────────┼──────────────────────────────────────────────┤
  │ Contrastive (SimCLR)  │ Augment image twice → embeddings should be  │
  │                       │ similar. Different images → dissimilar.      │
  ├───────────────────────┼──────────────────────────────────────────────┤
  │ BYOL                  │ Two networks: online + target. Online learns │
  │                       │ to predict target's output. No negatives.    │
  ├───────────────────────┼──────────────────────────────────────────────┤
  │ Masked prediction     │ Mask part of input, predict it.              │
  │ (BERT, MAE)           │ BERT masks tokens; MAE masks image patches.  │
  ├───────────────────────┼──────────────────────────────────────────────┤
  │ Next token (GPT)      │ Predict the next token in a sequence.        │
  │                       │ Entire internet is "labeled" data.           │
  └───────────────────────┴──────────────────────────────────────────────┘

Why it matters:
  - Labeled data is expensive ($1-10 per label for images, more for medical)
  - Unlabeled data is nearly infinite (internet, sensors, logs)
  - Foundation models (BERT, GPT, CLIP) are all pre-trained self-supervised
  - Fine-tuning a self-supervised backbone with 1% labels often beats
    training from scratch with 100% labels
""")


def run_semi_supervised_demo():
    log("SEMI-SUPERVISED & SELF-SUPERVISED LEARNING DEMO")
    log("=" * 60)
    semi_supervised_label_spreading()
    pseudo_labeling()
    contrastive_concepts()

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_semi_supervised_demo()
