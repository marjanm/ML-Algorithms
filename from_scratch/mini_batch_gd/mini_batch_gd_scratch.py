"""
Mini-Batch Gradient Descent — FROM SCRATCH
=============================================
No sklearn, no PyTorch optimizers. Hand-coded SGD, mini-batch, and full-batch
on a logistic regression problem to compare convergence.

Three modes:
  - Full-batch GD   : use ALL samples per update (smooth but slow per step)
  - Mini-batch GD   : use a random subset per update (best tradeoff)
  - Stochastic GD   : use 1 sample per update (noisy but fast steps)

Run:
    python mini_batch_gd_scratch.py
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


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def train_gd(X, y, X_test, y_test, batch_size, lr=0.1, n_epochs=50, seed=42):
    """Train logistic regression with a given batch size."""
    np.random.seed(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = []

    for epoch in range(n_epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            X_b, y_b = X[idx], y[idx]
            m = len(y_b)

            p = sigmoid(X_b @ w + b)
            error = p - y_b
            dw = (1 / m) * (X_b.T @ error)
            db = (1 / m) * np.sum(error)
            w -= lr * dw
            b -= lr * db

        p_all = sigmoid(X @ w + b)
        losses.append(bce_loss(y, p_all))

    acc = np.mean((sigmoid(X_test @ w + b) >= 0.5).astype(int) == y_test)
    return losses, acc


def run_mini_batch_demo():
    X_train, X_test, y_train, y_test, _ = generate_synthetic_data(
        n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
    )

    n = len(y_train)
    configs = {
        f"Full-batch (bs={n})": n,
        "Mini-batch (bs=64)": 64,
        "Mini-batch (bs=16)": 16,
        "SGD (bs=1)": 1,
    }

    n_epochs = 50
    lines = [
        "=" * 60, "  MINI-BATCH GRADIENT DESCENT — FROM SCRATCH", "=" * 60,
        f"  Train size : {n}",
        f"  Epochs     : {n_epochs}", "",
    ]

    results = {}
    for name, bs in configs.items():
        losses, acc = train_gd(X_train, y_train, X_test, y_test,
                               batch_size=bs, lr=0.1, n_epochs=n_epochs)
        results[name] = losses
        updates_per_epoch = (n + bs - 1) // bs
        lines.append(f"  {name:30s} | updates/epoch={updates_per_epoch:5d} | "
                     f"final_loss={losses[-1]:.4f} | acc={acc:.4f}")

    lines += [
        "", "  Key observations:",
        "    - Full-batch: smoothest curve but fewest weight updates per epoch",
        "    - Mini-batch (64): good balance — smooth enough, many updates",
        "    - SGD (bs=1): noisiest curve but most updates — explores the loss surface",
        "    - All converge to similar final loss; the PATH differs",
        "",
        "  Formulas (same as logistic regression, applied to each batch):",
        "    dw = 1/m * X_batch^T · (σ(z) - y_batch)",
        "    w  = w - lr * dw",
        "=" * 60,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    for (name, losses), col in zip(results.items(), colors):
        axes[0].plot(losses, label=name, color=col, linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Convergence by batch size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # zoomed-in last 20 epochs
    for (name, losses), col in zip(results.items(), colors):
        axes[1].plot(range(30, 50), losses[30:], label=name, color=col, linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Zoomed: last 20 epochs (noise difference)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "mini_batch_gd.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_mini_batch_demo()
