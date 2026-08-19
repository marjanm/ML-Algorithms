"""
Logistic Regression — FROM SCRATCH
=====================================
No sklearn. Hand-coded sigmoid, cross-entropy loss, and gradient descent.

You'll see every step the algorithm takes:
  1. Initialise weights to zeros
  2. Forward pass: z = Xw + b  →  σ(z) = 1/(1+e^-z)
  3. Compute binary cross-entropy loss
  4. Backward pass: compute gradients dw, db
  5. Update weights: w -= lr * dw

Run:
    python logistic_regression_scratch.py
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
    """σ(z) = 1 / (1 + e^-z).  Clips z to avoid overflow."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred):
    """BCE = -1/n * Σ [y*log(p) + (1-y)*log(1-p)]"""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_logistic_regression(
    X_train, y_train, X_test, y_test,
    lr: float = 0.1,                # learning rate
    n_epochs: int = 200,             # training iterations
    verbose_every: int = 20,         # print loss every N epochs
):
    n_samples, n_features = X_train.shape

    # 1. initialise weights and bias to zeros
    w = np.zeros(n_features)
    b = 0.0

    train_losses = []
    test_losses = []

    lines = [
        "=" * 60, "  LOGISTIC REGRESSION — FROM SCRATCH", "=" * 60,
        f"  Features   : {n_features}",
        f"  Train size : {n_samples}",
        f"  lr         : {lr}",
        f"  Epochs     : {n_epochs}", "",
    ]

    for epoch in range(n_epochs):
        # 2. forward pass
        z = X_train @ w + b             # linear combination
        y_pred = sigmoid(z)             # apply sigmoid

        # 3. compute loss
        loss = binary_cross_entropy(y_train, y_pred)
        train_losses.append(loss)

        z_test = X_test @ w + b
        y_pred_test = sigmoid(z_test)
        test_loss = binary_cross_entropy(y_test, y_pred_test)
        test_losses.append(test_loss)

        # 4. backward pass — compute gradients
        error = y_pred - y_train                       # (n_samples,)
        dw = (1 / n_samples) * (X_train.T @ error)    # (n_features,)
        db = (1 / n_samples) * np.sum(error)           # scalar

        # 5. update weights
        w -= lr * dw
        b -= lr * db

        if (epoch + 1) % verbose_every == 0 or epoch == 0:
            lines.append(f"  Epoch {epoch+1:4d} | train_loss={loss:.4f} | test_loss={test_loss:.4f}")

    # final evaluation
    y_final = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
    accuracy = np.mean(y_final == y_test)

    lines += [
        "", f"  Final weights : {w}",
        f"  Final bias    : {b:.4f}",
        f"  Test accuracy : {accuracy:.4f}",
        "",
        "  Key formulas:",
        "    sigmoid(z)  = 1 / (1 + e^-z)",
        "    loss        = -1/n * Σ [y·log(σ) + (1-y)·log(1-σ)]",
        "    dw          = 1/n * X^T · (σ(z) - y)",
        "    db          = 1/n * Σ (σ(z) - y)",
        "    w_new       = w - lr * dw",
        "=" * 60,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # loss curve
    axes[0].plot(train_losses, label="Train loss")
    axes[0].plot(test_losses, label="Test loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].set_title("Loss curve (from-scratch logistic regression)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # decision boundary (2D only)
    if n_features == 2:
        h = 0.05
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = (sigmoid(grid @ w + b) >= 0.5).astype(int).reshape(xx.shape)
        axes[1].contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
        axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdBu", edgecolors="k", s=15, alpha=0.7)
        axes[1].set_title(f"Decision boundary (acc={accuracy:.3f})")
        axes[1].set_xlabel("Feature 0")
        axes[1].set_ylabel("Feature 1")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "logistic_regression_scratch.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")

    return w, b, accuracy


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, _ = generate_synthetic_data()
    train_logistic_regression(X_tr, y_tr, X_te, y_te)
