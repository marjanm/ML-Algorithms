"""
Softmax — FROM SCRATCH
========================
Hand-coded softmax function, cross-entropy loss, and multi-class
logistic regression (softmax regression) trained with gradient descent.

Softmax converts a vector of raw scores (logits) into probabilities
that sum to 1:

    softmax(z_i) = e^z_i / Σ e^z_j

Run:
    python softmax_scratch.py
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def softmax(z):
    """Numerically stable softmax.

    Subtract max per row to prevent overflow:
        e^(z - max) / Σ e^(z - max)
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(y_onehot, probs):
    """CE = -1/n * Σ Σ y_ij * log(p_ij)"""
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))


def one_hot(y, n_classes):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1
    return oh


def train_softmax_regression(X, y, n_classes, lr=0.1, n_epochs=200):
    """Multi-class logistic regression using softmax + gradient descent."""
    n, d = X.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    y_oh = one_hot(y, n_classes)
    losses = []

    for epoch in range(n_epochs):
        logits = X @ W + b                  # (n, n_classes)
        probs = softmax(logits)              # (n, n_classes)
        loss = cross_entropy_loss(y_oh, probs)
        losses.append(loss)

        error = probs - y_oh                 # (n, n_classes)
        dW = (1 / n) * (X.T @ error)        # (d, n_classes)
        db = (1 / n) * np.sum(error, axis=0) # (n_classes,)
        W -= lr * dW
        b -= lr * db

    preds = np.argmax(X @ W + b, axis=1)
    acc = np.mean(preds == y)
    return W, b, losses, acc


def run_softmax_demo():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=1500, n_features=5, n_informative=4, n_redundant=0,
        n_classes=4, n_clusters_per_class=1, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_classes = len(np.unique(y))

    lines = [
        "=" * 60, "  SOFTMAX — FROM SCRATCH", "=" * 60, "",
    ]

    # demonstrate softmax on raw examples
    examples = [
        np.array([[2.0, 1.0, 0.1]]),
        np.array([[1.0, 1.0, 1.0]]),
        np.array([[10.0, 1.0, 0.1]]),
        np.array([[-1.0, 0.0, 1.0]]),
    ]
    lines.append("  Softmax examples (input → output):")
    for ex in examples:
        out = softmax(ex)[0]
        lines.append(f"    {ex[0]} → [{', '.join(f'{v:.4f}' for v in out)}]  (sum={out.sum():.4f})")

    lines += [
        "",
        "  Properties of softmax:",
        "    - All outputs are positive and sum to 1 (valid probability distribution)",
        "    - Larger inputs get exponentially larger probabilities",
        "    - Equal inputs → uniform distribution (1/n each)",
        "    - Subtracting max(z) for numerical stability doesn't change the result",
        "",
    ]

    # train softmax regression
    W, b, losses, train_acc = train_softmax_regression(X_train, y_train, n_classes)
    test_probs = softmax(X_test @ W + b)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = np.mean(test_preds == y_test)

    lines += [
        f"  Softmax regression (4-class, 5 features):",
        f"    Train accuracy : {train_acc:.4f}",
        f"    Test accuracy  : {test_acc:.4f}",
        f"    Final loss     : {losses[-1]:.4f}",
        "",
        "  Formulas:",
        "    softmax(z_i)  = e^z_i / Σ_j e^z_j",
        "    CE loss       = -1/n * Σ_i Σ_c y_ic * log(p_ic)",
        "    dW            = 1/n * X^T · (softmax(z) - y_onehot)",
        "    W_new         = W - lr * dW",
        "=" * 60,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_softmax_demo()
