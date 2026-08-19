"""
Multi-Layer Perceptron (MLP) — the basic feedforward neural network
====================================================================
Trains an MLP on synthetic data to show how a vanilla neural network
learns non-linear decision boundaries.

Run:
    python mlp_demo.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


class MLP(nn.Module):
    """Simple 3-layer feedforward network."""

    def __init__(
        self,
        input_dim: int = 2,          # number of input features
        hidden1: int = 64,           # neurons in first hidden layer
        hidden2: int = 32,           # neurons in second hidden layer
        output_dim: int = 1,         # 1 for binary classification (sigmoid output)
        dropout: float = 0.2,        # dropout rate — fraction of neurons randomly zeroed during training to prevent overfitting
        activation: str = "relu",    # activation function: "relu", "tanh", "leaky_relu"
    ):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}[activation]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),     # fully connected: input_dim -> hidden1
            nn.BatchNorm1d(hidden1),           # normalise activations — stabilises and speeds up training
            act_fn(),                          # non-linearity — without this, stacked layers collapse to one linear layer
            nn.Dropout(dropout),               # randomly zero neurons to prevent co-adaptation
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim),
            nn.Sigmoid(),                      # squash output to [0, 1] for binary probability
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(
    # --- data ---
    n_samples: int = 2000,            # total data points to generate
    noise: float = 0.2,              # noise level in the moon-shaped data
    test_size: float = 0.2,          # fraction held out for testing
    # --- architecture ---
    hidden1: int = 64,
    hidden2: int = 32,
    dropout: float = 0.2,
    activation: str = "relu",
    # --- training ---
    epochs: int = 200,               # number of full passes through the training data
    batch_size: int = 64,            # samples per gradient update — smaller = noisier but more updates
    learning_rate: float = 0.001,    # step size for the optimiser — too high = overshoot, too low = slow
    weight_decay: float = 1e-4,      # L2 regularisation strength in the optimiser
    optimizer_name: str = "adam",     # "adam" (adaptive lr), "sgd" (simple), "rmsprop"
    # --- misc ---
    random_state: int = 42,
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)

    model = MLP(input_dim=2, hidden1=hidden1, hidden2=hidden2,
                dropout=dropout, activation=activation)

    criterion = nn.BCELoss()              # binary cross-entropy loss — standard for binary classification
    opt_map = {"adam": optim.Adam, "sgd": optim.SGD, "rmsprop": optim.RMSprop}
    optimizer = opt_map[optimizer_name](model.parameters(), lr=learning_rate,
                                        weight_decay=weight_decay)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()                # backpropagation — compute gradients
            optimizer.step()               # update weights using gradients
            epoch_loss += loss.item()
        train_losses.append(epoch_loss)

    model.eval()
    with torch.no_grad():
        y_proba = model(X_test_t).squeeze().numpy()
    y_pred = (y_proba > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    lines = [
        "=" * 50,
        "  MLP (Multi-Layer Perceptron)  —  Results",
        "=" * 50,
        f"  Architecture : {2} -> {hidden1} -> {hidden2} -> 1",
        f"  Activation   : {activation}",
        f"  Dropout      : {dropout}",
        f"  Epochs       : {epochs}",
        f"  Batch size   : {batch_size}",
        f"  Learning rate: {learning_rate}",
        f"  Optimizer    : {optimizer_name}",
        f"  Accuracy     : {acc:.4f}",
        "=" * 50,
        "",
        "Classification Report:",
        report,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)

    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    # --- Plot: loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("MLP — Training Loss Curve")
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "mlp_loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

    # --- Plot: decision boundary ---
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        Z = model(grid).squeeze().numpy().reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, levels=50, cmap="RdYlGn", alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
               c="#e74c3c", edgecolors="k", s=20, label="Class 0", alpha=0.7)
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
               c="#2ecc71", edgecolors="k", s=20, label="Class 1", alpha=0.7)
    ax.set_title(f"MLP — Decision Boundary (acc={acc:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "mlp_decision_boundary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


if __name__ == "__main__":
    train_and_evaluate()
