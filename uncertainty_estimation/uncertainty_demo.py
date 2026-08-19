"""
Uncertainty Estimation — Demo
================================
Models say "class 1 with 95% confidence" but are they actually right 95%
of the time? This demo shows how to get honest uncertainty estimates:

1. Softmax baseline     — overconfident, not calibrated
2. MC Dropout           — run inference N times with dropout ON
3. Deep Ensembles       — train M independent models, average predictions
4. Comparison: uncertainty correlates with actual errors
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

torch.manual_seed(42)
np.random.seed(42)


class MCDropoutNet(nn.Module):
    """Network with dropout that stays ON during inference."""
    def __init__(self, input_dim=2, hidden=64, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, 2),
        )
        self.dropout_rate = dropout_rate

    def forward(self, x):
        return self.net(x)

    def predict_with_uncertainty(self, x, n_samples=50):
        """Run N forward passes with dropout ON → distribution of predictions."""
        self.train()  # keep dropout active
        probs_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs.numpy())

        probs_array = np.array(probs_list)  # (n_samples, n_points, n_classes)
        mean_probs = probs_array.mean(axis=0)
        std_probs = probs_array.std(axis=0)
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        return mean_probs, std_probs, predictive_entropy


def train_model(model, X_train, y_train, epochs=200, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
    return model


def run_demo():
    log("UNCERTAINTY ESTIMATION — DEMO")
    log("=" * 60)

    # Data: two moons with noise
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Add out-of-distribution points (far from training data)
    X_ood = np.array([[-2, 2], [3, -1], [0, 3], [-1, -2], [2.5, 2.5]])
    y_ood = np.array([0, 1, 0, 1, 0])

    X_all = np.vstack([X_test, X_ood])
    y_all = np.concatenate([y_test, y_ood])
    is_ood = np.array([False]*len(X_test) + [True]*len(X_ood))

    log(f"\n  Train: {len(X_train)}, Test: {len(X_test)}, OOD: {len(X_ood)}")

    # ═══════════════════════════════════════════════════════
    # Method 1: Standard softmax (overconfident baseline)
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("METHOD 1: STANDARD SOFTMAX (baseline)")
    log("=" * 60)

    model_base = MCDropoutNet(dropout_rate=0.0)
    train_model(model_base, X_train, y_train)

    model_base.eval()
    with torch.no_grad():
        logits = model_base(torch.FloatTensor(X_all))
        softmax_probs = torch.softmax(logits, dim=1).numpy()

    softmax_conf = softmax_probs.max(axis=1)
    softmax_pred = softmax_probs.argmax(axis=1)
    softmax_correct = (softmax_pred == y_all)

    log(f"\n  Accuracy (test): {accuracy_score(y_test, softmax_pred[:len(y_test)]):.4f}")
    log(f"  Mean confidence (test):  {softmax_conf[:len(y_test)].mean():.4f}")
    log(f"  Mean confidence (OOD):   {softmax_conf[len(y_test):].mean():.4f}")
    log(f"  → Softmax is overconfident even on OOD points!")

    # ═══════════════════════════════════════════════════════
    # Method 2: MC Dropout
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("METHOD 2: MC DROPOUT (50 forward passes)")
    log("=" * 60)

    model_mc = MCDropoutNet(dropout_rate=0.2)
    train_model(model_mc, X_train, y_train)

    mc_mean, mc_std, mc_entropy = model_mc.predict_with_uncertainty(
        torch.FloatTensor(X_all), n_samples=50
    )

    mc_pred = mc_mean.argmax(axis=1)
    mc_correct = (mc_pred == y_all)
    mc_uncertainty = mc_std.max(axis=1)

    log(f"\n  Accuracy (test): {accuracy_score(y_test, mc_pred[:len(y_test)]):.4f}")
    log(f"  Mean uncertainty (test): {mc_uncertainty[:len(y_test)].mean():.4f}")
    log(f"  Mean uncertainty (OOD):  {mc_uncertainty[len(y_test):].mean():.4f}")
    log(f"  → MC Dropout is more uncertain on OOD points")

    # ═══════════════════════════════════════════════════════
    # Method 3: Deep Ensembles
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("METHOD 3: DEEP ENSEMBLE (5 independent models)")
    log("=" * 60)

    n_ensemble = 5
    ensemble_probs = []

    for i in range(n_ensemble):
        torch.manual_seed(i * 100)
        m = MCDropoutNet(dropout_rate=0.0)
        train_model(m, X_train, y_train, epochs=200)
        m.eval()
        with torch.no_grad():
            probs = torch.softmax(m(torch.FloatTensor(X_all)), dim=1).numpy()
            ensemble_probs.append(probs)

    ens_probs = np.array(ensemble_probs)  # (5, n_points, 2)
    ens_mean = ens_probs.mean(axis=0)
    ens_std = ens_probs.std(axis=0)
    ens_pred = ens_mean.argmax(axis=1)
    ens_correct = (ens_pred == y_all)
    ens_uncertainty = ens_std.max(axis=1)

    log(f"\n  Accuracy (test): {accuracy_score(y_test, ens_pred[:len(y_test)]):.4f}")
    log(f"  Mean uncertainty (test): {ens_uncertainty[:len(y_test)].mean():.4f}")
    log(f"  Mean uncertainty (OOD):  {ens_uncertainty[len(y_test):].mean():.4f}")
    log(f"  → Ensemble gives highest OOD uncertainty")

    # ═══════════════════════════════════════════════════════
    # Uncertainty vs correctness
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("UNCERTAINTY vs CORRECTNESS")
    log("=" * 60)

    for name, uncertainty, correct in [
        ("Softmax", 1 - softmax_conf, softmax_correct),
        ("MC Dropout", mc_uncertainty, mc_correct),
        ("Ensemble", ens_uncertainty, ens_correct),
    ]:
        correct_unc = uncertainty[correct].mean()
        wrong_unc = uncertainty[~correct].mean() if (~correct).sum() > 0 else 0
        log(f"\n  {name}:")
        log(f"    Uncertainty when CORRECT: {correct_unc:.4f}")
        log(f"    Uncertainty when WRONG:   {wrong_unc:.4f}")
        log(f"    Ratio (want >> 1):        {wrong_unc / (correct_unc + 1e-10):.2f}x")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Decision boundaries with uncertainty
    xx, yy = np.meshgrid(np.linspace(-3, 4, 100), np.linspace(-2, 3.5, 100))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    grid_t = torch.FloatTensor(grid)

    # Softmax confidence
    model_base.eval()
    with torch.no_grad():
        grid_probs = torch.softmax(model_base(grid_t), dim=1).numpy()
    grid_conf = grid_probs.max(axis=1).reshape(xx.shape)

    ax = axes[0, 0]
    ax.contourf(xx, yy, grid_conf, levels=20, cmap="RdYlGn", alpha=0.7)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", s=5, alpha=0.5)
    ax.scatter(X_ood[:, 0], X_ood[:, 1], c="black", marker="x", s=100, linewidths=2, label="OOD")
    ax.set_title("Softmax Confidence")
    ax.legend()

    # MC Dropout uncertainty map
    mc_grid_mean, mc_grid_std, mc_grid_entropy = model_mc.predict_with_uncertainty(grid_t, n_samples=30)
    grid_unc = mc_grid_std.max(axis=1).reshape(xx.shape)

    ax = axes[0, 1]
    ax.contourf(xx, yy, grid_unc, levels=20, cmap="YlOrRd", alpha=0.7)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", s=5, alpha=0.5)
    ax.scatter(X_ood[:, 0], X_ood[:, 1], c="black", marker="x", s=100, linewidths=2, label="OOD")
    ax.set_title("MC Dropout Uncertainty")
    ax.legend()

    # Ensemble uncertainty map
    ens_grid_probs = []
    for i in range(n_ensemble):
        torch.manual_seed(i * 100)
        m = MCDropoutNet(dropout_rate=0.0)
        train_model(m, X_train, y_train, epochs=200)
        m.eval()
        with torch.no_grad():
            ens_grid_probs.append(torch.softmax(m(grid_t), dim=1).numpy())
    ens_grid = np.array(ens_grid_probs)
    grid_ens_unc = ens_grid.std(axis=0).max(axis=1).reshape(xx.shape)

    ax = axes[0, 2]
    ax.contourf(xx, yy, grid_ens_unc, levels=20, cmap="YlOrRd", alpha=0.7)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr", s=5, alpha=0.5)
    ax.scatter(X_ood[:, 0], X_ood[:, 1], c="black", marker="x", s=100, linewidths=2, label="OOD")
    ax.set_title("Ensemble Uncertainty")
    ax.legend()

    # Row 2: Uncertainty distributions
    ax = axes[1, 0]
    ax.hist(1 - softmax_conf[softmax_correct], bins=20, alpha=0.6, label="Correct", color="#2ecc71")
    if (~softmax_correct).sum() > 0:
        ax.hist(1 - softmax_conf[~softmax_correct], bins=20, alpha=0.6, label="Wrong", color="#e74c3c")
    ax.set_title("Softmax: Uncertainty by Correctness")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(mc_uncertainty[mc_correct], bins=20, alpha=0.6, label="Correct", color="#2ecc71")
    if (~mc_correct).sum() > 0:
        ax.hist(mc_uncertainty[~mc_correct], bins=20, alpha=0.6, label="Wrong", color="#e74c3c")
    ax.set_title("MC Dropout: Uncertainty by Correctness")
    ax.legend()

    ax = axes[1, 2]
    ax.hist(ens_uncertainty[ens_correct], bins=20, alpha=0.6, label="Correct", color="#2ecc71")
    if (~ens_correct).sum() > 0:
        ax.hist(ens_uncertainty[~ens_correct], bins=20, alpha=0.6, label="Wrong", color="#e74c3c")
    ax.set_title("Ensemble: Uncertainty by Correctness")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "uncertainty_estimation.png"), dpi=150)
    plt.close()
    log(f"\n→ Plot saved: plots/uncertainty_estimation.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
  ┌─────────────────┬──────────────────────────────────────────┐
  │ Method          │ How it works                             │
  ├─────────────────┼──────────────────────────────────────────┤
  │ Softmax         │ Single forward pass. Overconfident.      │
  │                 │ NOT a real probability.                   │
  │ MC Dropout      │ N forward passes with dropout ON.        │
  │                 │ Cheap. Approximates Bayesian inference.  │
  │ Deep Ensemble   │ Train M models with different seeds.     │
  │                 │ Best quality. M× training cost.          │
  │ Temperature     │ Scale logits before softmax.             │
  │ scaling         │ Calibrates probabilities (see calib/).   │
  │ Bayesian NN     │ Full posterior over weights.             │
  │                 │ Theoretically ideal, hard to scale.      │
  └─────────────────┴──────────────────────────────────────────┘

  When to use:
    • Medical/safety-critical: ensemble + abstain on high uncertainty
    • Active learning: query labels for high-uncertainty points
    • OOD detection: flag inputs the model hasn't seen before
    • Debugging: high uncertainty on "easy" examples → data issue
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
