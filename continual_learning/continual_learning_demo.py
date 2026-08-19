"""
Continual / Lifelong Learning Demo
====================================
Shows catastrophic forgetting and how to mitigate it.

Experiment:
  1. Train an MLP on MNIST digits 0-4 (Task A)
  2. Then train on digits 5-9 (Task B)
  3. Measure Task A accuracy → it crashes (catastrophic forgetting)
  4. Apply two mitigations:
     a) Replay Buffer — store a small subset of Task A data, mix into Task B training
     b) EWC (Elastic Weight Consolidation) — penalize changes to important weights
"""

import os
import numpy as np
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


class SimpleMLP(nn.Module):
    def __init__(self, n_input, n_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def compute_fisher(model, data_loader):
    """
    Estimate diagonal Fisher Information Matrix.
    Compute per-sample gradients and accumulate squared gradients.
    """
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    n_samples = 0

    for X_batch, y_batch in data_loader:
        for i in range(len(X_batch)):
            model.zero_grad()
            output = model(X_batch[i:i+1])
            loss = nn.functional.cross_entropy(output, y_batch[i:i+1])
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2
            n_samples += 1

    for n in fisher:
        fisher[n] /= n_samples

    return fisher


def evaluate_task(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X))
        preds = logits.argmax(dim=1).numpy()
    return (preds == y).mean()


def run_continual_demo():
    log("CONTINUAL / LIFELONG LEARNING DEMO")
    log("=" * 60)

    digits = load_digits()
    X, y = digits.data, digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    mask_a = y < 5
    mask_b = y >= 5

    X_a, y_a = X[mask_a], y[mask_a]
    X_b, y_b = X[mask_b], y[mask_b]

    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size=0.3, random_state=42)
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_b, y_b, test_size=0.3, random_state=42)

    log(f"\nTask A: digits 0-4 ({len(X_a_train)} train, {len(X_a_test)} test)")
    log(f"Task B: digits 5-9 ({len(X_b_train)} train, {len(X_b_test)} test)")

    n_input = X.shape[1]
    n_classes = 10
    N_EPOCHS_A = 60
    N_EPOCHS_B = 60

    # ═══════════════════════════════════════════════════════
    # Train on Task A (shared starting point)
    # ═══════════════════════════════════════════════════════
    torch.manual_seed(42)
    base_model = SimpleMLP(n_input, n_classes)
    optimizer = optim.Adam(base_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader_a = DataLoader(
        TensorDataset(torch.FloatTensor(X_a_train), torch.LongTensor(y_a_train)),
        batch_size=32, shuffle=True
    )

    for epoch in range(N_EPOCHS_A):
        base_model.train()
        for xb, yb in loader_a:
            optimizer.zero_grad()
            loss = criterion(base_model(xb), yb)
            loss.backward()
            optimizer.step()

    acc_a_initial = evaluate_task(base_model, X_a_test, y_a_test)
    log(f"\nAfter Task A training:")
    log(f"  Task A accuracy: {acc_a_initial:.3f}")
    log(f"  Task B accuracy: {evaluate_task(base_model, X_b_test, y_b_test):.3f} (untrained)")

    # ═══════════════════════════════════════════════════════
    # Scenario 1: Naive — just train on Task B
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SCENARIO 1: NAIVE (catastrophic forgetting)")
    log(f"{'=' * 60}")

    model_naive = copy.deepcopy(base_model)
    optimizer_n = optim.Adam(model_naive.parameters(), lr=0.001)
    loader_b = DataLoader(
        TensorDataset(torch.FloatTensor(X_b_train), torch.LongTensor(y_b_train)),
        batch_size=32, shuffle=True
    )

    naive_a_curve = [acc_a_initial]
    naive_b_curve = [evaluate_task(model_naive, X_b_test, y_b_test)]

    for epoch in range(N_EPOCHS_B):
        model_naive.train()
        for xb, yb in loader_b:
            optimizer_n.zero_grad()
            loss = criterion(model_naive(xb), yb)
            loss.backward()
            optimizer_n.step()
        naive_a_curve.append(evaluate_task(model_naive, X_a_test, y_a_test))
        naive_b_curve.append(evaluate_task(model_naive, X_b_test, y_b_test))

    log(f"  Task A accuracy: {naive_a_curve[-1]:.3f}  (was {acc_a_initial:.3f})")
    log(f"  Task B accuracy: {naive_b_curve[-1]:.3f}")
    log(f"  → CATASTROPHIC FORGETTING: Task A dropped by {acc_a_initial - naive_a_curve[-1]:.3f}")

    # ═══════════════════════════════════════════════════════
    # Scenario 2: Replay Buffer
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SCENARIO 2: REPLAY BUFFER (store 10% of Task A)")
    log(f"{'=' * 60}")

    REPLAY_FRAC = 0.10
    n_replay = int(len(X_a_train) * REPLAY_FRAC)
    replay_idx = np.random.RandomState(42).choice(len(X_a_train), n_replay, replace=False)
    X_replay = X_a_train[replay_idx]
    y_replay = y_a_train[replay_idx]
    log(f"  Replay buffer: {n_replay} samples from Task A ({REPLAY_FRAC:.0%})")

    model_replay = copy.deepcopy(base_model)
    optimizer_r = optim.Adam(model_replay.parameters(), lr=0.001)

    combined_dataset = ConcatDataset([
        TensorDataset(torch.FloatTensor(X_b_train), torch.LongTensor(y_b_train)),
        TensorDataset(torch.FloatTensor(X_replay), torch.LongTensor(y_replay)),
    ])
    loader_combined = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    replay_a_curve = [acc_a_initial]
    replay_b_curve = [evaluate_task(model_replay, X_b_test, y_b_test)]

    for epoch in range(N_EPOCHS_B):
        model_replay.train()
        for xb, yb in loader_combined:
            optimizer_r.zero_grad()
            loss = criterion(model_replay(xb), yb)
            loss.backward()
            optimizer_r.step()
        replay_a_curve.append(evaluate_task(model_replay, X_a_test, y_a_test))
        replay_b_curve.append(evaluate_task(model_replay, X_b_test, y_b_test))

    log(f"  Task A accuracy: {replay_a_curve[-1]:.3f}  (was {acc_a_initial:.3f})")
    log(f"  Task B accuracy: {replay_b_curve[-1]:.3f}")
    log(f"  Forgetting: {acc_a_initial - replay_a_curve[-1]:.3f}")

    # ═══════════════════════════════════════════════════════
    # Scenario 3: EWC
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SCENARIO 3: EWC (Elastic Weight Consolidation)")
    log(f"{'=' * 60}")

    model_ewc = copy.deepcopy(base_model)

    fisher = compute_fisher(model_ewc, loader_a)
    old_params = {n: p.data.clone() for n, p in model_ewc.named_parameters()}

    fisher_norms = {n: f.mean().item() for n, f in fisher.items()}
    log(f"\n  Fisher Information (mean per layer):")
    for n, v in sorted(fisher_norms.items(), key=lambda x: -x[1]):
        log(f"    {n}: {v:.6f}")

    EWC_LAMBDA = 400

    optimizer_e = optim.Adam(model_ewc.parameters(), lr=0.001)

    ewc_a_curve = [acc_a_initial]
    ewc_b_curve = [evaluate_task(model_ewc, X_b_test, y_b_test)]

    for epoch in range(N_EPOCHS_B):
        model_ewc.train()
        for xb, yb in loader_b:
            optimizer_e.zero_grad()
            ce_loss = criterion(model_ewc(xb), yb)
            ewc_penalty = 0
            for n, p in model_ewc.named_parameters():
                if n in fisher:
                    ewc_penalty += (fisher[n] * (p - old_params[n]) ** 2).sum()
            loss = ce_loss + EWC_LAMBDA * ewc_penalty
            loss.backward()
            optimizer_e.step()
        ewc_a_curve.append(evaluate_task(model_ewc, X_a_test, y_a_test))
        ewc_b_curve.append(evaluate_task(model_ewc, X_b_test, y_b_test))

    log(f"\n  After Task B (λ={EWC_LAMBDA}):")
    log(f"  Task A accuracy: {ewc_a_curve[-1]:.3f}  (was {acc_a_initial:.3f})")
    log(f"  Task B accuracy: {ewc_b_curve[-1]:.3f}")
    log(f"  Forgetting: {acc_a_initial - ewc_a_curve[-1]:.3f}")

    # ═══════════════════════════════════════════════════════
    # Replay buffer size sweep
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("REPLAY BUFFER SIZE SWEEP")
    log(f"{'=' * 60}")

    fracs = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50]
    sweep_a = []
    sweep_b = []

    for frac in fracs:
        m = copy.deepcopy(base_model)
        opt = optim.Adam(m.parameters(), lr=0.001)
        n_rep = max(int(len(X_a_train) * frac), 0)

        if n_rep > 0:
            idx = np.random.RandomState(42).choice(len(X_a_train), n_rep, replace=False)
            ds = ConcatDataset([
                TensorDataset(torch.FloatTensor(X_b_train), torch.LongTensor(y_b_train)),
                TensorDataset(torch.FloatTensor(X_a_train[idx]), torch.LongTensor(y_a_train[idx])),
            ])
        else:
            ds = TensorDataset(torch.FloatTensor(X_b_train), torch.LongTensor(y_b_train))

        ldr = DataLoader(ds, batch_size=32, shuffle=True)
        for epoch in range(N_EPOCHS_B):
            m.train()
            for xb, yb in ldr:
                opt.zero_grad()
                loss = criterion(m(xb), yb)
                loss.backward()
                opt.step()

        a_acc = evaluate_task(m, X_a_test, y_a_test)
        b_acc = evaluate_task(m, X_b_test, y_b_test)
        sweep_a.append(a_acc)
        sweep_b.append(b_acc)
        log(f"  {frac:>5.0%} replay ({n_rep:>3} samples) → Task A: {a_acc:.3f}  Task B: {b_acc:.3f}")

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SUMMARY")
    log(f"{'=' * 60}")
    log(f"\n{'Method':<25} {'Task A':>8} {'Task B':>8} {'Avg':>8} {'Forgetting':>12}")
    log(f"{'─' * 62}")
    log(f"{'After Task A only':<25} {acc_a_initial:>8.3f} {'—':>8} {'—':>8} {'—':>12}")
    log(f"{'Naive':<25} {naive_a_curve[-1]:>8.3f} {naive_b_curve[-1]:>8.3f} {(naive_a_curve[-1]+naive_b_curve[-1])/2:>8.3f} {acc_a_initial-naive_a_curve[-1]:>12.3f}")
    log(f"{'Replay (10%)':<25} {replay_a_curve[-1]:>8.3f} {replay_b_curve[-1]:>8.3f} {(replay_a_curve[-1]+replay_b_curve[-1])/2:>8.3f} {acc_a_initial-replay_a_curve[-1]:>12.3f}")
    log(f"{'EWC (λ=' + str(EWC_LAMBDA) + ')':<25} {ewc_a_curve[-1]:>8.3f} {ewc_b_curve[-1]:>8.3f} {(ewc_a_curve[-1]+ewc_b_curve[-1])/2:>8.3f} {acc_a_initial-ewc_a_curve[-1]:>12.3f}")
    log(f"{'─' * 62}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Plot 1: Accuracy curves during Task B training
    ax = axes[0]
    epochs = range(len(naive_a_curve))
    ax.plot(epochs, naive_a_curve, label="Naive — Task A", color="#e74c3c", linewidth=2)
    ax.plot(epochs, naive_b_curve, label="Naive — Task B", color="#e74c3c", linewidth=2, linestyle="--")
    ax.plot(epochs, replay_a_curve, label="Replay — Task A", color="#2ecc71", linewidth=2)
    ax.plot(epochs, replay_b_curve, label="Replay — Task B", color="#2ecc71", linewidth=2, linestyle="--")
    ax.plot(epochs, ewc_a_curve, label="EWC — Task A", color="#3498db", linewidth=2)
    ax.plot(epochs, ewc_b_curve, label="EWC — Task B", color="#3498db", linewidth=2, linestyle="--")
    ax.set_xlabel("Epochs on Task B")
    ax.set_ylabel("Accuracy")
    ax.set_title("Catastrophic Forgetting: Naive vs Mitigations")
    ax.legend(fontsize=7, loc="center right")
    ax.set_ylim(-0.05, 1.05)

    # Plot 2: Replay buffer size effect
    ax = axes[1]
    pct_labels = [f"{f:.0%}" for f in fracs]
    x = np.arange(len(fracs))
    w = 0.3
    ax.bar(x - w/2, sweep_a, w, label="Task A", color="#e74c3c")
    ax.bar(x + w/2, sweep_b, w, label="Task B", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(pct_labels)
    ax.set_xlabel("Replay Buffer Size (% of Task A)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Replay Buffer Size vs Performance")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Plot 3: Final comparison bar chart
    ax = axes[2]
    methods = ["Naive", "Replay\n(10%)", f"EWC\n(λ={EWC_LAMBDA})"]
    a_accs = [naive_a_curve[-1], replay_a_curve[-1], ewc_a_curve[-1]]
    b_accs = [naive_b_curve[-1], replay_b_curve[-1], ewc_b_curve[-1]]
    x = np.arange(len(methods))
    w = 0.3
    bars_a = ax.bar(x - w/2, a_accs, w, label="Task A (digits 0-4)", color="#e74c3c")
    bars_b = ax.bar(x + w/2, b_accs, w, label="Task B (digits 5-9)", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Final Performance Comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    for bar_group in [bars_a, bars_b]:
        for b in bar_group:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "continual_learning.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/continual_learning.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log(f"{'=' * 60}")
    log("""
Catastrophic Forgetting: neural nets overwrite old knowledge when learning
new tasks. The weights for Task A get repurposed for Task B.

Mitigation approaches:
  ┌────────────────────────┬──────────────────────────────────────────┐
  │ Method                 │ Idea                                     │
  ├────────────────────────┼──────────────────────────────────────────┤
  │ Replay Buffer          │ Store some old data, mix into training.  │
  │                        │ Simple, effective, needs memory.         │
  ├────────────────────────┼──────────────────────────────────────────┤
  │ EWC                    │ Penalize changes to important weights.   │
  │                        │ No stored data, but complex to tune.     │
  ├────────────────────────┼──────────────────────────────────────────┤
  │ Progressive Networks   │ Add new capacity per task, freeze old.   │
  │                        │ No forgetting, but model grows.          │
  ├────────────────────────┼──────────────────────────────────────────┤
  │ Knowledge Distillation │ Old model as teacher for new training.   │
  │                        │ Preserves soft knowledge.                │
  └────────────────────────┴──────────────────────────────────────────┘

The Stability-Plasticity Dilemma:
  • Too stable (high λ / large replay) → can't learn new tasks
  • Too plastic (low λ / no replay)    → forgets old tasks
  • Split-label tasks (0-4 vs 5-9) are especially hard for EWC because
    the output layer weights are disjoint — replay buffers work better here

Real-world example: a fraud detection model must learn 2024 fraud patterns
without forgetting 2023 patterns. Retraining from scratch is wasteful.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_continual_demo()
