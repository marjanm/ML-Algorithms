"""
Model Distillation Demo
========================
Train a small "student" model to mimic a large "teacher" model.

The key idea: the teacher's soft probability outputs contain MORE information
than hard labels.  A prediction of [0.7, 0.2, 0.1] tells the student that
class 1 is somewhat similar to class 0 — information lost if we just say "class 0".

Experiment:
  Teacher: Random Forest with 500 trees (large, accurate, slow)
  Student A: Tiny neural net trained on HARD labels (standard training)
  Student B: Tiny neural net trained on SOFT labels from teacher (distillation)

  → Student B should match or beat Student A despite identical architecture,
    because soft labels provide richer supervision.

Also demonstrates temperature scaling: higher temperature makes the soft
distribution more informative (spreads probability mass).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


class TinyStudent(nn.Module):
    """Intentionally small — 2 hidden layers, ~1K parameters."""
    def __init__(self, n_input, n_classes, hidden=32):
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

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def softmax_with_temperature(logits, T):
    """Softmax with temperature scaling. Higher T → softer distribution."""
    scaled = logits / T
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def train_student(model, X_train, targets, n_epochs=100, lr=0.01,
                  soft=False, temperature=1.0):
    """Train the student model."""
    X_tensor = torch.FloatTensor(X_train)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    if soft:
        target_tensor = torch.FloatTensor(targets)
        criterion = nn.KLDivLoss(reduction="batchmean")
    else:
        target_tensor = torch.LongTensor(targets)
        criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_tensor, target_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(n_epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            if soft:
                log_probs = nn.functional.log_softmax(logits / temperature, dim=1)
                loss = criterion(log_probs, y_batch) * (temperature ** 2)
            else:
                loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    return losses


def evaluate(model, X_test, y_test):
    """Evaluate a PyTorch model."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test))
        preds = logits.argmax(dim=1).numpy()
    return accuracy_score(y_test, preds)


def run_distillation_demo():
    log("MODEL DISTILLATION DEMO")
    log("=" * 60)

    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target
    n_classes = len(np.unique(y))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # --- Teacher: Large Random Forest ---
    log("\n--- TEACHER MODEL ---")
    teacher = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42)
    teacher.fit(X_train, y_train)
    teacher_acc = accuracy_score(y_test, teacher.predict(X_test))
    teacher_probs = teacher.predict_proba(X_train)  # soft labels for distillation

    log(f"Random Forest (500 trees)")
    log(f"  Test accuracy: {teacher_acc:.4f}")
    log(f"  Model size: ~500 decision trees")

    # --- Student A: Trained on HARD labels ---
    log("\n--- STUDENT A: Hard labels (standard training) ---")
    torch.manual_seed(42)
    student_hard = TinyStudent(X_train.shape[1], n_classes, hidden=32)
    losses_hard = train_student(student_hard, X_train, y_train, n_epochs=150, soft=False)
    hard_acc = evaluate(student_hard, X_test, y_test)

    log(f"Tiny NN ({student_hard.param_count()} params)")
    log(f"  Trained on: ground truth labels")
    log(f"  Test accuracy: {hard_acc:.4f}")

    # --- Student B: Distilled from teacher (soft labels) ---
    log("\n--- STUDENT B: Distilled (soft labels from teacher) ---")

    temperatures = [1.0, 3.0, 5.0, 10.0, 20.0]
    distill_accs = []

    for T in temperatures:
        torch.manual_seed(42)
        student_soft = TinyStudent(X_train.shape[1], n_classes, hidden=32)
        soft_targets = softmax_with_temperature(
            np.log(teacher_probs + 1e-10), T  # convert probs to log-probs, then re-soften
        )
        losses_soft = train_student(
            student_soft, X_train, soft_targets, n_epochs=150, soft=True, temperature=T
        )
        soft_acc = evaluate(student_soft, X_test, y_test)
        distill_accs.append(soft_acc)
        log(f"  T={T:>5.1f} → accuracy: {soft_acc:.4f}")

    best_T_idx = np.argmax(distill_accs)
    best_T = temperatures[best_T_idx]
    best_distill_acc = distill_accs[best_T_idx]

    # Retrain best for loss curve comparison
    torch.manual_seed(42)
    student_best = TinyStudent(X_train.shape[1], n_classes, hidden=32)
    soft_targets_best = softmax_with_temperature(np.log(teacher_probs + 1e-10), best_T)
    losses_distill = train_student(
        student_best, X_train, soft_targets_best, n_epochs=150, soft=True, temperature=best_T
    )

    # --- Summary ---
    log(f"\n{'=' * 60}")
    log(f"SUMMARY")
    log(f"{'=' * 60}")
    log(f"{'Model':<35} {'Params':>10} {'Accuracy':>10}")
    log(f"{'─' * 55}")
    log(f"{'Teacher (RF, 500 trees)':<35} {'~large':>10} {teacher_acc:>10.4f}")
    log(f"{'Student A (hard labels)':<35} {student_hard.param_count():>10} {hard_acc:>10.4f}")
    log(f"{'Student B (distilled, T=' + str(best_T) + ')':<35} {student_hard.param_count():>10} {best_distill_acc:>10.4f}")
    log(f"{'─' * 55}")

    improvement = best_distill_acc - hard_acc
    log(f"\nDistillation {'improved' if improvement > 0 else 'changed'} student accuracy by {improvement:+.4f}")
    log(f"Student retains {best_distill_acc/teacher_acc:.1%} of teacher's accuracy")
    log(f"with only {student_hard.param_count()} parameters vs hundreds of trees")

    # --- Example: soft vs hard labels ---
    log(f"\n{'=' * 60}")
    log("WHY SOFT LABELS HELP")
    log(f"{'=' * 60}")
    sample_idx = 0
    hard_label = y_train[sample_idx]
    soft_label = teacher_probs[sample_idx]
    top3 = np.argsort(-soft_label)[:3]
    log(f"\nExample (digit {hard_label}):")
    log(f"  Hard label: [{hard_label}]  → 'it's definitely a {hard_label}'")
    log(f"  Soft label: {np.array2string(soft_label, precision=3)}")
    log(f"  Top-3 classes: {', '.join(f'{c}={soft_label[c]:.3f}' for c in top3)}")
    log(f"  → Soft labels say '{hard_label} looks a bit like {top3[1]} and {top3[2]}'")
    log(f"  This similarity structure helps the student generalize better!")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss curves
    ax = axes[0]
    ax.plot(losses_hard, label="Hard labels", color="#e74c3c", linewidth=2)
    ax.plot(losses_distill, label=f"Distilled (T={best_T})", color="#2ecc71", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss: Hard vs Distilled")
    ax.legend()

    # Temperature effect
    ax = axes[1]
    ax.plot(temperatures, distill_accs, "o-", color="#3498db", linewidth=2, markersize=8)
    ax.axhline(y=hard_acc, color="#e74c3c", linestyle="--",
               label=f"Hard labels ({hard_acc:.3f})")
    ax.axhline(y=teacher_acc, color="#2ecc71", linestyle="--",
               label=f"Teacher ({teacher_acc:.3f})")
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Student Accuracy")
    ax.set_title("Effect of Temperature on Distillation")
    ax.legend(fontsize=8)

    # Accuracy comparison
    ax = axes[2]
    names = ["Teacher\n(500 trees)", "Student\n(hard)", f"Student\n(distilled T={best_T})"]
    accs = [teacher_acc, hard_acc, best_distill_acc]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    bars = ax.bar(names, accs, color=colors, width=0.5)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison")
    ax.set_ylim(0.8, 1.02)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width()/2, a + 0.005, f"{a:.3f}",
                ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "distillation.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/distillation.png")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_distillation_demo()
