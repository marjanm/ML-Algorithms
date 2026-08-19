"""
Robustness & Adversarial Thinking — Demo
==========================================
Shows how tiny, imperceptible perturbations can fool neural networks:

1. Train a classifier on MNIST digits
2. FGSM attack — create adversarial examples
3. PGD attack — iterative, stronger attack
4. C&W attack — optimization-based, minimal perturbation
5. AutoAttack — ensemble of attacks (gold standard evaluation)
6. Show before/after: human sees "7", model says "3"
7. Adversarial training — retrain with adversarial examples as defense
8. Compare all attacks against standard vs adversarially-trained models
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


torch.manual_seed(42)
np.random.seed(42)


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def load_mnist_subset(n_train=5000, n_test=1000):
    """Load MNIST subset using torchvision."""
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    X_train = train_ds.data[:n_train].float().unsqueeze(1) / 255.0
    y_train = train_ds.targets[:n_train]
    X_test = test_ds.data[:n_test].float().unsqueeze(1) / 255.0
    y_test = test_ds.targets[:n_test]
    return X_train, y_train, X_test, y_test


def fgsm_attack(model, images, labels, epsilon):
    """
    Fast Gradient Sign Method:
    perturb = epsilon * sign(gradient of loss w.r.t. input)
    """
    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()

    perturbation = epsilon * images_adv.grad.sign()
    adv_images = torch.clamp(images_adv + perturbation, 0, 1)
    return adv_images.detach(), perturbation.detach()


def pgd_attack(model, images, labels, epsilon, alpha=0.01, n_steps=40):
    """
    Projected Gradient Descent: iterative FGSM with small steps.
    Stronger than FGSM — takes many small gradient steps instead of one big one,
    projecting back into the ε-ball after each step.
    """
    adv_images = images.clone().detach()
    adv_images += torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)

    for _ in range(n_steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        grad_sign = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha * grad_sign
        # Project back into ε-ball around original image
        delta = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + delta, 0, 1).detach()

    perturbation = adv_images - images
    return adv_images, perturbation


def cw_attack(model, images, labels, c=1.0, lr=0.01, n_steps=100):
    """
    Carlini & Wagner (simplified L2 attack):
    Optimizes for the smallest perturbation that changes the prediction.
    Instead of a fixed ε, it finds the minimum noise needed.
    """
    adv_images = images.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([adv_images], lr=lr)

    for _ in range(n_steps):
        outputs = model(adv_images)
        # f(x) = max(Z_true - max(Z_other), 0) — want to make true class score lower
        one_hot = torch.zeros_like(outputs).scatter_(1, labels.unsqueeze(1), 1)
        real = (one_hot * outputs).sum(dim=1)
        other = ((1 - one_hot) * outputs - one_hot * 1e4).max(dim=1)[0]
        f_loss = torch.clamp(real - other, min=0).sum()

        l2_loss = ((adv_images - images) ** 2).sum()
        loss = l2_loss + c * f_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        adv_images.data = torch.clamp(adv_images.data, 0, 1)

    perturbation = adv_images.detach() - images
    return adv_images.detach(), perturbation.detach()


def auto_attack(model, images, labels, epsilon):
    """
    Simplified AutoAttack: runs multiple attacks and picks the one
    that fools each sample. In practice, AutoAttack uses APGD-CE,
    APGD-T, FAB, and Square Attack. We approximate with PGD variants.
    """
    model.eval()
    best_adv = images.clone()
    with torch.no_grad():
        orig_correct = (model(images).argmax(dim=1) == labels)

    # Attack 1: PGD with cross-entropy loss (untargeted)
    adv1, _ = pgd_attack(model, images, labels, epsilon, alpha=epsilon/4, n_steps=50)
    with torch.no_grad():
        preds1 = model(adv1).argmax(dim=1)
        fooled1 = (preds1 != labels) & orig_correct
    best_adv[fooled1] = adv1[fooled1]

    # Attack 2: PGD with different step size
    adv2, _ = pgd_attack(model, images, labels, epsilon, alpha=epsilon/10, n_steps=100)
    with torch.no_grad():
        preds2 = model(adv2).argmax(dim=1)
        still_correct = (model(best_adv).argmax(dim=1) == labels) & orig_correct
        fooled2 = (preds2 != labels) & still_correct
    best_adv[fooled2] = adv2[fooled2]

    # Attack 3: Random-start PGD (different initialization)
    imgs_rand = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    imgs_rand = torch.clamp(imgs_rand, 0, 1)
    adv3, _ = pgd_attack(model, imgs_rand, labels, epsilon, alpha=epsilon/4, n_steps=50)
    with torch.no_grad():
        preds3 = model(adv3).argmax(dim=1)
        still_correct = (model(best_adv).argmax(dim=1) == labels) & orig_correct
        fooled3 = (preds3 != labels) & still_correct
    best_adv[fooled3] = adv3[fooled3]

    perturbation = best_adv - images
    return best_adv.detach(), perturbation.detach()


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    return (preds == y).float().mean().item()


def train_standard(model, X_train, y_train, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def train_adversarial(model, X_train, y_train, epsilon=0.2, epochs=10, lr=0.001):
    """Adversarial training: augment each batch with FGSM adversarial examples."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            # Generate adversarial examples
            xb_adv, _ = fgsm_attack(model, xb, yb, epsilon)

            # Train on both clean and adversarial
            x_combined = torch.cat([xb, xb_adv])
            y_combined = torch.cat([yb, yb])

            optimizer.zero_grad()
            loss = criterion(model(x_combined), y_combined)
            loss.backward()
            optimizer.step()
    return model


def run_demo():
    log("ADVERSARIAL ROBUSTNESS — DEMO")
    log("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist_subset()
    log(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

    # ═══════════════════════════════════════════════════════
    # Train standard model
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("STEP 1: TRAIN STANDARD MODEL")
    log("=" * 60)

    model_std = SimpleConvNet()
    train_standard(model_std, X_train, y_train, epochs=10)
    clean_acc = evaluate(model_std, X_test, y_test)
    log(f"\n  Clean accuracy: {clean_acc:.4f}")

    # ═══════════════════════════════════════════════════════
    # FGSM attack at various epsilon
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("STEP 2: FGSM ATTACK")
    log("=" * 60)

    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.3]
    std_accs = []

    log(f"\n  {'Epsilon':>8} {'Accuracy':>10} {'Drop':>8}")
    log(f"  {'-' * 30}")

    for eps in epsilons:
        if eps == 0:
            acc = clean_acc
        else:
            X_adv, _ = fgsm_attack(model_std, X_test, y_test, eps)
            acc = evaluate(model_std, X_adv, y_test)
        std_accs.append(acc)
        log(f"  {eps:>8.2f} {acc:>10.4f} {acc - clean_acc:>+8.4f}")

    # ═══════════════════════════════════════════════════════
    # PGD attack
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("STEP 3: PGD ATTACK (iterative, stronger)")
    log("=" * 60)

    pgd_accs = []
    log(f"\n  {'Epsilon':>8} {'FGSM':>10} {'PGD':>10} {'PGD stronger?':>14}")
    log(f"  {'-' * 46}")

    for i, eps in enumerate(epsilons):
        if eps == 0:
            acc = clean_acc
        else:
            X_adv_pgd, _ = pgd_attack(model_std, X_test, y_test, eps)
            acc = evaluate(model_std, X_adv_pgd, y_test)
        pgd_accs.append(acc)
        stronger = "YES" if acc < std_accs[i] else "—"
        log(f"  {eps:>8.2f} {std_accs[i]:>10.4f} {acc:>10.4f} {stronger:>14}")

    # ═══════════════════════════════════════════════════════
    # C&W attack (at a single epsilon for comparison)
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("STEP 4: C&W ATTACK (optimization-based, L2 minimal)")
    log("=" * 60)

    X_adv_cw, pert_cw = cw_attack(model_std, X_test[:200], y_test[:200], c=10.0, lr=0.01, n_steps=100)
    cw_acc = evaluate(model_std, X_adv_cw, y_test[:200])
    cw_l2 = torch.norm(pert_cw.view(pert_cw.size(0), -1), dim=1).mean().item()
    fgsm_adv_200, fgsm_pert_200 = fgsm_attack(model_std, X_test[:200], y_test[:200], epsilon=0.2)
    fgsm_l2 = torch.norm(fgsm_pert_200.view(fgsm_pert_200.size(0), -1), dim=1).mean().item()

    log(f"\n  C&W  accuracy (first 200 test): {cw_acc:.4f}")
    log(f"  C&W  avg L2 perturbation:       {cw_l2:.4f}")
    log(f"  FGSM accuracy (ε=0.2, same 200): {evaluate(model_std, fgsm_adv_200, y_test[:200]):.4f}")
    log(f"  FGSM avg L2 perturbation:        {fgsm_l2:.4f}")
    log(f"\n  → C&W uses much smaller perturbations to achieve similar fooling.")

    # ═══════════════════════════════════════════════════════
    # AutoAttack (ensemble, gold standard)
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("STEP 5: AUTOATTACK (ensemble — gold standard)")
    log("=" * 60)

    auto_accs = []
    log(f"\n  {'Epsilon':>8} {'FGSM':>10} {'PGD':>10} {'AutoAttack':>12}")
    log(f"  {'-' * 44}")

    for i, eps in enumerate(epsilons):
        if eps == 0:
            acc = clean_acc
        else:
            X_adv_auto, _ = auto_attack(model_std, X_test, y_test, eps)
            acc = evaluate(model_std, X_adv_auto, y_test)
        auto_accs.append(acc)
        log(f"  {eps:>8.2f} {std_accs[i]:>10.4f} {pgd_accs[i]:>10.4f} {acc:>12.4f}")

    # ═══════════════════════════════════════════════════════
    # Adversarial training
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("STEP 6: ADVERSARIAL TRAINING (defense)")
    log("=" * 60)

    model_adv = SimpleConvNet()
    train_adversarial(model_adv, X_train, y_train, epsilon=0.2, epochs=10)
    adv_clean_acc = evaluate(model_adv, X_test, y_test)
    log(f"\n  Clean accuracy (adv-trained): {adv_clean_acc:.4f}")

    adv_fgsm_accs = []
    adv_pgd_accs = []
    log(f"\n  Adversarially-trained model under each attack:")
    log(f"  {'Epsilon':>8} {'FGSM':>10} {'PGD':>10} {'Δ vs std FGSM':>14}")
    log(f"  {'-' * 46}")

    for i, eps in enumerate(epsilons):
        if eps == 0:
            fgsm_a = adv_clean_acc
            pgd_a = adv_clean_acc
        else:
            X_af, _ = fgsm_attack(model_adv, X_test, y_test, eps)
            fgsm_a = evaluate(model_adv, X_af, y_test)
            X_ap, _ = pgd_attack(model_adv, X_test, y_test, eps)
            pgd_a = evaluate(model_adv, X_ap, y_test)
        adv_fgsm_accs.append(fgsm_a)
        adv_pgd_accs.append(pgd_a)
        log(f"  {eps:>8.2f} {fgsm_a:>10.4f} {pgd_a:>10.4f} {fgsm_a - std_accs[i]:>+14.4f}")

    # ═══════════════════════════════════════════════════════
    # Visualize adversarial examples
    # ═══════════════════════════════════════════════════════
    X_adv_vis, perturbation = fgsm_attack(model_std, X_test[:8], y_test[:8], epsilon=0.2)

    model_std.eval()
    with torch.no_grad():
        clean_preds = model_std(X_test[:8]).argmax(dim=1)
        adv_preds = model_std(X_adv_vis).argmax(dim=1)

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 12))

    # Top: adversarial examples (clean → perturbation → adversarial)
    for i in range(6):
        ax = fig.add_subplot(5, 6, i + 1)
        ax.imshow(X_test[i, 0].numpy(), cmap="gray")
        ax.set_title(f"Clean: {clean_preds[i].item()}", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(5, 6, i + 7)
        ax.imshow(perturbation[i, 0].numpy(), cmap="RdBu", vmin=-0.3, vmax=0.3)
        ax.set_title("Perturbation", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(5, 6, i + 13)
        ax.imshow(X_adv_vis[i, 0].numpy(), cmap="gray")
        color = "red" if adv_preds[i] != y_test[i] else "green"
        ax.set_title(f"Adv: {adv_preds[i].item()}", fontsize=9, color=color)
        ax.axis("off")

    # Attack comparison on standard model
    ax = fig.add_subplot(5, 1, 4)
    ax.plot(epsilons, std_accs, "o-", label="FGSM", color="#e74c3c", linewidth=2)
    ax.plot(epsilons, pgd_accs, "s-", label="PGD", color="#e67e22", linewidth=2)
    ax.plot(epsilons, auto_accs, "^-", label="AutoAttack", color="#8e44ad", linewidth=2)
    ax.set_xlabel("Perturbation (ε)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Standard Model: Attack Comparison (stronger → lower)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Defense comparison under PGD
    ax = fig.add_subplot(5, 1, 5)
    ax.plot(epsilons, pgd_accs, "o-", label="Standard (PGD)", color="#e74c3c", linewidth=2)
    ax.plot(epsilons, adv_pgd_accs, "s-", label="Adv-trained (PGD)", color="#2ecc71", linewidth=2)
    ax.set_xlabel("Perturbation (ε)")
    ax.set_ylabel("Accuracy")
    ax.set_title("PGD Attack: Standard vs Adversarially Trained")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "adversarial_robustness.png"), dpi=150)
    plt.close()
    log(f"\n→ Plot saved: plots/adversarial_robustness.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
  ┌─────────────────────┬────────────────────────────────────┐
  │ Attack              │ How it works                       │
  ├─────────────────────┼────────────────────────────────────┤
  │ FGSM                │ One-step gradient sign attack.     │
  │                     │ Fast, but not the strongest.       │
  │ PGD                 │ Iterative FGSM. Stronger attack.   │
  │ C&W                 │ Optimization-based. L2 minimal.   │
  │ AutoAttack          │ Ensemble of attacks. Standard eval.│
  └─────────────────────┴────────────────────────────────────┘
  ┌─────────────────────┬────────────────────────────────────┐
  │ Defense             │ How it works                       │
  ├─────────────────────┼────────────────────────────────────┤
  │ Adversarial train   │ Augment training with adv examples.│
  │ Input preprocessing │ JPEG compression, smoothing.       │
  │ Certified defense   │ Provable robustness bounds.        │
  │ Randomized smooth   │ Average predictions over noise.    │
  └─────────────────────┴────────────────────────────────────┘

  Key insight: there's a tradeoff between clean accuracy and
  robustness. Adversarial training often reduces clean accuracy
  slightly but dramatically improves robustness.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
