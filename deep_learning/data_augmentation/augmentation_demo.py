"""
Data Augmentation — Demo
==========================
Critical for making CNNs work with small datasets.
Artificially increase training data by applying realistic transforms.

Demonstrates:
  1. Common image transforms (rotation, flip, crop, jitter, erasing)
  2. Visual: original vs augmented grid
  3. Impact: train CNN with vs without augmentation on small MNIST subset

Run:
    python augmentation_demo.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cnn", "data")


def show_augmentations():
    """Visualise different augmentation transforms on a single image."""
    base_transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=base_transform)
    img_tensor, label = dataset[3]  # a digit
    img_pil = T.ToPILImage()(img_tensor)

    augmentations = {
        "Original": T.ToTensor(),
        "Rotation (30°)": T.Compose([T.RandomRotation(30), T.ToTensor()]),
        "Horizontal Flip": T.Compose([T.RandomHorizontalFlip(p=1.0), T.ToTensor()]),
        "Random Crop": T.Compose([T.RandomCrop(28, padding=4), T.ToTensor()]),
        "Affine": T.Compose([T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)), T.ToTensor()]),
        "Perspective": T.Compose([T.RandomPerspective(distortion_scale=0.3, p=1.0), T.ToTensor()]),
        "Gaussian Blur": T.Compose([T.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)), T.ToTensor()]),
        "Random Erasing": T.Compose([T.ToTensor(), T.RandomErasing(p=1.0, scale=(0.1, 0.3))]),
    }

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, (name, transform) in zip(axes.flatten(), augmentations.items()):
        aug_img = transform(img_pil)
        ax.imshow(aug_img.squeeze(), cmap="gray")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    plt.suptitle(f"Data Augmentation Transforms (digit={label})", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "augmentation_examples.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return label


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(train_loader, test_loader, epochs=10):
    """Train CNN and return per-epoch accuracies."""
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
        train_accs.append(correct / total)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                out = model(X_batch)
                correct += (out.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        test_accs.append(correct / total)

    return train_accs, test_accs


def run_augmentation_demo():
    lines = [
        "=" * 65,
        "  DATA AUGMENTATION  —  Demo",
        "=" * 65, "",
    ]

    # Part 1: show transforms
    label = show_augmentations()
    lines.append(f"  [saved] augmentation examples → plots/augmentation_examples.png")

    # Part 2: train with vs without augmentation on small subset
    lines += ["", "  ── CNN Comparison: With vs Without Augmentation ──"]
    lines.append(f"  Using a small MNIST subset (1000 train samples) to amplify the effect.")

    transform_plain = T.Compose([T.ToTensor()])
    transform_aug = T.Compose([
        T.RandomRotation(15),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.05, 0.15)),
    ])

    # small training subset
    full_train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform_plain)
    test_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform_plain)

    rng = np.random.RandomState(42)
    subset_idx = rng.choice(len(full_train), 1000, replace=False).tolist()

    train_plain = Subset(full_train, subset_idx)

    full_train_aug = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform_aug)
    train_aug = Subset(full_train_aug, subset_idx)

    plain_loader = DataLoader(train_plain, batch_size=64, shuffle=True)
    aug_loader = DataLoader(train_aug, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    epochs = 15
    lines.append(f"  Training for {epochs} epochs...")

    torch.manual_seed(42)
    train_acc_plain, test_acc_plain = train_and_evaluate(plain_loader, test_loader, epochs)

    torch.manual_seed(42)
    train_acc_aug, test_acc_aug = train_and_evaluate(aug_loader, test_loader, epochs)

    lines += [
        f"",
        f"  {'':15s} | {'Train Acc':>10s} | {'Test Acc':>10s}",
        f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}",
        f"  {'No augmentation':15s} | {train_acc_plain[-1]:10.4f} | {test_acc_plain[-1]:10.4f}",
        f"  {'With augment.':15s} | {train_acc_aug[-1]:10.4f} | {test_acc_aug[-1]:10.4f}",
        f"",
        f"  Augmentation effect on test accuracy: {test_acc_aug[-1] - test_acc_plain[-1]:+.4f}",
    ]

    # learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep_range = range(1, epochs + 1)

    axes[0].plot(ep_range, train_acc_plain, label="No augmentation", marker=".")
    axes[0].plot(ep_range, train_acc_aug, label="With augmentation", marker=".")
    axes[0].set_title("Train Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep_range, test_acc_plain, label="No augmentation", marker=".")
    axes[1].plot(ep_range, test_acc_aug, label="With augmentation", marker=".")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Impact of Data Augmentation (1000 training samples)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "augmentation_impact.png"), dpi=150, bbox_inches="tight")
    plt.close()
    lines.append(f"  [saved] → plots/augmentation_impact.png")

    lines += [
        "", "  ── Key Takeaways ──",
        "    • Augmentation helps most when training data is scarce",
        "    • It acts as a regulariser — reduces overfitting (train acc lower, test acc higher)",
        "    • Common transforms: rotation, flip, crop, affine, color jitter, erasing",
        "    • Domain-specific: medical images may only allow rotation, not flipping",
        "    • Advanced: mixup, cutmix, AutoAugment (learned augmentation policies)",
        "", "=" * 65,
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_augmentation_demo()
