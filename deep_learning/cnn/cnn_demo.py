"""
Convolutional Neural Network (CNN) — image classification
===========================================================
Trains a CNN on MNIST (handwritten digits 0-9) to show how
convolutions learn spatial features from images.

Run:
    python cnn_demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


class CNN(nn.Module):
    """Small CNN for 28x28 grayscale images (MNIST)."""

    def __init__(
        self,
        in_channels: int = 1,        # 1 = grayscale, 3 = RGB
        conv1_filters: int = 32,     # number of filters in first conv layer — each learns a different pattern (edges, curves)
        conv2_filters: int = 64,     # more filters in deeper layers to capture complex combinations
        kernel_size: int = 3,        # size of the sliding window (3x3 pixels) that scans the image
        pool_size: int = 2,          # max-pooling window — keeps the strongest activation in each 2x2 region, halves resolution
        fc_hidden: int = 128,        # neurons in the fully connected layer after flattening
        num_classes: int = 10,       # output classes (digits 0-9)
        dropout: float = 0.25,       # dropout rate to prevent overfitting
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv1_filters, kernel_size, padding=1),  # slide 3x3 filters across the image, padding keeps size
            nn.BatchNorm2d(conv1_filters),                                  # normalise per channel — stabilises training
            nn.ReLU(),
            nn.MaxPool2d(pool_size),                                        # 28x28 -> 14x14
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size, padding=1),
            nn.BatchNorm2d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),                                        # 14x14 -> 7x7
            nn.Dropout2d(dropout),                                          # drop random feature maps
        )
        flat_size = conv2_filters * 7 * 7
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # reshape 64x7x7 -> 3136
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),  # raw logits — softmax applied by CrossEntropyLoss
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_and_evaluate(
    # --- training ---
    epochs: int = 5,                 # number of full passes through training data
    batch_size: int = 128,           # images per gradient update
    learning_rate: float = 0.001,    # step size for Adam optimiser
    weight_decay: float = 1e-4,      # L2 regularisation
    # --- architecture ---
    conv1_filters: int = 32,
    conv2_filters: int = 64,
    kernel_size: int = 3,
    fc_hidden: int = 128,
    dropout: float = 0.25,
    # --- misc ---
    random_state: int = 42,
):
    torch.manual_seed(random_state)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean and std
    ])

    data_dir = os.path.join(OUTPUT_DIR, "..", "datasets")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = CNN(conv1_filters=conv1_filters, conv2_filters=conv2_filters,
                kernel_size=kernel_size, fc_hidden=fc_hidden, dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss()     # combines LogSoftmax + NLLLoss — standard for multi-class
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_losses.append(epoch_loss / len(train_ds))

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
        test_accs.append(correct / len(test_ds))
        print(f"  Epoch {epoch+1}/{epochs}  loss={train_losses[-1]:.4f}  "
              f"test_acc={test_accs[-1]:.4f}")

    lines = [
        "=" * 50,
        "  CNN  —  Results (MNIST)",
        "=" * 50,
        f"  Architecture : Conv({conv1_filters}) -> Conv({conv2_filters}) -> FC({fc_hidden}) -> 10",
        f"  Kernel size  : {kernel_size}x{kernel_size}",
        f"  Dropout      : {dropout}",
        f"  Epochs       : {epochs}",
        f"  Batch size   : {batch_size}",
        f"  Learning rate: {learning_rate}",
        f"  Device       : {device}",
        f"  Final acc    : {test_accs[-1]:.4f}",
        "=" * 50,
    ]
    for i, (l, a) in enumerate(zip(train_losses, test_accs)):
        lines.append(f"  Epoch {i+1}: loss={l:.4f}  acc={a:.4f}")

    output_text = "\n".join(lines)
    print("\n" + output_text)
    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    # --- Plot: loss & accuracy ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, epochs + 1), train_losses, "b-o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("CNN — Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(range(1, epochs + 1), test_accs, "g-s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("CNN — Test Accuracy")
    ax2.set_ylim(0.9, 1.0)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "cnn_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

    # --- Plot: sample predictions ---
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    with torch.no_grad():
        for i, ax in enumerate(axes.flat):
            img, label = test_ds[i]
            pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).item()
            ax.imshow(img.squeeze().cpu(), cmap="gray")
            color = "green" if pred == label else "red"
            ax.set_title(f"pred={pred} true={label}", color=color, fontsize=9)
            ax.axis("off")
    fig.suptitle("CNN — Sample Predictions", fontsize=12)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "cnn_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


if __name__ == "__main__":
    train_and_evaluate()
