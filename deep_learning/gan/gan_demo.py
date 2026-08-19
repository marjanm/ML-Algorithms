"""
Generative Adversarial Network (GAN) — generate synthetic images
==================================================================
Trains a simple GAN on MNIST to generate handwritten digit images.
Two networks compete: Generator creates fakes, Discriminator detects them.

Run:
    python gan_demo.py
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


class Generator(nn.Module):
    """Maps random noise (latent vector) to a 28x28 image."""

    def __init__(
        self,
        latent_dim: int = 100,       # size of the random noise input — the "seed" for generating images
        hidden_dim: int = 256,       # neurons in hidden layers
        img_size: int = 784,         # 28*28 = 784 flattened pixels
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),                # leaky ReLU lets small negative values through (avoids "dead neurons")
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, img_size),
            nn.Tanh(),                        # output in [-1, 1] to match normalised images
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Classifies an image as real (1) or fake (0)."""

    def __init__(
        self,
        img_size: int = 784,
        hidden_dim: int = 256,
        dropout: float = 0.3,        # dropout in discriminator prevents it from becoming too strong too fast
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_size, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),                     # output probability: 1 = "I think this is real"
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(
    # --- architecture ---
    latent_dim: int = 100,           # dimension of the random noise vector fed to the generator
    g_hidden: int = 256,             # generator hidden size
    d_hidden: int = 256,             # discriminator hidden size
    d_dropout: float = 0.3,
    # --- training ---
    epochs: int = 50,
    batch_size: int = 128,
    lr_g: float = 0.0002,           # generator learning rate
    lr_d: float = 0.0002,           # discriminator learning rate
    beta1: float = 0.5,             # Adam beta1 — lower than default (0.9) for GAN stability
    # --- misc ---
    random_state: int = 42,
):
    torch.manual_seed(random_state)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # scale to [-1, 1]
    ])

    data_dir = os.path.join(OUTPUT_DIR, "..", "datasets")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    G = Generator(latent_dim=latent_dim, hidden_dim=g_hidden).to(device)
    D = Discriminator(hidden_dim=d_hidden, dropout=d_dropout).to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(beta1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, 0.999))

    g_losses, d_losses = [], []
    fixed_noise = torch.randn(64, latent_dim, device=device)   # fixed for consistent progress images

    for epoch in range(epochs):
        g_loss_sum, d_loss_sum, n_batches = 0, 0, 0

        for real_imgs, _ in train_loader:
            bs = real_imgs.size(0)
            real_imgs = real_imgs.view(bs, -1).to(device)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # --- Train Discriminator: maximise log(D(real)) + log(1 - D(G(z))) ---
            opt_D.zero_grad()
            d_real = D(real_imgs)
            loss_real = criterion(d_real, real_labels)

            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = G(z).detach()       # detach so gradients don't flow to G
            d_fake = D(fake_imgs)
            loss_fake = criterion(d_fake, fake_labels)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            opt_D.step()

            # --- Train Generator: maximise log(D(G(z))) ---
            opt_G.zero_grad()
            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = G(z)
            d_fake = D(fake_imgs)
            g_loss = criterion(d_fake, real_labels)   # generator wants D to say "real"
            g_loss.backward()
            opt_G.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            n_batches += 1

        g_losses.append(g_loss_sum / n_batches)
        d_losses.append(d_loss_sum / n_batches)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  D_loss={d_losses[-1]:.4f}  "
                  f"G_loss={g_losses[-1]:.4f}")

    lines = [
        "=" * 50,
        "  GAN  —  Results (MNIST generation)",
        "=" * 50,
        f"  Latent dim    : {latent_dim}",
        f"  G hidden      : {g_hidden}",
        f"  D hidden      : {d_hidden}",
        f"  Epochs        : {epochs}",
        f"  Batch size    : {batch_size}",
        f"  LR (G/D)      : {lr_g} / {lr_d}",
        f"  Device        : {device}",
        f"  Final G loss  : {g_losses[-1]:.4f}",
        f"  Final D loss  : {d_losses[-1]:.4f}",
        "=" * 50,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    # --- Plot: loss curves ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g_losses, label="Generator", linewidth=1.5)
    ax.plot(d_losses, label="Discriminator", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GAN — Training Losses")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "gan_losses.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

    # --- Plot: generated samples ---
    G.eval()
    with torch.no_grad():
        samples = G(fixed_noise).cpu().view(-1, 28, 28).numpy()
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(f"GAN — Generated Digits (epoch {epochs})", fontsize=12)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "gan_generated.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


if __name__ == "__main__":
    train_and_evaluate()
