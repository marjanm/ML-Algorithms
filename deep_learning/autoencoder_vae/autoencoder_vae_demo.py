"""
Autoencoder & VAE — learn compressed representations
=======================================================
Trains both a vanilla Autoencoder and a Variational Autoencoder (VAE)
on MNIST. The encoder compresses 784 pixels into a small latent vector,
the decoder reconstructs the image from that vector.

Run:
    python autoencoder_vae_demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ════════════════════════════════════════
#  Vanilla Autoencoder
# ════════════════════════════════════════
class Autoencoder(nn.Module):
    """Compress input to a bottleneck, then reconstruct."""

    def __init__(
        self,
        input_dim: int = 784,        # 28*28 flattened
        hidden_dim: int = 256,       # intermediate layer size
        latent_dim: int = 16,        # bottleneck size — how many numbers represent the image; smaller = more compression
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),   # compress to latent_dim numbers
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),    # expand back
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),                         # output in [0, 1] to match pixel values
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ════════════════════════════════════════
#  Variational Autoencoder (VAE)
# ════════════════════════════════════════
class VAE(nn.Module):
    """Like an autoencoder, but the latent space is a probability distribution.
    This lets you sample new images by sampling from that distribution."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        latent_dim: int = 16,        # dimension of the latent Gaussian
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # mean of the latent Gaussian
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # log-variance of the latent Gaussian

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Sample z = mu + eps * std.  The 'reparameterization trick' lets gradients flow through the sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = reconstruction loss + KL divergence.
    KL divergence pushes the learned distribution toward a standard normal N(0,1)."""
    recon = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl


def train_and_evaluate(
    # --- architecture ---
    hidden_dim: int = 256,
    latent_dim: int = 16,
    # --- training ---
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    # --- misc ---
    random_state: int = 42,
):
    torch.manual_seed(random_state)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    data_dir = os.path.join(OUTPUT_DIR, "..", "datasets")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    output_lines = []

    for model_name, ModelClass in [("Autoencoder", Autoencoder), ("VAE", VAE)]:
        print(f"\n  Training {model_name} …")
        model = ModelClass(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for imgs, _ in train_loader:
                bs = imgs.size(0)
                imgs = imgs.view(bs, -1).to(device)
                optimizer.zero_grad()

                if model_name == "VAE":
                    recon, mu, logvar = model(imgs)
                    loss = vae_loss(recon, imgs, mu, logvar)
                else:
                    recon, z = model(imgs)
                    loss = F.binary_cross_entropy(recon, imgs, reduction="sum")

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_ds)
            train_losses.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

        lines = [
            "=" * 50,
            f"  {model_name.upper()}  —  Results",
            "=" * 50,
            f"  Hidden dim   : {hidden_dim}",
            f"  Latent dim   : {latent_dim}",
            f"  Epochs       : {epochs}",
            f"  Final loss   : {train_losses[-1]:.4f}",
            "=" * 50,
        ]
        output_lines.extend(lines)

        # --- Plot: loss ---
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, epochs + 1), train_losses, "b-o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model_name} — Training Loss")
        ax.grid(alpha=0.3)
        path = os.path.join(PLOT_DIR, f"{model_name.lower()}_loss.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {path}")

        # --- Plot: reconstruction ---
        model.eval()
        with torch.no_grad():
            test_imgs, _ = next(iter(test_loader))
            test_imgs_flat = test_imgs.view(test_imgs.size(0), -1).to(device)
            if model_name == "VAE":
                recon, _, _ = model(test_imgs_flat)
            else:
                recon, _ = model(test_imgs_flat)
            recon = recon.cpu().view(-1, 28, 28)

        fig, axes = plt.subplots(2, 8, figsize=(14, 4))
        for i in range(8):
            axes[0, i].imshow(test_imgs[i].squeeze(), cmap="gray")
            axes[0, i].axis("off")
            axes[0, i].set_title("Original" if i == 0 else "", fontsize=8)
            axes[1, i].imshow(recon[i].squeeze(), cmap="gray")
            axes[1, i].axis("off")
            axes[1, i].set_title("Reconstructed" if i == 0 else "", fontsize=8)
        fig.suptitle(f"{model_name} — Reconstruction", fontsize=12)
        fig.tight_layout()
        path = os.path.join(PLOT_DIR, f"{model_name.lower()}_reconstruction.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {path}")

        # --- Plot: generated samples (VAE only) ---
        if model_name == "VAE":
            with torch.no_grad():
                z = torch.randn(32, latent_dim, device=device)
                generated = model.decoder(z).cpu().view(-1, 28, 28).numpy()

            fig, axes = plt.subplots(2, 8, figsize=(14, 4))
            for i, ax in enumerate(axes.flat):
                if i < len(generated):
                    ax.imshow(generated[i], cmap="gray")
                ax.axis("off")
            fig.suptitle("VAE — Generated Samples (sampled from latent space)", fontsize=12)
            fig.tight_layout()
            path = os.path.join(PLOT_DIR, "vae_generated.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [saved] {path}")

    output_text = "\n".join(output_lines)
    print("\n" + output_text)
    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    train_and_evaluate()
