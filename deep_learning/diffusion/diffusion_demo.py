"""
Diffusion Model — denoising-based image generation
=====================================================
Implements a simplified Denoising Diffusion Probabilistic Model (DDPM)
on MNIST. The model learns to reverse a noise-adding process step by step.

This is the same family of models behind Stable Diffusion, DALL-E 2, and Imagen.

Run:
    python diffusion_demo.py
"""

import os
import math
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


class SinusoidalPosEmb(nn.Module):
    """Encode the diffusion timestep as a vector using sine/cosine frequencies."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SimpleUNet(nn.Module):
    """Tiny U-Net-style network that predicts noise from a noisy image + timestep."""

    def __init__(
        self,
        img_size: int = 784,         # 28*28 flattened
        hidden_dim: int = 256,       # width of hidden layers
        time_dim: int = 64,          # dimension of the timestep embedding
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),       # encode timestep as a vector
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(img_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, img_size),  # predict the noise that was added
        )
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.net[0](x)          # first linear
        h = self.net[1](h)          # GELU
        h = h + self.time_proj(t_emb)    # inject timestep info
        h = self.net[2](h)
        h = self.net[3](h)
        return self.net[4](h)


class DiffusionSchedule:
    """Linear beta schedule for the forward (noise-adding) process."""

    def __init__(
        self,
        timesteps: int = 300,        # total diffusion steps — more = finer noise schedule but slower
        beta_start: float = 1e-4,    # noise level at t=0 (almost no noise)
        beta_end: float = 0.02,      # noise level at t=T (almost pure noise)
    ):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)      # cumulative product — total signal remaining at step t
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def add_noise(self, x0, t, noise):
        """Forward process: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise"""
        s1 = self.sqrt_alpha_cumprod[t].view(-1, 1).to(x0.device)
        s2 = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1).to(x0.device)
        return s1 * x0 + s2 * noise


def train_and_evaluate(
    # --- schedule ---
    timesteps: int = 300,            # total diffusion steps
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    # --- architecture ---
    hidden_dim: int = 256,
    time_dim: int = 64,
    # --- training ---
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    # --- sampling ---
    n_samples: int = 64,            # images to generate after training
    # --- misc ---
    random_state: int = 42,
):
    torch.manual_seed(random_state)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    data_dir = os.path.join(OUTPUT_DIR, "..", "datasets")
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    schedule = DiffusionSchedule(timesteps, beta_start, beta_end)
    model = SimpleUNet(hidden_dim=hidden_dim, time_dim=time_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, _ in train_loader:
            bs = imgs.size(0)
            imgs = imgs.view(bs, -1).to(device)

            t = torch.randint(0, timesteps, (bs,))
            noise = torch.randn_like(imgs)
            x_noisy = schedule.add_noise(imgs, t, noise)

            predicted_noise = model(x_noisy, t.to(device))
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")

    # --- Sampling (reverse process) ---
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, 784, device=device)
        for t in reversed(range(timesteps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            alpha = schedule.alphas[t]
            alpha_bar = schedule.alpha_cumprod[t]
            beta = schedule.betas[t]

            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean

        samples = x.cpu().view(-1, 28, 28).numpy()

    lines = [
        "=" * 50,
        "  DIFFUSION MODEL (DDPM)  —  Results",
        "=" * 50,
        f"  Timesteps     : {timesteps}",
        f"  Hidden dim    : {hidden_dim}",
        f"  Time emb dim  : {time_dim}",
        f"  Epochs        : {epochs}",
        f"  Batch size    : {batch_size}",
        f"  Learning rate : {learning_rate}",
        f"  Device        : {device}",
        f"  Final loss    : {losses[-1]:.6f}",
        "=" * 50,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    # --- Plot: loss ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, epochs + 1), losses, "b-o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Diffusion — Training Loss")
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "diffusion_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

    # --- Plot: generated samples ---
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(f"Diffusion — Generated Samples (epoch {epochs})", fontsize=12)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "diffusion_generated.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

    # --- Plot: forward diffusion process visualisation ---
    sample_img = train_ds[0][0].view(1, -1)
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    steps_to_show = [0, 50, 100, 150, 200, 299]
    for ax, step in zip(axes, steps_to_show):
        noise = torch.randn_like(sample_img)
        noisy = schedule.add_noise(sample_img, torch.tensor([step]), noise)
        ax.imshow(noisy.view(28, 28).numpy(), cmap="gray")
        ax.set_title(f"t={step}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Forward Diffusion Process — Adding Noise Over Time", fontsize=11)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "diffusion_forward_process.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


if __name__ == "__main__":
    train_and_evaluate()
