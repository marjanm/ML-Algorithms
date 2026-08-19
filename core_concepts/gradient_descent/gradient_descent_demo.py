"""
Gradient Descent — visual demo
================================
How models learn: walk downhill on the loss surface to find the best weights.

This demo:
  1. Creates a 2D loss surface (bowl-shaped)
  2. Runs gradient descent with THREE different learning rates
  3. Plots the paths on a contour map so you can see:
       - lr too small  → creeps slowly, may not converge in time
       - lr just right → smooth descent to the minimum
       - lr too large  → overshoots, bounces around, may diverge

Run:
    python gradient_descent_demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def loss_fn(w1, w2):
    """Quadratic loss surface with a minimum at OPTIMUM."""
    return 0.5 * (w1 - OPTIMUM[0]) ** 2 + 2.0 * (w2 - OPTIMUM[1]) ** 2


OPTIMUM = np.array([3.0, 2.0])


def grad_fn(w1, w2):
    """Gradient of the loss function."""
    return np.array([w1 - OPTIMUM[0], 4.0 * (w2 - OPTIMUM[1])])


def run_gradient_descent(
    start: tuple = (-4.0, -3.0),
    learning_rates: list = [0.01, 0.15, 0.55],
    n_steps: int = 50,
):
    all_lines = [
        "=" * 60, "  GRADIENT DESCENT  —  Learning Rate Comparison", "=" * 60,
        f"  Loss function : 0.5*(w1-{OPTIMUM[0]})² + 2*(w2-{OPTIMUM[1]})²",
        f"  Minimum at    : ({OPTIMUM[0]}, {OPTIMUM[1]})",
        f"  Start point   : {start}",
        f"  Steps         : {n_steps}", "",
    ]

    paths = {}
    for lr in learning_rates:
        w = np.array(start, dtype=float)
        trajectory = [w.copy()]
        for _ in range(n_steps):
            g = grad_fn(w[0], w[1])
            w = w - lr * g
            trajectory.append(w.copy())
        trajectory = np.array(trajectory)
        paths[lr] = trajectory
        final_loss = loss_fn(trajectory[-1, 0], trajectory[-1, 1])
        dist_to_min = np.linalg.norm(trajectory[-1] - OPTIMUM)
        all_lines += [
            f"  lr = {lr}:",
            f"    Final position : ({trajectory[-1, 0]:.4f}, {trajectory[-1, 1]:.4f})",
            f"    Final loss     : {final_loss:.6f}",
            f"    Distance to min: {dist_to_min:.6f}",
            "",
        ]

    all_lines += [
        "  Takeaway:",
        "    - lr too small (0.01) → barely moves in 50 steps",
        "    - lr just right (0.15) → smooth convergence",
        "    - lr too large (0.55) → overshoots, zigzags",
        "=" * 60,
    ]
    output_text = "\n".join(all_lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plot ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    w1 = np.linspace(-6, 8, 200)
    w2 = np.linspace(-5, 7, 200)
    W1, W2 = np.meshgrid(w1, w2)
    Z = loss_fn(W1, W2)

    colors = ["tab:blue", "tab:green", "tab:red"]
    labels = ["Too small", "Just right", "Too large"]

    for ax, lr, col, label in zip(axes, learning_rates, colors, labels):
        ax.contour(W1, W2, Z, levels=30, cmap="coolwarm", alpha=0.6)
        traj = paths[lr]
        ax.plot(traj[:, 0], traj[:, 1], "o-", color=col, markersize=3, linewidth=1.5)
        ax.plot(traj[0, 0], traj[0, 1], "ko", markersize=8, label="Start")
        ax.plot(*OPTIMUM, "r*", markersize=15, label="Minimum")
        ax.set_title(f"lr = {lr}  ({label})", fontsize=12)
        ax.set_xlabel("w₁")
        ax.set_ylabel("w₂")
        ax.legend(fontsize=8)
        ax.set_xlim(-6, 8)
        ax.set_ylim(-5, 7)

    plt.suptitle("Gradient Descent — Effect of Learning Rate", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "gradient_descent.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # loss-over-steps plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for lr, col, label in zip(learning_rates, colors, labels):
        traj = paths[lr]
        losses = [loss_fn(t[0], t[1]) for t in traj]
        ax.plot(losses, color=col, label=f"lr={lr} ({label})", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over gradient descent steps")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "loss_curves.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_gradient_descent()
