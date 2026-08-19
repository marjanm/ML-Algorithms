"""
Reinforcement Learning — Q-Learning & DQN Demo
================================================
Demonstrates the third paradigm of ML (beyond supervised & unsupervised):
  1. Tabular Q-Learning on a simple grid world
  2. Deep Q-Network (DQN) on CartPole-v1

No external RL library needed — everything from scratch with NumPy + PyTorch.

Run:
    python rl_demo.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
import random

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# PART 1: Tabular Q-Learning on a Grid World
# ─────────────────────────────────────────────
#
# 4x4 grid.  Agent starts at (0,0), goal at (3,3).
# Actions: 0=up, 1=right, 2=down, 3=left
# Reward: -1 per step, +10 at goal, -5 for hitting a wall trap.

GRID_SIZE = 4
GOAL = (3, 3)
TRAPS = {(1, 1), (2, 3)}


def step_grid(state, action):
    """Take an action in the grid world, return (next_state, reward, done)."""
    r, c = state
    if action == 0:   r -= 1
    elif action == 1: c += 1
    elif action == 2: r += 1
    elif action == 3: c -= 1

    # stay in bounds
    r = max(0, min(GRID_SIZE - 1, r))
    c = max(0, min(GRID_SIZE - 1, c))
    next_state = (r, c)

    if next_state == GOAL:
        return next_state, 10.0, True
    if next_state in TRAPS:
        return next_state, -5.0, True
    return next_state, -1.0, False


def run_q_learning(
    n_episodes=500,       # total training episodes
    alpha=0.1,            # learning rate — how much new info overrides old Q values
    gamma=0.99,           # discount factor — importance of future rewards (0=myopic, 1=far-sighted)
    epsilon_start=1.0,    # initial exploration rate — probability of random action
    epsilon_end=0.01,     # minimum exploration rate after decay
    epsilon_decay=0.995,  # multiplicative decay per episode
    max_steps=50,         # max steps per episode to avoid infinite loops
):
    Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))
    epsilon = epsilon_start
    episode_rewards = []
    episode_steps = []

    for ep in range(n_episodes):
        state = (0, 0)
        total_reward = 0
        for t in range(max_steps):
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state[0], state[1]])

            next_state, reward, done = step_grid(state, action)
            # Q-learning update: Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
            best_next = np.max(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] += alpha * (
                reward + gamma * best_next - Q[state[0], state[1], action]
            )
            state = next_state
            total_reward += reward
            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(t + 1)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return Q, episode_rewards, episode_steps


def visualise_q_learning(Q, episode_rewards, lines):
    """Plot learned policy + reward curve."""
    action_symbols = ["↑", "→", "↓", "←"]

    lines.append("\n  Learned policy (best action per cell):")
    lines.append("  " + "-" * 25)
    for r in range(GRID_SIZE):
        row_str = "  |"
        for c in range(GRID_SIZE):
            if (r, c) == GOAL:
                row_str += "  G  |"
            elif (r, c) in TRAPS:
                row_str += "  X  |"
            else:
                best_a = np.argmax(Q[r, c])
                row_str += f"  {action_symbols[best_a]}  |"
        lines.append(row_str)
    lines.append("  " + "-" * 25)
    lines.append(f"  G = Goal (+10),  X = Trap (-5)")

    # smoothed reward curve
    window = 20
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_rewards, alpha=0.3, label="raw")
    ax.plot(range(window - 1, len(episode_rewards)), smoothed, label=f"{window}-ep avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Q-Learning — Episode Rewards (4×4 Grid World)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "q_learning_rewards.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] reward curve → plots/q_learning_rewards.png")

    # value heatmap
    V = np.max(Q, axis=2)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(V, cmap="RdYlGn")
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            label = f"{V[r, c]:.1f}"
            if (r, c) == GOAL: label = "G"
            if (r, c) in TRAPS: label = "X"
            ax.text(c, r, label, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title("State Value V(s) = max_a Q(s,a)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "q_learning_values.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] value heatmap → plots/q_learning_values.png")

    return lines


# ─────────────────────────────────────────────
# PART 2: Deep Q-Network (DQN) on CartPole
# ─────────────────────────────────────────────

def run_dqn_demo(lines):
    """DQN with experience replay and target network on CartPole-v1."""
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError:
            lines.append("\n  ⚠ gym/gymnasium not installed — skipping DQN demo.")
            lines.append("    Install with: pip3 install gymnasium")
            return lines

    import torch
    import torch.nn as nn
    import torch.optim as optim

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n             # 2

    # ── hyperparameters ──
    LR = 1e-3             # learning rate for the Adam optimiser
    GAMMA = 0.99           # discount factor
    EPSILON_START = 1.0    # initial exploration rate
    EPSILON_END = 0.01     # floor for exploration
    EPSILON_DECAY = 0.995  # multiplicative decay per episode
    BATCH_SIZE = 64        # mini-batch size sampled from replay buffer
    BUFFER_SIZE = 10000    # max transitions stored in replay memory
    TARGET_UPDATE = 10     # sync target network every N episodes
    N_EPISODES = 300       # total training episodes
    MAX_STEPS = 500        # max steps per episode

    class QNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )

        def forward(self, x):
            return self.net(x)

    device = torch.device("cpu")
    policy_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epsilon = EPSILON_START
    episode_rewards = []

    for ep in range(N_EPISODES):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.array(state, dtype=np.float32)
        total_reward = 0

        for t in range(MAX_STEPS):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state).unsqueeze(0))
                    action = q_vals.argmax(dim=1).item()

            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_state = np.array(next_state, dtype=np.float32)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # train on a mini-batch from replay buffer
            if len(replay_buffer) >= BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                states_b = torch.tensor(np.array([b[0] for b in batch]))
                actions_b = torch.tensor([b[1] for b in batch], dtype=torch.long)
                rewards_b = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                nexts_b = torch.tensor(np.array([b[3] for b in batch]))
                dones_b = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                current_q = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(nexts_b).max(dim=1)[0]
                    target_q = rewards_b + GAMMA * next_q * (1 - dones_b)

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (ep + 1) % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()

    # results
    window = 20
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
    last_50_avg = np.mean(episode_rewards[-50:])

    lines += [
        "", "=" * 60,
        "  DQN  —  CartPole-v1 Results",
        "=" * 60,
        f"  Episodes       : {N_EPISODES}",
        f"  LR             : {LR}",
        f"  Gamma          : {GAMMA}",
        f"  Batch size     : {BATCH_SIZE}",
        f"  Buffer size    : {BUFFER_SIZE}",
        f"  Target update  : every {TARGET_UPDATE} episodes",
        f"  Last-50 avg    : {last_50_avg:.1f} steps (goal: 500)",
        f"  Max achieved   : {max(episode_rewards):.0f} steps",
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_rewards, alpha=0.3, label="raw")
    ax.plot(range(window - 1, len(episode_rewards)), smoothed, label=f"{window}-ep avg", color="red")
    ax.axhline(y=475, color="green", linestyle="--", label="solved threshold (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward (steps survived)")
    ax.set_title("DQN — CartPole-v1 Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "dqn_cartpole.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] DQN reward curve → plots/dqn_cartpole.png")

    return lines


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    lines = [
        "=" * 60,
        "  REINFORCEMENT LEARNING  —  Demo",
        "=" * 60,
        "",
        "  Part 1: Tabular Q-Learning (4×4 Grid World)",
        "  Part 2: Deep Q-Network (CartPole-v1)",
        "",
    ]

    # Part 1
    lines.append("=" * 60)
    lines.append("  PART 1: Q-LEARNING  —  Grid World")
    lines.append("=" * 60)
    Q, rewards, steps = run_q_learning()
    last_50_avg = np.mean(rewards[-50:])
    lines.append(f"  Episodes trained : 500")
    lines.append(f"  Last-50 avg reward: {last_50_avg:.2f}")
    lines.append(f"  Final epsilon    : ~0.01")
    lines = visualise_q_learning(Q, rewards, lines)

    # Part 2
    lines = run_dqn_demo(lines)

    lines += ["", "=" * 60]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    main()
