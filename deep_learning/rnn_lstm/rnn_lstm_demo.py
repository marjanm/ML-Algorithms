"""
RNN / LSTM — sequence modelling
=================================
Trains an LSTM on a synthetic sine wave to predict future values,
demonstrating how recurrent networks handle sequential/temporal data.

Run:
    python rnn_lstm_demo.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


class LSTMPredictor(nn.Module):
    """LSTM for time-series forecasting."""

    def __init__(
        self,
        input_size: int = 1,          # features per time step (1 for univariate)
        hidden_size: int = 64,        # number of LSTM units — the "memory capacity" of each layer
        num_layers: int = 2,          # stacked LSTM layers — deeper = can learn more abstract temporal patterns
        dropout: float = 0.2,         # dropout between LSTM layers (not applied to last layer)
        bidirectional: bool = False,  # if True, reads sequence forwards AND backwards (useful for NLP, not forecasting)
        output_size: int = 1,         # predicting one value (next time step)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,        # features per time step
            hidden_size=hidden_size,      # size of the hidden state (memory vector)
            num_layers=num_layers,        # stack multiple LSTM layers
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,             # input shape: (batch, seq_len, features)
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_step = lstm_out[:, -1, :]    # take output from the last time step
        return self.fc(last_step)


def create_sequences(data, seq_length):
    """Slide a window of seq_length across the data to create (input, target) pairs."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


def train_and_evaluate(
    # --- data ---
    n_points: int = 1000,            # total data points in the sine wave
    seq_length: int = 30,            # number of past time steps fed to the model
    train_ratio: float = 0.8,        # fraction used for training
    # --- architecture ---
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    # --- training ---
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    # --- misc ---
    random_state: int = 42,
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    t = np.linspace(0, 4 * np.pi, n_points)
    data = np.sin(t) + 0.1 * np.random.randn(n_points)
    data = data.astype(np.float32)

    X, y = create_sequences(data, seq_length)
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)   # (batch, seq, 1)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(-1)
    X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(-1)

    model = LSTMPredictor(hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, bidirectional=bidirectional)

    criterion = nn.MSELoss()          # mean squared error — standard for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={epoch_loss:.6f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).squeeze().numpy()
    y_true = y_test

    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    lines = [
        "=" * 50,
        "  LSTM  —  Results (sine wave forecasting)",
        "=" * 50,
        f"  Architecture  : LSTM({hidden_size}) x {num_layers} layers",
        f"  Bidirectional : {bidirectional}",
        f"  Seq length    : {seq_length}",
        f"  Dropout       : {dropout}",
        f"  Epochs        : {epochs}",
        f"  MSE           : {mse:.6f}",
        f"  MAE           : {mae:.6f}",
        "=" * 50,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    # --- Plot: loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title("LSTM — Training Loss")
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "lstm_loss.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

    # --- Plot: predictions vs actual ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Actual", linewidth=1.5, alpha=0.8)
    ax.plot(y_pred, label="LSTM prediction", linewidth=1.5, alpha=0.8, linestyle="--")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title(f"LSTM — Sine Wave Prediction (MSE={mse:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "lstm_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


if __name__ == "__main__":
    train_and_evaluate()
