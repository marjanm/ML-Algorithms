"""
Transformer from Scratch — understanding attention
=====================================================
Builds the core Transformer components by hand in PyTorch:
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Positional Encoding
  - Transformer Encoder Block
  - Full classifier

Trains on a simple sequence classification task (synthetic)
to show how attention learns which tokens matter.

Run:
    python transformer_from_scratch.py
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ════════════════════════════════════════
#  1. Positional Encoding
# ════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """Adds position information to token embeddings.
    Transformers have no notion of order — this fixes that by injecting
    sine/cosine signals at different frequencies for each position."""

    def __init__(
        self,
        d_model: int = 64,          # embedding dimension
        max_len: int = 100,         # maximum sequence length supported
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions get sine
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions get cosine
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ════════════════════════════════════════
#  2. Scaled Dot-Product Attention
# ════════════════════════════════════════
def scaled_dot_product_attention(Q, K, V, mask=None):
    """The core attention operation.
    score = softmax( Q @ K^T / sqrt(d_k) ) @ V

    Q (query)  = "what am I looking for?"
    K (key)    = "what do I contain?"
    V (value)  = "what information do I give if selected?"

    Each token queries all other tokens, gets back a weighted mix of their values.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


# ════════════════════════════════════════
#  3. Multi-Head Attention
# ════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    """Run attention multiple times in parallel with different learned projections.
    Each head can learn to attend to different things (syntax, semantics, position)."""

    def __init__(
        self,
        d_model: int = 64,          # total embedding dimension
        n_heads: int = 4,           # number of parallel attention heads — d_model must be divisible by n_heads
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads    # dimension per head
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)   # project input to queries
        self.W_k = nn.Linear(d_model, d_model)   # project input to keys
        self.W_v = nn.Linear(d_model, d_model)   # project input to values
        self.W_o = nn.Linear(d_model, d_model)   # project concatenated heads back
        self.dropout = nn.Dropout(dropout)

        self.attn_weights = None   # stored for visualisation

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        self.attn_weights = attn_w.detach()

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(out)


# ════════════════════════════════════════
#  4. Transformer Encoder Block
# ════════════════════════════════════════
class TransformerBlock(nn.Module):
    """One encoder block: Multi-Head Attention + Feed-Forward, with residual connections and LayerNorm."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,            # hidden size of the feed-forward network (usually 2-4x d_model)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)     # normalise after attention
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),                          # smooth activation, used in modern transformers
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)     # normalise after feed-forward

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)           # residual connection: add input back
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)             # another residual connection
        return x


# ════════════════════════════════════════
#  5. Full Transformer Classifier
# ════════════════════════════════════════
class TransformerClassifier(nn.Module):
    """Stack multiple transformer blocks for sequence classification."""

    def __init__(
        self,
        vocab_size: int = 50,        # number of distinct tokens
        d_model: int = 64,           # embedding dimension
        n_heads: int = 4,            # attention heads per block
        n_layers: int = 2,           # number of stacked transformer blocks
        d_ff: int = 128,             # feed-forward hidden size
        max_len: int = 20,           # max sequence length
        num_classes: int = 2,        # binary classification
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)     # convert token IDs to vectors
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)             # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.pos_encoding(x)          # add position info
        for block in self.blocks:
            x = block(x, mask)
        x = x.mean(dim=1)                 # global average pooling over sequence
        return self.classifier(x)

    def get_attention_weights(self):
        return [block.attention.attn_weights for block in self.blocks]


# ════════════════════════════════════════
#  Synthetic data: classify sequences by whether they contain a "signal" token
# ════════════════════════════════════════
def generate_data(n_samples=3000, seq_len=15, vocab_size=50, signal_token=42, random_state=42):
    """Binary classification: does the sequence contain the signal token?
    This tests whether the transformer can learn to attend to specific tokens."""
    np.random.seed(random_state)
    X = np.random.randint(1, vocab_size - 1, (n_samples, seq_len))
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples // 2):
        pos = np.random.randint(0, seq_len)
        X[i, pos] = signal_token
        y[i] = 1

    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]


def train_and_evaluate(
    # --- data ---
    n_samples: int = 3000,
    seq_len: int = 15,
    vocab_size: int = 50,
    # --- architecture ---
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    dropout: float = 0.1,
    # --- training ---
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    # --- misc ---
    random_state: int = 42,
):
    torch.manual_seed(random_state)

    X, y = generate_data(n_samples, seq_len, vocab_size, random_state=random_state)
    split = int(0.8 * len(X))
    X_train, X_test = torch.LongTensor(X[:split]), torch.LongTensor(X[split:])
    y_train, y_test = torch.LongTensor(y[:split]), torch.LongTensor(y[split:])

    model = TransformerClassifier(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, max_len=seq_len, dropout=dropout,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()
        test_accs.append(acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}  acc={acc:.4f}")

    # Get attention weights for a sample
    model.eval()
    with torch.no_grad():
        sample = X_test[:1]
        _ = model(sample)
        attn_weights = model.get_attention_weights()

    lines = [
        "=" * 60,
        "  TRANSFORMER (from scratch)  —  Results",
        "=" * 60,
        f"  d_model      : {d_model}",
        f"  n_heads      : {n_heads}",
        f"  n_layers     : {n_layers}",
        f"  d_ff         : {d_ff}",
        f"  vocab_size   : {vocab_size}",
        f"  seq_len      : {seq_len}",
        f"  Epochs       : {epochs}",
        f"  Final acc    : {test_accs[-1]:.4f}",
        "=" * 60,
        "",
        "How it works:",
        "  1. Token IDs are converted to dense vectors (embeddings)",
        "  2. Positional encoding adds position info (sine/cosine signals)",
        "  3. Each transformer block applies self-attention: every token",
        "     looks at every other token and decides what to pay attention to",
        "  4. Multi-head attention runs this in parallel with different",
        "     learned projections (each head can focus on different patterns)",
        "  5. Feed-forward network processes each position independently",
        "  6. Residual connections + LayerNorm keep gradients flowing",
        "  7. Final: average pool all token representations -> classify",
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    out_path = os.path.join(OUTPUT_DIR, "output_from_scratch.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    # --- Plot: training ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Transformer — Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(test_accs, linewidth=1.5, color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Transformer — Test Accuracy")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "transformer_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

    # --- Plot: attention heatmap ---
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    layer0_attn = attn_weights[0][0]   # first sample, first layer
    for h, ax in enumerate(axes):
        ax.imshow(layer0_attn[h].cpu().numpy(), cmap="viridis", aspect="auto")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.set_title(f"Head {h+1}", fontsize=10)
    fig.suptitle("Self-Attention Weights (Layer 1) — each head attends differently", fontsize=11)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "transformer_attention_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")

    # --- Plot: positional encoding visualisation ---
    pe = PositionalEncoding(d_model=64, max_len=50, dropout=0.0)
    pe_matrix = pe.pe[0, :50, :].numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pe_matrix.T, cmap="RdBu", aspect="auto")
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Embedding dimension")
    ax.set_title("Positional Encoding — sine/cosine patterns encode position")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "positional_encoding.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


if __name__ == "__main__":
    train_and_evaluate()
