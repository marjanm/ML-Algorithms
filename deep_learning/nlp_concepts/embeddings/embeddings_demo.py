"""
Embeddings — visual demo
==========================
How words/tokens become numerical vectors that capture meaning.

This demo:
  1. Extracts word embeddings from a pre-trained model (BERT)
  2. Computes cosine similarity between word pairs
  3. Shows "king - man + woman ≈ queen" style arithmetic
  4. Visualises word clusters in 2D using t-SNE

Run:
    python embeddings_demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

WORD_GROUPS = {
    "Animals": ["cat", "dog", "fish", "bird", "horse", "lion", "tiger"],
    "Countries": ["france", "germany", "japan", "canada", "brazil", "india"],
    "Tech": ["computer", "software", "algorithm", "database", "network", "code"],
    "Food": ["pizza", "pasta", "sushi", "burger", "salad", "rice", "bread"],
    "Emotions": ["happy", "sad", "angry", "excited", "fearful", "calm"],
}


def get_word_embedding(word, tokenizer, model):
    """Get the embedding for a single word from BERT."""
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def run_embeddings_demo():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    all_words = []
    all_labels = []
    all_embeddings = []

    for group, words in WORD_GROUPS.items():
        for word in words:
            emb = get_word_embedding(word, tokenizer, model)
            all_words.append(word)
            all_labels.append(group)
            all_embeddings.append(emb)

    embeddings = np.array(all_embeddings)
    lines = [
        "=" * 70, "  EMBEDDINGS  —  Demo", "=" * 70,
        f"  Model       : bert-base-uncased",
        f"  Embedding dim: {embeddings.shape[1]}",
        f"  Words loaded : {len(all_words)}", "",
    ]

    # similarity pairs
    pairs = [
        ("cat", "dog"), ("cat", "computer"), ("happy", "sad"),
        ("happy", "excited"), ("france", "germany"), ("france", "pizza"),
        ("software", "code"), ("software", "salad"),
    ]
    lines.append("  Cosine similarity between word pairs:")
    for w1, w2 in pairs:
        e1 = all_embeddings[all_words.index(w1)]
        e2 = all_embeddings[all_words.index(w2)]
        sim = cosine_similarity([e1], [e2])[0][0]
        lines.append(f"    {w1:12s} ↔ {w2:12s} : {sim:.4f}")

    # word arithmetic
    lines += ["", "  Word arithmetic (king − man + woman ≈ ?):"]
    arithmetic_words = ["king", "man", "woman", "queen", "prince", "princess"]
    arith_embs = {w: get_word_embedding(w, tokenizer, model) for w in arithmetic_words}

    result_vec = arith_embs["king"] - arith_embs["man"] + arith_embs["woman"]
    for w, emb in arith_embs.items():
        sim = cosine_similarity([result_vec], [emb])[0][0]
        lines.append(f"    king − man + woman ↔ {w:10s} : {sim:.4f}")

    lines += [
        "", "  Key concepts:",
        "    - Each word is a 768-dimensional vector",
        "    - Similar words have similar vectors (high cosine similarity)",
        "    - Semantic relationships are captured as vector directions",
        "    - Embeddings are the input to all transformer-based models",
        "=" * 70,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- t-SNE visualisation ---
    tsne = TSNE(n_components=2, perplexity=8, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(WORD_GROUPS)))
    for i, (group, words) in enumerate(WORD_GROUPS.items()):
        mask = [all_labels[j] == group for j in range(len(all_labels))]
        idxs = [j for j, m in enumerate(mask) if m]
        plt.scatter(coords[idxs, 0], coords[idxs, 1], color=colors[i], s=100, label=group, edgecolors="black")
        for j in idxs:
            plt.annotate(all_words[j], (coords[j, 0] + 0.5, coords[j, 1] + 0.5), fontsize=8)

    plt.title("Word Embeddings (BERT) projected to 2D via t-SNE\nSimilar words cluster together")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "embeddings_tsne.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")

    # --- similarity heatmap ---
    sample_words = ["cat", "dog", "computer", "code", "happy", "sad", "france", "pizza"]
    sample_embs = np.array([all_embeddings[all_words.index(w)] for w in sample_words])
    sim_matrix = cosine_similarity(sample_embs)
    plt.figure(figsize=(8, 7))
    plt.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    plt.xticks(range(len(sample_words)), sample_words, rotation=45, ha="right")
    plt.yticks(range(len(sample_words)), sample_words)
    for i in range(len(sample_words)):
        for j in range(len(sample_words)):
            plt.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(label="Cosine similarity")
    plt.title("Pairwise similarity heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "similarity_heatmap.png"), dpi=150)
    plt.close()
    print(f"  [saved] heatmap → {PLOT_DIR}")


if __name__ == "__main__":
    run_embeddings_demo()
