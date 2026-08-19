"""
t-SNE & UMAP — fully parameterised
=====================================
Non-linear dimensionality reduction for visualisation.
Projects high-dimensional data to 2D so you can see clusters with your eyes.

- t-SNE: preserves local structure (nearby points stay nearby)
- UMAP: preserves both local and global structure, and is much faster

Run:
    python tsne_umap_model.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

HAS_UMAP = False
try:
    import umap
    HAS_UMAP = True
except ImportError:
    pass


def run_tsne_umap(
    # t-SNE params
    tsne_perplexity: float = 30.0,      # effective number of nearest neighbours; 5-50 typical
    tsne_learning_rate: float = 200.0,  # step size; "auto" or float, 10-1000
    tsne_n_iter: int = 1000,            # max optimisation iterations
    tsne_early_exaggeration: float = 12.0,  # how tightly clusters form early on
    tsne_metric: str = "euclidean",     # distance metric
    tsne_init: str = "pca",             # initialisation: "pca" (deterministic) or "random"
    tsne_method: str = "barnes_hut",    # "barnes_hut" (O(N log N)) or "exact" (O(N²))
    tsne_angle: float = 0.5,            # trade-off speed vs accuracy for barnes_hut; lower = more accurate
    # UMAP params (only used if umap is installed)
    umap_n_neighbors: int = 15,         # size of local neighbourhood; larger = more global structure
    umap_min_dist: float = 0.1,         # min distance between embedded points; lower = tighter clusters
    umap_n_components: int = 2,
    umap_metric: str = "euclidean",
    umap_spread: float = 1.0,           # scale of embedded points
    random_state: int = 42,
):
    digits = load_digits()
    X, y = digits.data, digits.target

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    all_lines = ["=" * 60, "  t-SNE / UMAP  —  Results", "=" * 60, ""]

    n_plots = 2 if HAS_UMAP else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    # ---- t-SNE ----
    tsne = TSNE(
        n_components=2, perplexity=tsne_perplexity, learning_rate=tsne_learning_rate,
        n_iter=tsne_n_iter, early_exaggeration=tsne_early_exaggeration,
        metric=tsne_metric, init=tsne_init, method=tsne_method,
        angle=tsne_angle, random_state=random_state,
    )
    start = time.perf_counter()
    X_tsne = tsne.fit_transform(X_s)
    t_tsne = time.perf_counter() - start
    all_lines += [
        "  t-SNE:",
        f"    perplexity        : {tsne_perplexity}",
        f"    n_iter            : {tsne_n_iter}",
        f"    KL divergence     : {tsne.kl_divergence_:.4f}",
        f"    Time              : {t_tsne:.2f}s", "",
    ]
    scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", s=8, alpha=0.6)
    axes[0].set_title(f"t-SNE (perplexity={tsne_perplexity})")
    plt.colorbar(scatter, ax=axes[0], label="digit")

    # ---- UMAP ----
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
            n_components=umap_n_components, metric=umap_metric,
            spread=umap_spread, random_state=random_state,
        )
        start = time.perf_counter()
        X_umap = reducer.fit_transform(X_s)
        t_umap = time.perf_counter() - start
        all_lines += [
            "  UMAP:",
            f"    n_neighbors : {umap_n_neighbors}",
            f"    min_dist    : {umap_min_dist}",
            f"    Time        : {t_umap:.2f}s", "",
        ]
        scatter = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="tab10", s=8, alpha=0.6)
        axes[1].set_title(f"UMAP (n_neighbors={umap_n_neighbors})")
        plt.colorbar(scatter, ax=axes[1], label="digit")
    else:
        all_lines.append("  UMAP: not installed — pip install umap-learn")

    all_lines.append("=" * 60)
    output_text = "\n".join(all_lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "tsne_umap.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_tsne_umap()
