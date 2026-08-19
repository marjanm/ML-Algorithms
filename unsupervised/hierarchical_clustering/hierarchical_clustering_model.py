"""
Hierarchical (Agglomerative) Clustering — fully parameterised
===============================================================
Builds a tree of clusters from the bottom up. You can cut the tree
at any level to get different numbers of clusters.
Also produces a dendrogram to visualise cluster merging order.

Run:
    python hierarchical_clustering_model.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_hierarchical(
    n_clusters: int = 4,                # target number of clusters (set None to use distance_threshold)
    linkage_method: str = "ward",       # "ward" (min variance), "complete" (max dist), "average", "single" (min dist)
    metric: str = "euclidean",          # distance metric (only euclidean works with ward)
    distance_threshold: float = None,   # distance threshold to cut the tree; if set, n_clusters must be None
    compute_full_tree: str = "auto",    # whether to compute full tree even if n_clusters is set
    compute_distances: bool = True,     # store inter-cluster distances (needed for dendrogram)
    random_state: int = 42,
):
    X, y_true = make_blobs(
        n_samples=400, centers=4, cluster_std=[1.0, 1.5, 0.5, 1.2],
        random_state=random_state,
    )
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage_method, metric=metric,
        distance_threshold=distance_threshold,
        compute_full_tree=compute_full_tree,
        compute_distances=compute_distances,
    )

    start = time.perf_counter()
    labels = model.fit_predict(X_s)
    t = time.perf_counter() - start

    n_found = len(set(labels))
    sil = silhouette_score(X_s, labels) if n_found > 1 else float("nan")

    lines = [
        "=" * 60, "  HIERARCHICAL CLUSTERING  —  Results", "=" * 60,
        f"  Linkage      : {linkage_method}",
        f"  n_clusters   : {n_clusters}",
        f"  Found        : {n_found} clusters",
        f"  Silhouette   : {sil:.4f}" if not np.isnan(sil) else "  Silhouette   : N/A",
        f"  Time         : {t:.4f}s",
        "=" * 60,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # cluster scatter
    for k in sorted(set(labels)):
        mask = labels == k
        axes[0].scatter(X_s[mask, 0], X_s[mask, 1], s=15, alpha=0.7, label=f"Cluster {k}")
    axes[0].set_title(f"Agglomerative Clustering ({linkage_method})")
    axes[0].legend(fontsize=8)

    # dendrogram (using scipy linkage for the full tree)
    Z = linkage(X_s, method=linkage_method)
    dendrogram(Z, ax=axes[1], truncate_mode="lastp", p=30, leaf_rotation=90,
               leaf_font_size=8, color_threshold=0)
    axes[1].set_title("Dendrogram (last 30 merges)")
    axes[1].set_xlabel("Sample index / cluster size")
    axes[1].set_ylabel("Distance")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hierarchical_clustering.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_hierarchical()
