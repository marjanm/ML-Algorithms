"""
DBSCAN — fully parameterised
==============================
Density-Based Spatial Clustering of Applications with Noise.
Finds clusters of arbitrary shape and labels outliers as noise (-1).
Unlike K-Means, doesn't need the number of clusters in advance.

Run:
    python dbscan_model.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_dbscan(
    eps: float = 0.3,                   # max distance between two samples to be neighbours
    min_samples: int = 5,               # min points in a neighbourhood to form a core point
    metric: str = "euclidean",          # distance metric: "euclidean", "manhattan", "cosine", etc.
    algorithm: str = "auto",            # nearest-neighbour algorithm: "auto", "ball_tree", "kd_tree", "brute"
    leaf_size: int = 30,                # leaf size for ball_tree/kd_tree (affects speed & memory)
    p: float = None,                    # power for Minkowski metric (2 = euclidean, 1 = manhattan)
    n_jobs: int = -1,
    random_state: int = 42,
):
    # two datasets to show DBSCAN's strength vs K-Means
    datasets = {
        "Moons": make_moons(n_samples=500, noise=0.1, random_state=random_state),
        "Blobs": make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=random_state),
    }

    all_lines = ["=" * 60, "  DBSCAN  —  Results", "=" * 60, ""]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (name, (X, y_true)) in zip(axes, datasets.items()):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        model = DBSCAN(
            eps=eps, min_samples=min_samples, metric=metric,
            algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs,
        )

        start = time.perf_counter()
        labels = model.fit_predict(X_s)
        t = time.perf_counter() - start

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        sil = silhouette_score(X_s, labels) if n_clusters > 1 else float("nan")

        all_lines += [
            f"  Dataset     : {name}",
            f"  eps         : {eps}",
            f"  min_samples : {min_samples}",
            f"  Clusters    : {n_clusters}",
            f"  Noise pts   : {n_noise}",
            f"  Silhouette  : {sil:.4f}" if not np.isnan(sil) else "  Silhouette  : N/A",
            f"  Time        : {t:.4f}s", "",
        ]

        unique = set(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
        for k, col in zip(sorted(unique), colors):
            if k == -1:
                col = "grey"
            mask = labels == k
            ax.scatter(X_s[mask, 0], X_s[mask, 1], c=[col], s=10, alpha=0.7,
                       label=f"{'Noise' if k == -1 else f'Cluster {k}'}")
        ax.set_title(f"DBSCAN on {name} — {n_clusters} clusters, {n_noise} noise")
        ax.legend(fontsize=7)

    all_lines.append("=" * 60)
    output_text = "\n".join(all_lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "dbscan_clusters.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_dbscan()
