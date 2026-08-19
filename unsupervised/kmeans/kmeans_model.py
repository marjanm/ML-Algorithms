"""
K-Means Clustering — fully parameterised (standalone)
======================================================
K-Means is unsupervised — it discovers clusters without labels.
This script generates 2D data, runs K-Means with all hyper-parameters
exposed, evaluates with internal metrics, and saves plots.

Run:
    python kmeans_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_kmeans(
    X: np.ndarray,
    # --- core hyper-parameters ---
    n_clusters: int = 4,                 # number of clusters to find; use elbow method or silhouette to pick
    init: str = "k-means++",            # centroid initialisation: "k-means++" (smart, spread-out seeds), "random", or an ndarray of starting points
    n_init: int = 10,                   # how many times to re-run with different seeds; picks the best result (lowest inertia)
    max_iter: int = 300,                # max iterations per single run before stopping
    tol: float = 1e-4,                  # convergence threshold; stop when centroid movement < tol
    algorithm: str = "lloyd",           # "lloyd" (classic EM-style) or "elkan" (faster with triangle inequality, needs euclidean)
    # --- behaviour ---
    random_state: int = 42,             # seed for reproducibility
    verbose: int = 0,                   # verbosity level; 0 = silent
    copy_x: bool = True,               # if True, keeps original data unchanged; False = modifies in-place to save memory
):
    """Run K-Means clustering and return (model, results_dict)."""

    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        algorithm=algorithm,
        random_state=random_state,
        verbose=verbose,
        copy_x=copy_x,
    )

    labels = model.fit_predict(X)

    results = {
        "model_name": "K-Means",
        "n_clusters": n_clusters,
        "inertia": model.inertia_,
        "n_iter": model.n_iter_,
        "cluster_centers": model.cluster_centers_,
        "labels": labels,
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }

    lines = [
        "=" * 50,
        f"  K-MEANS  —  Results  (k={n_clusters})",
        "=" * 50,
        f"  Inertia          : {results['inertia']:.2f}",
        f"  Iterations       : {results['n_iter']}",
        f"  Silhouette       : {results['silhouette']:.4f}",
        f"  Calinski-Harabasz: {results['calinski_harabasz']:.2f}",
        f"  Davies-Bouldin   : {results['davies_bouldin']:.4f}",
        "=" * 50,
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)

    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    return model, results


def plot_clusters(X, model, results):
    """Scatter plot of clusters with centroids."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=results["labels"],
                         cmap="viridis", s=15, alpha=0.6)
    centers = results["cluster_centers"]
    ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="X",
               s=200, edgecolors="k", linewidths=1.5, label="Centroids")
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title(f"K-Means Clustering  (k={results['n_clusters']}, "
                 f"silhouette={results['silhouette']:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    path = os.path.join(PLOT_DIR, "kmeans_clusters.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_elbow(X, k_range=range(2, 11)):
    """Elbow method: inertia vs number of clusters."""
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(list(k_range), inertias, "bo-", linewidth=2)
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_title("Elbow Method")
    ax1.grid(alpha=0.3)

    ax2.plot(list(k_range), silhouettes, "rs-", linewidth=2)
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs k")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "kmeans_elbow.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def main():
    print("\n" + "━" * 50)
    print("  K-MEANS CLUSTERING DEMO")
    print("━" * 50 + "\n")

    X, y_true = make_blobs(
        n_samples=1500, n_features=2, centers=4,
        cluster_std=1.2, random_state=42,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model, results = run_kmeans(X, n_clusters=4)

    print("Generating plots …")
    plot_clusters(X, model, results)
    plot_elbow(X)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
