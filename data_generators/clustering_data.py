"""
Clustering Data Generator
==========================
Generates several clustering datasets with different shapes:
  - Blobs       (spherical, well-separated — K-Means friendly)
  - Moons       (non-convex crescents — DBSCAN friendly)
  - Circles     (concentric rings — DBSCAN friendly)
  - Anisotropic (stretched blobs — tests linkage assumptions)

Saves all to CSV and returns them as a dict.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def generate_clustering_data(
    n_samples: int = 500,
    random_state: int = 42,
    save_csv: bool = True,
):
    """Create several clustering datasets.

    Returns
    -------
    datasets : dict[str, (X, y_true)]
        Key = dataset name, value = (feature array, ground-truth labels).
    """
    datasets = {}

    # 1. Blobs — spherical, well-separated
    X, y = make_blobs(
        n_samples=n_samples, centers=4,
        cluster_std=[1.0, 1.5, 0.5, 1.2],
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    datasets["blobs"] = (X, y)

    # 2. Moons — two interleaving crescents
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    datasets["moons"] = (X, y)

    # 3. Circles — concentric rings
    X, y = make_circles(n_samples=n_samples, noise=0.05, factor=0.4, random_state=random_state)
    X = StandardScaler().fit_transform(X)
    datasets["circles"] = (X, y)

    # 4. Anisotropic blobs — stretched ellipses
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = X @ transformation
    X = StandardScaler().fit_transform(X)
    datasets["anisotropic"] = (X, y)

    if save_csv:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for name, (X_d, y_d) in datasets.items():
            df = pd.DataFrame(X_d, columns=["feature_0", "feature_1"])
            df["cluster"] = y_d
            path = os.path.join(OUTPUT_DIR, f"clustering_{name}.csv")
            df.to_csv(path, index=False)
        print(f"[clustering] Saved {len(datasets)} datasets  ->  {OUTPUT_DIR}")

    return datasets


if __name__ == "__main__":
    ds = generate_clustering_data()
    for name, (X, y) in ds.items():
        print(f"  {name:15s} | shape={X.shape} | clusters={len(set(y))}")
