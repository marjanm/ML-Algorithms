"""
Anomaly Detection Data Generator
==================================
Generates a dataset of "normal" points with injected anomalies.
Useful for Isolation Forest, One-Class SVM, autoencoder anomaly detection.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def generate_anomaly_data(
    n_normal: int = 950,             # number of normal data points
    n_anomalies: int = 50,           # number of injected outliers
    n_features: int = 2,             # dimensionality
    contamination: float = None,     # if set, overrides n_anomalies = int(total * contamination)
    normal_std: float = 1.0,         # spread of normal data
    anomaly_range: float = 6.0,      # outliers sampled uniformly in [-range, range]
    random_state: int = 42,
    save_csv: bool = True,
):
    """Create a dataset with normal points and outliers.

    Returns
    -------
    X : numpy array (n_normal + n_anomalies, n_features)
    y : numpy array — 0 = normal, 1 = anomaly
    """
    np.random.seed(random_state)

    if contamination is not None:
        total = n_normal + n_anomalies
        n_anomalies = int(total * contamination)
        n_normal = total - n_anomalies

    X_normal, _ = make_blobs(
        n_samples=n_normal, n_features=n_features,
        centers=1, cluster_std=normal_std, random_state=random_state,
    )
    X_anomaly = np.random.uniform(
        -anomaly_range, anomaly_range, size=(n_anomalies, n_features),
    )

    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)]).astype(int)

    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]

    if save_csv:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["is_anomaly"] = y
        path = os.path.join(OUTPUT_DIR, "anomaly_dataset.csv")
        df.to_csv(path, index=False)
        print(f"[anomaly] Saved {len(df)} rows ({n_anomalies} anomalies)  ->  {path}")

    return X, y


if __name__ == "__main__":
    X, y = generate_anomaly_data()
    print(f"Shape: {X.shape} | Normal: {(y == 0).sum()} | Anomalies: {(y == 1).sum()}")
