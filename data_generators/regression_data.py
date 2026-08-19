"""
Regression Data Generator
==========================
Generates a regression dataset with controllable noise and feature count,
saves it to CSV, and returns train/test splits.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def generate_regression_data(
    n_samples: int = 1000,
    n_features: int = 10,           # total number of features
    n_informative: int = 6,         # features actually useful for predicting the target
    noise: float = 15.0,            # stdev of gaussian noise added to the target
    random_state: int = 42,
    csv_path: str = None,
    test_size: float = 0.2,
    scale: bool = True,             # whether to standardise features
):
    """Create a synthetic regression dataset and persist it as CSV.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    feature_names : list[str]
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    if csv_path is None:
        csv_path = os.path.join(OUTPUT_DIR, "regression_dataset.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[regression] Saved {len(df)} rows  ->  {csv_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, names = generate_regression_data()
    print(f"Train shape: {X_tr.shape}  |  Test shape: {X_te.shape}")
    print(f"Target range: [{y_tr.min():.1f}, {y_tr.max():.1f}]")
