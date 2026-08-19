"""
Classification Data Generator
===============================
Generates a binary classification dataset with controllable complexity,
saves it to CSV, and returns train/test splits for downstream use.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def generate_synthetic_data(
    n_samples: int = 5000,
    n_features: int = 2,
    n_informative: int = 2,
    n_redundant: int = 0,
    n_classes: int = 2,
    class_sep: float = 1.0,
    flip_y: float = 0.03,
    random_state: int = 42,
    csv_path: str = None,
    test_size: float = 0.2,
):
    """Create a synthetic classification dataset and persist it as CSV.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    feature_names : list[str]
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=2,
        n_classes=n_classes,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    if csv_path is None:
        csv_path = os.path.join(OUTPUT_DIR, "classification_dataset.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[classification] Saved {len(df)} rows  ->  {csv_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, names = generate_synthetic_data()
    print(f"Train shape: {X_tr.shape}  |  Test shape: {X_te.shape}")
    print(f"Class distribution (train): {np.bincount(y_tr)}")
