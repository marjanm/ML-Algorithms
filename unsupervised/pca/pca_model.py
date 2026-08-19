"""
PCA — fully parameterised
===========================
Principal Component Analysis. Projects high-dimensional data into fewer
dimensions while preserving the maximum variance.

Run:
    python pca_model.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_pca(
    n_components: int = 2,              # number of components to keep; int or float (fraction of variance)
    whiten: bool = False,               # if True, each component is scaled to unit variance
    svd_solver: str = "auto",           # "auto", "full", "arpack", "randomized"
    tol: float = 0.0,                   # tolerance for singular values (arpack solver)
    iterated_power: str = "auto",       # number of iterations for randomized SVD
    random_state: int = 42,
):
    digits = load_digits()
    X, y = digits.data, digits.target   # 1797 samples × 64 features (8×8 pixel images of digits 0-9)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # full PCA to get explained variance ratios for all components
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X_s)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    # reduced PCA for visualisation
    pca = PCA(
        n_components=n_components, whiten=whiten, svd_solver=svd_solver,
        tol=tol, iterated_power=iterated_power, random_state=random_state,
    )

    start = time.perf_counter()
    X_reduced = pca.fit_transform(X_s)
    t = time.perf_counter() - start

    lines = [
        "=" * 60, "  PCA  —  Results", "=" * 60,
        f"  Original dims  : {X.shape[1]}",
        f"  Reduced dims   : {n_components}",
        f"  Explained var  : {pca.explained_variance_ratio_}",
        f"  Total captured : {sum(pca.explained_variance_ratio_):.4f}",
        f"  Time           : {t:.4f}s",
        "",
        "  Components needed for 90% variance : "
            f"{np.searchsorted(cumvar, 0.90) + 1}",
        "  Components needed for 95% variance : "
            f"{np.searchsorted(cumvar, 0.95) + 1}",
        "  Components needed for 99% variance : "
            f"{np.searchsorted(cumvar, 0.99) + 1}",
        "=" * 60,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 2D scatter
    scatter = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="tab10", s=8, alpha=0.6)
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].set_title("Digits projected to 2 PCs")
    plt.colorbar(scatter, ax=axes[0], label="digit")

    # scree / cumulative variance
    axes[1].bar(range(1, len(pca_full.explained_variance_ratio_[:20]) + 1),
                pca_full.explained_variance_ratio_[:20], alpha=0.6, label="Individual")
    axes[1].plot(range(1, 21), cumvar[:20], "ro-", label="Cumulative")
    axes[1].axhline(0.95, ls="--", color="grey", label="95 % threshold")
    axes[1].set_xlabel("Component")
    axes[1].set_ylabel("Explained variance ratio")
    axes[1].set_title("Scree Plot (first 20 components)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pca_results.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_pca()
