"""
Gaussian Mixture Models — fully parameterised (standalone)
===========================================================
GMM is a probabilistic clustering method. Unlike K-Means (hard assignments),
GMM gives soft probabilities — each point has a probability of belonging to
each cluster. Clusters can be elliptical, not just spherical.

Internally fitted via Expectation-Maximization (EM).

Run:
    python gmm_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_gmm(
    X: np.ndarray,
    # --- core hyper-parameters ---
    n_components: int = 4,              # number of Gaussian components (clusters)
    covariance_type: str = "full",      # "full" (each cluster has its own covariance matrix — most flexible)
                                        # "tied" (all clusters share one covariance matrix)
                                        # "diag" (diagonal covariance — axis-aligned ellipses)
                                        # "spherical" (single variance per cluster — like K-Means)
    max_iter: int = 200,                # max EM iterations
    n_init: int = 5,                    # number of initialisations; picks best by log-likelihood
    init_params: str = "k-means++",     # how to initialise: "k-means++", "kmeans", "random", "random_from_data"
    tol: float = 1e-3,                  # convergence threshold on log-likelihood change
    reg_covar: float = 1e-6,            # regularisation added to covariance diagonal (prevents singularity)
    # --- behaviour ---
    random_state: int = 42,
):
    """Run GMM clustering and return (model, results_dict)."""

    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_init=n_init,
        init_params=init_params,
        tol=tol,
        reg_covar=reg_covar,
        random_state=random_state,
    )

    model.fit(X)
    labels = model.predict(X)
    probs = model.predict_proba(X)

    results = {
        "model_name": "Gaussian Mixture Model",
        "n_components": n_components,
        "covariance_type": covariance_type,
        "converged": model.converged_,
        "n_iter": model.n_iter_,
        "log_likelihood": model.score(X) * len(X),
        "bic": model.bic(X),
        "aic": model.aic(X),
        "labels": labels,
        "probs": probs,
        "means": model.means_,
        "covariances": model.covariances_,
        "weights": model.weights_,
        "silhouette": silhouette_score(X, labels),
    }

    lines = [
        "=" * 55,
        f"  GMM  —  Results  (k={n_components}, cov={covariance_type})",
        "=" * 55,
        f"  Converged        : {results['converged']}",
        f"  Iterations       : {results['n_iter']}",
        f"  Log-likelihood   : {results['log_likelihood']:.2f}",
        f"  BIC              : {results['bic']:.2f}",
        f"  AIC              : {results['aic']:.2f}",
        f"  Silhouette       : {results['silhouette']:.4f}",
        f"  Component weights: {np.round(results['weights'], 3)}",
        "=" * 55,
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)

    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    return model, results


def draw_ellipse(ax, mean, cov, color, n_std=2.0):
    """Draw a covariance ellipse for a 2D Gaussian."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color, alpha=0.15, linewidth=2)
    ax.add_patch(ellipse)


def plot_clusters(X, model, results):
    """Scatter plot coloured by hard assignment, with covariance ellipses."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.viridis
    n_k = results["n_components"]
    colors = [cmap(i / max(n_k - 1, 1)) for i in range(n_k)]

    for k in range(n_k):
        mask = results["labels"] == k
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[k]], s=15, alpha=0.6,
                   label=f"Cluster {k}")
        draw_ellipse(ax, results["means"][k], results["covariances"][k], colors[k])

    ax.scatter(results["means"][:, 0], results["means"][:, 1],
               c="red", marker="X", s=200, edgecolors="k", linewidths=1.5,
               label="Means", zorder=5)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title(f"GMM Clustering  (k={n_k}, silhouette={results['silhouette']:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "gmm_clusters.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_soft_assignments(X, results):
    """Show the max assignment probability — highlights uncertain points."""
    fig, ax = plt.subplots(figsize=(8, 6))
    max_prob = results["probs"].max(axis=1)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=max_prob, cmap="RdYlGn",
                         s=15, alpha=0.7, vmin=0.5, vmax=1.0)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title("GMM Soft Assignments — Max Cluster Probability")
    plt.colorbar(scatter, ax=ax, label="P(most likely cluster)")
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "gmm_soft_assignments.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_model_selection(X, k_range=range(2, 11)):
    """BIC and AIC vs number of components — pick the elbow / minimum."""
    bics, aics = [], []
    for k in k_range:
        gm = GaussianMixture(n_components=k, n_init=3, random_state=42)
        gm.fit(X)
        bics.append(gm.bic(X))
        aics.append(gm.aic(X))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), bics, "bo-", linewidth=2, label="BIC")
    ax.plot(list(k_range), aics, "rs--", linewidth=2, label="AIC")
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("Score (lower is better)")
    ax.set_title("GMM Model Selection — BIC / AIC")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "gmm_model_selection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def main():
    print("\n" + "━" * 55)
    print("  GAUSSIAN MIXTURE MODEL DEMO")
    print("━" * 55 + "\n")

    X, _ = make_blobs(
        n_samples=1500, n_features=2, centers=4,
        cluster_std=[1.0, 1.5, 0.8, 1.2], random_state=42,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model, results = run_gmm(X, n_components=4)

    print("Generating plots …")
    plot_clusters(X, model, results)
    plot_soft_assignments(X, results)
    plot_model_selection(X)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
