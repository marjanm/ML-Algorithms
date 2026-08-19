"""
Expectation-Maximization (EM) — from scratch
=============================================
EM is the algorithm behind GMM fitting. It alternates between:
  E-step: compute responsibilities (soft cluster assignments) given current params
  M-step: update params (means, covariances, weights) given responsibilities

This implements EM for a Gaussian Mixture Model using only numpy.

Run:
    python em_from_scratch.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


class GMMFromScratch:
    """Gaussian Mixture Model fitted via Expectation-Maximization."""

    def __init__(self, n_components: int = 4, max_iter: int = 200,
                 tol: float = 1e-4, random_state: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihoods_ = []

    def _init_params(self, X):
        """K-means++ style initialisation for better starting means."""
        n_samples, n_features = X.shape
        k = self.n_components

        means = [X[self.rng.randint(n_samples)]]
        for _ in range(1, k):
            dists = np.array([np.min([np.sum((x - m) ** 2) for m in means]) for x in X])
            probs = dists / dists.sum()
            means.append(X[self.rng.choice(n_samples, p=probs)])
        self.means_ = np.array(means)

        self.covariances_ = np.array([np.eye(n_features) for _ in range(k)])

        self.weights_ = np.ones(k) / k

    def _gaussian_pdf(self, X, mean, cov):
        """Multivariate Gaussian density for each row of X."""
        n_features = X.shape[1]
        diff = X - mean
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)

        norm_const = 1.0 / (np.power(2 * np.pi, n_features / 2) * np.sqrt(det))
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        return norm_const * np.exp(exponent)

    def _e_step(self, X):
        """Compute responsibilities: gamma[i, k] = P(z_i = k | x_i, params)."""
        n_samples = X.shape[0]
        gamma = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            gamma[:, k] = self.weights_[k] * self._gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )

        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.clip(gamma_sum, 1e-300, None)
        gamma /= gamma_sum
        return gamma

    def _m_step(self, X, gamma):
        """Update means, covariances, and weights from responsibilities."""
        n_samples, n_features = X.shape
        Nk = gamma.sum(axis=0)

        for k in range(self.n_components):
            self.means_[k] = (gamma[:, k] @ X) / Nk[k]

            diff = X - self.means_[k]
            self.covariances_[k] = (diff.T @ (diff * gamma[:, k:k+1])) / Nk[k]
            self.covariances_[k] += 1e-6 * np.eye(n_features)

            self.weights_[k] = Nk[k] / n_samples

    def _log_likelihood(self, X):
        """Compute total log-likelihood of the data under the current model."""
        n_samples = X.shape[0]
        ll = np.zeros(n_samples)
        for k in range(self.n_components):
            ll += self.weights_[k] * self._gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )
        return np.sum(np.log(np.clip(ll, 1e-300, None)))

    def fit(self, X):
        """Run EM until convergence or max_iter."""
        self._init_params(X)
        self.log_likelihoods_ = []

        for iteration in range(self.max_iter):
            gamma = self._e_step(X)
            self._m_step(X, gamma)

            ll = self._log_likelihood(X)
            self.log_likelihoods_.append(ll)

            if iteration > 0 and abs(ll - self.log_likelihoods_[-2]) < self.tol:
                self.n_iter_ = iteration + 1
                self.converged_ = True
                return self
        self.n_iter_ = self.max_iter
        self.converged_ = False
        return self

    def predict(self, X):
        """Hard cluster assignments."""
        gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def predict_proba(self, X):
        """Soft cluster assignments (responsibilities)."""
        return self._e_step(X)


def draw_ellipse(ax, mean, cov, color, n_std=2.0):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color, alpha=0.15, linewidth=2)
    ax.add_patch(ellipse)


def plot_clusters(X, model):
    """Scatter plot with covariance ellipses."""
    labels = model.predict(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.viridis
    k = model.n_components
    colors = [cmap(i / max(k - 1, 1)) for i in range(k)]

    for c in range(k):
        mask = labels == c
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[c]], s=15, alpha=0.6,
                   label=f"Cluster {c}")
        draw_ellipse(ax, model.means_[c], model.covariances_[c], colors[c])

    ax.scatter(model.means_[:, 0], model.means_[:, 1],
               c="red", marker="X", s=200, edgecolors="k", linewidths=1.5,
               label="Means", zorder=5)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_title(f"EM from Scratch — GMM Clustering (k={k})")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "em_clusters.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_convergence(model):
    """Log-likelihood vs iteration — should monotonically increase."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(model.log_likelihoods_) + 1),
            model.log_likelihoods_, "b.-", linewidth=2)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("EM Convergence — Log-Likelihood per Iteration")
    ax.grid(alpha=0.3)
    path = os.path.join(PLOT_DIR, "em_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_em_steps(X, n_components=4, steps=(1, 3, 10, 50)):
    """Visualise EM at different iteration counts to show how it converges."""
    fig, axes = plt.subplots(1, len(steps), figsize=(5 * len(steps), 5))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_components - 1, 1)) for i in range(n_components)]

    for ax, max_it in zip(axes, steps):
        model = GMMFromScratch(n_components=n_components, max_iter=max_it,
                               tol=0, random_state=42)
        model.fit(X)
        labels = model.predict(X)

        for c in range(n_components):
            mask = labels == c
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[c]], s=8, alpha=0.5)
            draw_ellipse(ax, model.means_[c], model.covariances_[c], colors[c])

        ax.set_title(f"Iter {max_it}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2)

    fig.suptitle("EM Algorithm — Convergence Over Iterations", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "em_steps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def main():
    print("\n" + "━" * 55)
    print("  EXPECTATION-MAXIMIZATION (FROM SCRATCH) DEMO")
    print("━" * 55 + "\n")

    X, _ = make_blobs(
        n_samples=1500, n_features=2, centers=4,
        cluster_std=[1.0, 1.5, 0.8, 1.2], random_state=42,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = GMMFromScratch(n_components=4, max_iter=100, random_state=42)
    model.fit(X)

    labels = model.predict(X)

    lines = [
        "=" * 55,
        f"  EM FROM SCRATCH  —  Results  (k={model.n_components})",
        "=" * 55,
        f"  Converged        : {model.converged_}",
        f"  Iterations       : {model.n_iter_}",
        f"  Final log-lik    : {model.log_likelihoods_[-1]:.2f}",
        f"  Component weights: {np.round(model.weights_, 3)}",
        "",
        "  Learned means:",
    ]
    for i, m in enumerate(model.means_):
        lines.append(f"    Cluster {i}: [{m[0]:+.3f}, {m[1]:+.3f}]")
    lines.append("=" * 55)

    output_text = "\n".join(lines)
    print("\n" + output_text)

    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    print("\nGenerating plots …")
    plot_clusters(X, model)
    plot_convergence(model)
    plot_em_steps(X)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
