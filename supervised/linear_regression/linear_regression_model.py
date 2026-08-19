"""
Linear Regression — fully parameterised
=========================================
The baseline for regression tasks.  We also show Ridge (L2) and Lasso (L1)
variants so you can compare regularisation effects.

Generates its own synthetic regression data, evaluates MSE/MAE/R², and
plots actual-vs-predicted plus residual distributions.

Run:
    python linear_regression_model.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_linear_regression(
    n_samples: int = 500,
    n_features: int = 10,           # total number of features
    n_informative: int = 6,         # features actually useful for predicting target
    noise: float = 15.0,            # stdev of gaussian noise added to the target
    random_state: int = 42,
):
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, noise=noise, random_state=random_state,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
    )

    models = {
        "OLS (no reg.)": LinearRegression(
            fit_intercept=True,     # learn a bias term
            n_jobs=-1,              # parallelise across cores
        ),
        "Ridge (L2)": Ridge(
            alpha=1.0,              # regularisation strength; larger = stronger
            fit_intercept=True,
            solver="auto",          # "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"
            max_iter=1000,          # max solver iterations
            tol=1e-4,              # convergence tolerance
            random_state=random_state,
        ),
        "Lasso (L1)": Lasso(
            alpha=0.5,              # L1 penalty; drives small weights to exactly zero → feature selection
            fit_intercept=True,
            max_iter=5000,
            tol=1e-4,
            warm_start=False,
            selection="cyclic",     # "cyclic" or "random" — order of feature updates
            random_state=random_state,
        ),
        "ElasticNet (L1+L2)": ElasticNet(
            alpha=0.5,              # overall regularisation strength
            l1_ratio=0.5,           # mix of L1 vs L2: 0 = pure Ridge, 1 = pure Lasso
            fit_intercept=True,
            max_iter=5000,
            tol=1e-4,
            warm_start=False,
            selection="cyclic",
            random_state=random_state,
        ),
    }

    all_lines = ["=" * 60, "  LINEAR REGRESSION  —  Results", "=" * 60, ""]
    results_dict = {}

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        t = time.perf_counter() - start
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results_dict[name] = {"mse": mse, "mae": mae, "r2": r2, "time": t, "y_pred": y_pred}
        all_lines += [
            f"  {name}:", f"    MSE   = {mse:.4f}", f"    MAE   = {mae:.4f}",
            f"    R²    = {r2:.4f}", f"    Time  = {t:.4f}s", "",
        ]

    all_lines.append("=" * 60)
    output_text = "\n".join(all_lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, res) in zip(axes.flat, results_dict.items()):
        ax.scatter(y_test, res["y_pred"], alpha=0.5, s=15)
        lo, hi = y_test.min(), y_test.max()
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_title(f"{name}  (R²={res['r2']:.3f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, res) in zip(axes.flat, results_dict.items()):
        residuals = y_test - res["y_pred"]
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(f"{name} residuals")
        ax.set_xlabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "residuals.png"), dpi=150)
    plt.close()

    print(f"  [saved] plots → {PLOT_DIR}")
    return results_dict


if __name__ == "__main__":
    run_linear_regression()
