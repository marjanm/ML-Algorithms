"""
Regularization — visual demo
==============================
Techniques to prevent overfitting: L1 (Lasso), L2 (Ridge), Dropout.

This demo:
  1. Fits polynomial regression with increasing degree (no regularisation)
     → shows severe overfitting on high-degree polynomials
  2. Adds Ridge (L2) and Lasso (L1) → watch the wiggly curve smooth out
  3. Shows coefficient magnitudes: L2 shrinks all, L1 zeros some out

Run:
    python regularization_demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_regularization_demo():
    np.random.seed(42)
    n_train, n_test = 30, 100
    X = np.sort(np.random.uniform(0, 1, n_train))
    y_true = np.sin(2 * np.pi * X)
    y = y_true + np.random.normal(0, 0.3, n_train)

    X_test = np.sort(np.random.uniform(0, 1, n_test))
    y_test = np.sin(2 * np.pi * X_test) + np.random.normal(0, 0.3, n_test)

    X_plot = np.linspace(0, 1, 300)
    y_plot_true = np.sin(2 * np.pi * X_plot)

    degrees = [1, 4, 15]
    alpha = 1.0
    alpha_lasso = 0.01

    lines = [
        "=" * 65, "  REGULARIZATION  —  Demo", "=" * 65,
        "  True function: sin(2πx) + noise",
        f"  Train samples: {n_train}, Test samples: {n_test}",
        "",
    ]

    fig, axes = plt.subplots(len(degrees), 3, figsize=(18, 5 * len(degrees)))

    for row, deg in enumerate(degrees):
        models = {
            f"No reg (deg={deg})": make_pipeline(PolynomialFeatures(deg), LinearRegression()),
            f"Ridge L2 (deg={deg})": make_pipeline(PolynomialFeatures(deg), Ridge(alpha=alpha)),
            f"Lasso L1 (deg={deg})": make_pipeline(PolynomialFeatures(deg), Lasso(alpha=alpha_lasso, max_iter=10000)),
        }

        for col, (name, model) in enumerate(models.items()):
            model.fit(X.reshape(-1, 1), y)
            y_pred = model.predict(X_plot.reshape(-1, 1))
            mse_train = mean_squared_error(y, model.predict(X.reshape(-1, 1)))
            mse_test = mean_squared_error(y_test, model.predict(X_test.reshape(-1, 1)))

            ax = axes[row, col]
            ax.scatter(X, y, color="black", s=20, zorder=5, label="Train data")
            ax.scatter(X_test, y_test, color="blue", s=12, alpha=0.4, zorder=4, label="Test data")
            ax.plot(X_plot, y_plot_true, "g--", alpha=0.5, label="True sin(2πx)")
            ax.plot(X_plot, y_pred, "r-", linewidth=2,
                    label=f"Train MSE={mse_train:.3f}\nTest  MSE={mse_test:.3f}")
            ax.set_title(name, fontsize=11)
            ax.set_ylim(-2, 2)
            ax.legend(fontsize=7)

            coefs = model[-1].coef_
            lines.append(f"  {name}: train_MSE={mse_train:.4f}, test_MSE={mse_test:.4f}, "
                         f"max|coef|={np.max(np.abs(coefs)):.2f}, "
                         f"zeros={np.sum(np.abs(coefs) < 1e-6)}/{len(coefs)}")

        lines.append("")

    lines += [
        "  Takeaway:",
        "    - No regularisation + high degree → wild oscillations (overfitting)",
        "    - Ridge (L2) → shrinks all coefficients, smooths the curve",
        "    - Lasso (L1) → zeros out coefficients → automatic feature selection",
        "=" * 65,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "regularization.png"), dpi=150)
    plt.close()

    # --- coefficient magnitude comparison for degree-15 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    deg = 15
    models = {
        "No regularisation": make_pipeline(PolynomialFeatures(deg), LinearRegression()),
        "Ridge (L2, α=1.0)": make_pipeline(PolynomialFeatures(deg), Ridge(alpha=alpha)),
        "Lasso (L1, α=0.01)": make_pipeline(PolynomialFeatures(deg), Lasso(alpha=alpha_lasso, max_iter=10000)),
    }
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X.reshape(-1, 1), y)
        coefs = model[-1].coef_
        ax.bar(range(len(coefs)), np.abs(coefs), color="tab:blue", alpha=0.7)
        ax.set_xlabel("Coefficient index")
        ax.set_ylabel("|coefficient|")
        ax.set_title(f"{name}\nmax={np.max(np.abs(coefs)):.1f}, zeros={np.sum(np.abs(coefs)<1e-6)}")
        ax.set_yscale("symlog")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "coefficient_comparison.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_regularization_demo()
