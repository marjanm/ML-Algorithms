"""
Hyperparameter Tuning — visual demo
======================================
GridSearch vs RandomSearch vs Bayesian optimisation.
Systematically finding the best model settings.

This demo:
  1. GridSearch  — tests every combination on a grid (exhaustive but slow)
  2. RandomSearch — samples random combos (faster, often as good)
  3. Compares search time, best score, and visualises which combos were tried

Run:
    python hyperparameter_tuning_demo.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from scipy.stats import randint, uniform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_generators.classification_data import generate_synthetic_data

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_hyperparameter_tuning_demo():
    X_train, X_test, y_train, y_test, _ = generate_synthetic_data(
        n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # --- 1. GridSearch ---
    grid_params = {
        "n_estimators": [50, 100, 200],         # 3 values
        "max_depth": [3, 5, 10, None],           # 4 values
        "min_samples_split": [2, 5, 10],         # 3 values
        "min_samples_leaf": [1, 2, 4],           # 3 values
    }
    # Total combos: 3 × 4 × 3 × 3 = 108

    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=grid_params,
        cv=cv, scoring="accuracy", n_jobs=-1, return_train_score=True,
    )
    t0 = time.perf_counter()
    gs.fit(X_train, y_train)
    grid_time = time.perf_counter() - t0

    # --- 2. RandomSearch (same budget: 108 iterations, but sampling randomly) ---
    random_params = {
        "n_estimators": randint(30, 300),
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
    }
    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=random_params,
        n_iter=108, cv=cv, scoring="accuracy", n_jobs=-1,
        random_state=42, return_train_score=True,
    )
    t0 = time.perf_counter()
    rs.fit(X_train, y_train)
    rand_time = time.perf_counter() - t0

    lines = [
        "=" * 65, "  HYPERPARAMETER TUNING  —  Demo", "=" * 65, "",
        "  GridSearch (108 combos, exhaustive):",
        f"    Best accuracy : {gs.best_score_:.4f}",
        f"    Best params   : {gs.best_params_}",
        f"    Time          : {grid_time:.2f}s",
        "",
        "  RandomSearch (108 random combos):",
        f"    Best accuracy : {rs.best_score_:.4f}",
        f"    Best params   : {rs.best_params_}",
        f"    Time          : {rand_time:.2f}s",
        "",
        "  Takeaway:",
        "    - GridSearch: guaranteed to find the best combo in the grid, but exponential cost",
        "    - RandomSearch: same budget, often finds equally good or better results",
        "    - For 5+ hyperparameters, RandomSearch is strongly preferred",
        "    - Bayesian optimisation (Optuna, scikit-optimize) is even smarter —",
        "      it uses past results to decide where to search next",
        "=" * 65,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. score distribution
    axes[0].hist(gs.cv_results_["mean_test_score"], bins=20, alpha=0.6, label="GridSearch", color="tab:blue", edgecolor="black")
    axes[0].hist(rs.cv_results_["mean_test_score"], bins=20, alpha=0.6, label="RandomSearch", color="tab:orange", edgecolor="black")
    axes[0].axvline(gs.best_score_, color="tab:blue", ls="--", linewidth=2)
    axes[0].axvline(rs.best_score_, color="tab:orange", ls="--", linewidth=2)
    axes[0].set_xlabel("Mean CV accuracy")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score distribution of all tried combos")
    axes[0].legend()

    # 2. max_depth vs n_estimators heatmap (grid)
    depths = sorted([d for d in grid_params["max_depth"] if d is not None])
    n_ests = sorted(grid_params["n_estimators"])
    scores_grid = np.zeros((len(depths), len(n_ests)))
    for i, row in enumerate(gs.cv_results_["params"]):
        if row["max_depth"] is not None and row["min_samples_split"] == 2 and row["min_samples_leaf"] == 1:
            di = depths.index(row["max_depth"])
            ni = n_ests.index(row["n_estimators"])
            scores_grid[di, ni] = gs.cv_results_["mean_test_score"][i]
    im = axes[1].imshow(scores_grid, cmap="YlGn", aspect="auto")
    axes[1].set_xticks(range(len(n_ests)))
    axes[1].set_xticklabels(n_ests)
    axes[1].set_yticks(range(len(depths)))
    axes[1].set_yticklabels(depths)
    axes[1].set_xlabel("n_estimators")
    axes[1].set_ylabel("max_depth")
    axes[1].set_title("GridSearch heatmap\n(max_depth × n_estimators, min_samples=2/1)")
    plt.colorbar(im, ax=axes[1], label="accuracy")

    # 3. cumulative best score
    gs_cummax = np.maximum.accumulate(gs.cv_results_["mean_test_score"])
    rs_cummax = np.maximum.accumulate(rs.cv_results_["mean_test_score"])
    axes[2].plot(gs_cummax, label="GridSearch", linewidth=2)
    axes[2].plot(rs_cummax, label="RandomSearch", linewidth=2)
    axes[2].set_xlabel("Number of combinations tried")
    axes[2].set_ylabel("Best accuracy so far")
    axes[2].set_title("Cumulative best — how fast do we find a good combo?")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "hyperparameter_tuning.png"), dpi=150)
    plt.close()
    print(f"  [saved] plots → {PLOT_DIR}")


if __name__ == "__main__":
    run_hyperparameter_tuning_demo()
