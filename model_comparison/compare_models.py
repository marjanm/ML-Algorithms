"""
Model Comparison — Random Forest vs XGBoost vs KNN
====================================================
Generates synthetic data, trains all models, and produces a rich set
of comparison plots saved to the `plots/` directory, with results
written to `output.txt`.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_ROOT = os.path.dirname(SCRIPT_DIR)
if ML_ROOT not in sys.path:
    sys.path.insert(0, ML_ROOT)

from data_generators.classification_data import generate_synthetic_data
from ensemble.random_forest.random_forest_model import train_random_forest
from ensemble.xgboost_model.xgboost_model import train_xgboost
from supervised.knn.knn_model import train_knn

PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))

COLORS = {
    "Random Forest": "#2ecc71",
    "XGBoost": "#e74c3c",
    "KNN": "#3498db",
}


def _save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  [saved] plots/{name}")


# ──────────────────────────────────────────────
# 1.  Metric bar chart
# ──────────────────────────────────────────────
def plot_metric_bars(all_results):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    n_models = len(all_results)
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, res in enumerate(all_results):
        vals = [res[m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=res["model_name"],
                      color=COLORS[res["model_name"]], edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics — All Models")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "01_metric_bars.png")


# ──────────────────────────────────────────────
# 2.  ROC curves
# ──────────────────────────────────────────────
def plot_roc_curves(y_test, all_results):
    fig, ax = plt.subplots(figsize=(7, 6))
    for res in all_results:
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, label=f"{res['model_name']} (AUC={res['roc_auc']:.3f})",
                color=COLORS[res["model_name"]], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "02_roc_curves.png")


# ──────────────────────────────────────────────
# 3.  Precision-Recall curves
# ──────────────────────────────────────────────
def plot_precision_recall(y_test, all_results):
    fig, ax = plt.subplots(figsize=(7, 6))
    for res in all_results:
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ax.plot(rec, prec, label=res["model_name"],
                color=COLORS[res["model_name"]], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "03_precision_recall.png")


# ──────────────────────────────────────────────
# 4.  Confusion matrices side-by-side
# ──────────────────────────────────────────────
def plot_confusion_matrices(y_test, all_results):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, all_results):
        ConfusionMatrixDisplay.from_predictions(
            y_test, res["y_pred"], ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{res['model_name']} — Confusion Matrix")
    fig.tight_layout()
    _save(fig, "04_confusion_matrices.png")


# ──────────────────────────────────────────────
# 5.  Feature importance comparison (RF & XGB only)
# ──────────────────────────────────────────────
def plot_feature_importances(feature_names, all_results, top_n=15):
    fi_results = [r for r in all_results if "feature_importances" in r]
    if not fi_results:
        return
    n = len(fi_results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, fi_results):
        importances = res["feature_importances"]
        idx = np.argsort(importances)[-top_n:]
        ax.barh(
            [feature_names[i] for i in idx],
            importances[idx],
            color=COLORS[res["model_name"]],
            edgecolor="white",
        )
        ax.set_xlabel("Importance")
        ax.set_title(f"{res['model_name']} — Top {top_n} Features")
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    _save(fig, "05_feature_importances.png")


# ──────────────────────────────────────────────
# 6.  Probability distribution (histogram)
# ──────────────────────────────────────────────
def plot_proba_distribution(y_test, all_results):
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, all_results):
        for cls, label in [(0, "Class 0"), (1, "Class 1")]:
            mask = y_test == cls
            ax.hist(res["y_proba"][mask], bins=40, alpha=0.6, label=label)
        ax.set_xlabel("Predicted probability of class 1")
        ax.set_ylabel("Count")
        ax.set_title(f"{res['model_name']} — Probability Distribution")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "06_proba_distribution.png")


# ──────────────────────────────────────────────
# 7.  Calibration curve
# ──────────────────────────────────────────────
def plot_calibration(y_test, all_results):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfectly calibrated")
    for res in all_results:
        frac_pos, mean_pred = calibration_curve(y_test, res["y_proba"], n_bins=15)
        ax.plot(mean_pred, frac_pos, "s-", label=res["model_name"],
                color=COLORS[res["model_name"]], linewidth=2)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "07_calibration_curve.png")


# ──────────────────────────────────────────────
# 8.  Training time comparison
# ──────────────────────────────────────────────
def plot_timing(all_results):
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [r["model_name"] for r in all_results]
    train_times = [r["train_time_sec"] for r in all_results]
    pred_times = [r["predict_time_sec"] for r in all_results]

    x = np.arange(len(names))
    w = 0.3
    ax.bar(x - w / 2, train_times, w, label="Train", color="#3498db", edgecolor="white")
    ax.bar(x + w / 2, pred_times, w, label="Predict", color="#f39c12", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training & Prediction Time")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (t, p) in enumerate(zip(train_times, pred_times)):
        ax.text(i - w / 2, t + 0.005, f"{t:.3f}s", ha="center", fontsize=8)
        ax.text(i + w / 2, p + 0.0005, f"{p:.4f}s", ha="center", fontsize=8)

    _save(fig, "08_timing.png")


# ──────────────────────────────────────────────
# 9.  Learning curves (train size vs score)
# ──────────────────────────────────────────────
def plot_learning_curves(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1,
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    train_sizes = np.linspace(0.1, 1.0, 8)

    for ax, (name, model) in zip(axes, models.items()):
        sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, train_sizes=train_sizes,
            cv=5, scoring="accuracy", n_jobs=-1,
        )
        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_std = test_scores.std(axis=1)

        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
        ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha=0.15)
        ax.plot(sizes, train_mean, "o-", label="Train", linewidth=2)
        ax.plot(sizes, test_mean, "s-", label="Validation", linewidth=2)
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name} — Learning Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "09_learning_curves.png")


# ──────────────────────────────────────────────
# 10.  Summary dashboard (single image)
# ──────────────────────────────────────────────
def plot_summary_dashboard(y_test, all_results, feature_names):
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.35)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    n_models = len(all_results)

    # ---- Metric bars ----
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(metrics))
    w = 0.8 / n_models
    for i, res in enumerate(all_results):
        offset = (i - (n_models - 1) / 2) * w
        ax0.bar(x + offset, [res[m] for m in metrics], w,
                label=res["model_name"][:3], color=COLORS[res["model_name"]])
    ax0.set_xticks(x)
    ax0.set_xticklabels([m[:5].upper() for m in metrics], fontsize=6)
    ax0.set_ylim(0, 1.1)
    ax0.legend(fontsize=6)
    ax0.set_title("Metrics", fontsize=9)
    ax0.grid(axis="y", alpha=0.3)

    # ---- ROC ----
    ax1 = fig.add_subplot(gs[0, 1])
    for res in all_results:
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax1.plot(fpr, tpr, label=f"{res['model_name'][:3]} {res['roc_auc']:.3f}",
                 color=COLORS[res["model_name"]], linewidth=1.5)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.legend(fontsize=6)
    ax1.set_title("ROC", fontsize=9)
    ax1.grid(alpha=0.3)

    # ---- PR ----
    ax2 = fig.add_subplot(gs[0, 2])
    for res in all_results:
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ax2.plot(rec, prec, color=COLORS[res["model_name"]],
                 label=res["model_name"][:3], linewidth=1.5)
    ax2.legend(fontsize=6)
    ax2.set_title("Precision-Recall", fontsize=9)
    ax2.grid(alpha=0.3)

    # ---- Calibration ----
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.3)
    for res in all_results:
        frac_pos, mean_pred = calibration_curve(y_test, res["y_proba"], n_bins=15)
        ax3.plot(mean_pred, frac_pos, "s-", color=COLORS[res["model_name"]],
                 label=res["model_name"][:3], linewidth=1.5)
    ax3.legend(fontsize=6)
    ax3.set_title("Calibration", fontsize=9)
    ax3.grid(alpha=0.3)

    # ---- Confusion matrices ----
    for i, res in enumerate(all_results):
        ax = fig.add_subplot(gs[1, i])
        ConfusionMatrixDisplay.from_predictions(
            y_test, res["y_pred"], ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{res['model_name']} CM", fontsize=9)

    # ---- Timing ----
    ax_t = fig.add_subplot(gs[1, 3])
    names_short = [r["model_name"][:3] for r in all_results]
    train_t = [r["train_time_sec"] for r in all_results]
    pred_t = [r["predict_time_sec"] for r in all_results]
    x = np.arange(n_models)
    ax_t.bar(x - 0.15, train_t, 0.3, label="Train", color="#3498db")
    ax_t.bar(x + 0.15, pred_t, 0.3, label="Predict", color="#f39c12")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(names_short)
    ax_t.set_title("Timing (s)", fontsize=9)
    ax_t.legend(fontsize=6)
    ax_t.grid(axis="y", alpha=0.3)

    # ---- Feature importances (bottom row, only models that have them) ----
    fi_results = [r for r in all_results if "feature_importances" in r]
    for i, res in enumerate(fi_results):
        ax = fig.add_subplot(gs[2, i])
        imp = res["feature_importances"]
        idx = np.argsort(imp)[-10:]
        ax.barh([feature_names[j] for j in idx], imp[idx],
                color=COLORS[res["model_name"]])
        ax.set_title(f"{res['model_name']} Top Features", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Random Forest  vs  XGBoost  vs  KNN  —  Full Comparison Dashboard",
                 fontsize=14, fontweight="bold", y=0.98)
    _save(fig, "10_summary_dashboard.png")


# ──────────────────────────────────────────────
# 11.  2D decision boundary
# ──────────────────────────────────────────────
def plot_decision_boundaries(X_train, y_train, X_test, y_test, models_dict):
    h = 0.05
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models_dict.items()):
        Z = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=50, cmap="RdYlGn", alpha=0.7)
        ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
                   c="#e74c3c", edgecolors="k", s=20, label="Class 0", alpha=0.7)
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                   c="#2ecc71", edgecolors="k", s=20, label="Class 1", alpha=0.7)
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        ax.set_title(f"{name} — Decision Boundary")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    _save(fig, "11_decision_boundaries.png")


# ══════════════════════════════════════════════
#                    MAIN
# ══════════════════════════════════════════════
def main():
    log("\n" + "━" * 60)
    log("  RF  vs  XGBOOST  vs  KNN  —  Comparison Pipeline")
    log("━" * 60 + "\n")

    X_train, X_test, y_train, y_test, feature_names = generate_synthetic_data()

    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    knn_model, knn_results = train_knn(X_train, y_train, X_test, y_test)

    all_results = [rf_results, xgb_results, knn_results]

    log("\nGenerating comparison plots …")
    plot_metric_bars(all_results)
    plot_roc_curves(y_test, all_results)
    plot_precision_recall(y_test, all_results)
    plot_confusion_matrices(y_test, all_results)
    plot_feature_importances(feature_names, all_results)
    plot_proba_distribution(y_test, all_results)
    plot_calibration(y_test, all_results)
    plot_timing(all_results)
    plot_learning_curves(X_train, y_train)
    plot_summary_dashboard(y_test, all_results, feature_names)
    if X_train.shape[1] == 2:
        plot_decision_boundaries(X_train, y_train, X_test, y_test, {
            "Random Forest": rf_model,
            "XGBoost": xgb_model,
            "KNN": knn_model,
        })

    log("\n" + "━" * 60)
    log("  FINAL COMPARISON")
    log("━" * 60)
    names = [r["model_name"] for r in all_results]
    header = f"{'Metric':<18}" + "".join(f"{n:>14}" for n in names) + f"  {'Winner':>10}"
    log(header)
    log("─" * len(header))

    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        vals = [r[m] for r in all_results]
        best_idx = int(np.argmax(vals))
        winner = names[best_idx][:3]
        row = f"  {m.upper():<16}" + "".join(f"{v:>14.4f}" for v in vals)
        row += f"  {'<-- ' + winner:>10}"
        log(row)

    train_times = [r["train_time_sec"] for r in all_results]
    best_idx = int(np.argmin(train_times))
    row = f"  {'TRAIN TIME':<16}" + "".join(f"{t:>13.3f}s" for t in train_times)
    row += f"  {'<-- ' + names[best_idx][:3]:>10}"
    log(row)

    log("━" * 60)
    log(f"\n  All plots saved to plots/\n")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"→ Output saved to output.txt")


if __name__ == "__main__":
    main()
