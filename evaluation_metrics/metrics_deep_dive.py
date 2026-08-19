"""
Evaluation Metrics — Deep Dive
================================
Goes far beyond accuracy:
  1. Confusion matrix analysis + per-class metrics
  2. Precision-Recall tradeoff & PR curves
  3. ROC curves + AUC comparison
  4. When to use F1 vs AUC vs log-loss
  5. Threshold tuning for business cost optimisation
  6. Calibration curves (reliability diagrams)

Run:
    python metrics_deep_dive.py
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    roc_curve, auc,
    classification_report,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_metrics_demo():
    lines = [
        "=" * 70,
        "  EVALUATION METRICS  —  Deep Dive",
        "=" * 70, "",
    ]

    # imbalanced dataset — 10% positive rate (realistic for fraud, churn, disease)
    X, y = make_classification(
        n_samples=3000, n_features=10, n_informative=6,
        n_redundant=2, n_clusters_per_class=2,
        weights=[0.9, 0.1],  # 90/10 imbalance
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    lines.append(f"  Dataset: 3000 samples, 10% positive rate (imbalanced)")
    lines.append(f"  Train: {len(y_train)} ({y_train.sum()} pos)  |  Test: {len(y_test)} ({y_test.sum()} pos)")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "y_pred": y_pred, "y_proba": y_proba,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "avg_precision": average_precision_score(y_test, y_proba),
            "log_loss": log_loss(y_test, y_proba),
            "brier": brier_score_loss(y_test, y_proba),
        }

    # ── 1. Metric comparison table ──
    lines += [
        "", "  ── 1. Metric Comparison (default threshold = 0.5) ──", "",
        f"  {'Model':25s} | {'Acc':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'AUC':>6s} | {'AP':>6s} | {'LogLoss':>8s} | {'Brier':>6s}",
        f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}",
    ]
    for name, m in results.items():
        lines.append(
            f"  {name:25s} | {m['accuracy']:6.4f} | {m['precision']:6.4f} | {m['recall']:6.4f} | "
            f"{m['f1']:6.4f} | {m['roc_auc']:6.4f} | {m['avg_precision']:6.4f} | {m['log_loss']:8.4f} | {m['brier']:6.4f}"
        )

    # ── 2. Why accuracy is misleading ──
    majority_acc = 1 - y_test.mean()
    lines += [
        "", "  ── 2. Why Accuracy Misleads on Imbalanced Data ──",
        f"    A 'predict all negative' model gets accuracy = {majority_acc:.4f}",
        f"    But it catches ZERO positives (recall = 0) — useless for fraud/disease.",
        f"    → Use F1, AUC, or Average Precision instead.",
    ]

    # ── 3. Confusion matrices ──
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i, (name, m) in enumerate(results.items()):
        cm = confusion_matrix(y_test, m["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"]).plot(ax=axes[i], cmap="Blues")
        axes[i].set_title(name, fontsize=10)
    plt.suptitle("Confusion Matrices (threshold = 0.5)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    lines.append(f"\n  [saved] → plots/confusion_matrices.png")

    # ── 4. ROC curves ──
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, m in results.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] → plots/roc_curves.png")

    # ── 5. Precision-Recall curves ──
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, m in results.items():
        prec, rec, _ = precision_recall_curve(y_test, m["y_proba"])
        ax.plot(rec, prec, label=f"{name} (AP={m['avg_precision']:.3f})")
    ax.axhline(y=y_test.mean(), color="k", linestyle="--", alpha=0.3, label=f"Baseline (prevalence={y_test.mean():.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (better for imbalanced data)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pr_curves.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] → plots/pr_curves.png")

    # ── 6. Threshold tuning ──
    lines += ["", "  ── 6. Threshold Tuning (Gradient Boosting) ──"]
    gb_proba = results["Gradient Boosting"]["y_proba"]
    thresholds = np.arange(0.1, 0.9, 0.05)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_t = (gb_proba >= t).astype(int)
        precisions.append(precision_score(y_test, y_t, zero_division=0))
        recalls.append(recall_score(y_test, y_t))
        f1s.append(f1_score(y_test, y_t, zero_division=0))

    best_f1_idx = np.argmax(f1s)
    best_threshold = thresholds[best_f1_idx]
    lines.append(f"    Default threshold 0.50 → F1 = {results['Gradient Boosting']['f1']:.4f}")
    lines.append(f"    Best threshold   {best_threshold:.2f} → F1 = {f1s[best_f1_idx]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision", marker=".")
    ax.plot(thresholds, recalls, label="Recall", marker=".")
    ax.plot(thresholds, f1s, label="F1", marker=".", linewidth=2)
    ax.axvline(x=best_threshold, color="red", linestyle="--", alpha=0.5, label=f"Best F1 @ {best_threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "threshold_tuning.png"), dpi=150)
    plt.close()
    lines.append(f"    [saved] → plots/threshold_tuning.png")

    # ── 7. Cost-based threshold ──
    lines += ["", "  ── 7. Cost-Based Threshold Selection ──",
              "    Scenario: missed fraud costs $10,000, false alarm costs $50"]
    cost_fn = 10000   # cost of false negative
    cost_fp = 50      # cost of false positive
    costs = []
    for t in thresholds:
        y_t = (gb_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_t)
        tn, fp, fn, tp = cm.ravel()
        total_cost = fn * cost_fn + fp * cost_fp
        costs.append(total_cost)
    best_cost_idx = np.argmin(costs)

    y_def = (gb_proba >= 0.5).astype(int)
    cm_def = confusion_matrix(y_test, y_def)
    tn_d, fp_d, fn_d, tp_d = cm_def.ravel()
    default_cost = fn_d * cost_fn + fp_d * cost_fp

    lines.append(f"    Best cost threshold: {thresholds[best_cost_idx]:.2f} (total cost: ${costs[best_cost_idx]:,.0f})")
    lines.append(f"    vs default 0.50                 (total cost: ${default_cost:,})")

    # ── 8. Calibration curves ──
    lines += ["", "  ── 8. Calibration Curves ──",
              "    'When the model says 80% sure, is it right 80% of the time?'"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for name, m in results.items():
        fraction_pos, mean_predicted = calibration_curve(y_test, m["y_proba"], n_bins=10)
        ax.plot(mean_predicted, fraction_pos, marker="o", label=f"{name} (Brier={m['brier']:.3f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves (Reliability Diagram)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "calibration_curves.png"), dpi=150)
    plt.close()
    lines.append(f"    [saved] → plots/calibration_curves.png")

    # ── 9. When to use which metric ──
    lines += [
        "", "  ── 9. When to Use Which Metric ──", "",
        "    Metric          | Best For",
        "    ────────────────┼──────────────────────────────────────────────",
        "    Accuracy         | Balanced classes, equal misclassification cost",
        "    Precision        | False positives are expensive (spam filter, content moderation)",
        "    Recall           | False negatives are expensive (disease screening, fraud)",
        "    F1               | Need balance of precision & recall, single threshold",
        "    ROC AUC          | Ranking quality across all thresholds, balanced/moderate imbalance",
        "    Average Prec.    | Imbalanced data (< 5% positive), ranking quality",
        "    Log Loss         | Calibrated probabilities matter (pricing, risk scoring)",
        "    Brier Score      | Calibration quality (reliability of predicted probabilities)",
    ]

    lines += ["", "=" * 70]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_metrics_demo()
