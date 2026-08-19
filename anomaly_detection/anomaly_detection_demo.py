"""
Anomaly Detection — Demo
==========================
Find rare, unusual data points. Used in fraud detection, system monitoring,
manufacturing quality control.

Demonstrates:
  1. Isolation Forest      — isolates anomalies via random splits (fewer splits = more anomalous)
  2. One-Class SVM         — learns boundary of normal data in kernel space
  3. Local Outlier Factor  — density-based: anomalies live in low-density regions
  4. Comparison across contamination levels

Run:
    python anomaly_detection_demo.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_generators.anomaly_data import generate_anomaly_data

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_anomaly_demo():
    lines = [
        "=" * 65,
        "  ANOMALY DETECTION  —  Demo",
        "=" * 65, "",
    ]

    X, y_true = generate_anomaly_data(n_normal=950, n_anomalies=50, save_csv=False)
    contamination = y_true.mean()
    lines.append(f"  Dataset: {len(y_true)} samples, {y_true.sum():.0f} anomalies ({contamination:.1%} contamination)")

    models = {
        "Isolation Forest": IsolationForest(
            n_estimators=100,          # number of isolation trees
            contamination=contamination,  # expected fraction of outliers
            max_samples="auto",        # samples per tree (auto = min(256, n))
            max_features=1.0,          # features per tree
            random_state=42,
        ),
        "One-Class SVM": OneClassSVM(
            kernel="rbf",              # kernel type
            gamma="scale",             # kernel coefficient
            nu=contamination,          # upper bound on fraction of outliers
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=20,            # neighbours for density estimation
            contamination=contamination,
            novelty=False,             # False = use fit_predict (transductive)
        ),
    }

    results = {}
    for name, model in models.items():
        y_pred_raw = model.fit_predict(X)
        # sklearn convention: -1 = outlier, 1 = inlier → convert to 0/1
        y_pred = (y_pred_raw == -1).astype(int)

        results[name] = {
            "y_pred": y_pred,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    # ── Results table ──
    lines += [
        "", "  ── Results ──", "",
        f"  {'Model':25s} | {'Acc':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s}",
        f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}",
    ]
    for name, m in results.items():
        lines.append(
            f"  {name:25s} | {m['accuracy']:6.4f} | {m['precision']:6.4f} | "
            f"{m['recall']:6.4f} | {m['f1']:6.4f}"
        )

    # ── Confusion matrices ──
    for name, m in results.items():
        cm = confusion_matrix(y_true, m["y_pred"])
        lines += [
            f"\n  {name}:",
            f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}",
            f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}",
        ]

    # ── Visualisation: scatter + decision boundaries ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, m) in enumerate(results.items()):
        ax = axes[i]
        y_pred = m["y_pred"]
        ax.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c="steelblue", s=10, alpha=0.5, label="Normal")
        ax.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c="red", s=30, alpha=0.8, label="Anomaly", marker="x")

        # mark ground-truth anomalies that were missed (false negatives)
        fn_mask = (y_true == 1) & (y_pred == 0)
        if fn_mask.sum() > 0:
            ax.scatter(X[fn_mask, 0], X[fn_mask, 1], facecolors="none", edgecolors="orange",
                       s=80, linewidths=2, label="Missed (FN)")

        ax.set_title(f"{name}\nF1={m['f1']:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Anomaly Detection — Model Comparison", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "anomaly_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    lines.append(f"\n  [saved] → plots/anomaly_comparison.png")

    # ── Contamination sensitivity (Isolation Forest) ──
    lines += ["", "  ── Contamination Sensitivity (Isolation Forest) ──"]
    contam_levels = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
    f1_scores = []
    for c in contam_levels:
        ifo = IsolationForest(n_estimators=100, contamination=c, random_state=42)
        y_pred = (ifo.fit_predict(X) == -1).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
        lines.append(f"    contamination={c:.2f} → F1={f1:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(contam_levels, f1_scores, marker="o", color="coral")
    ax.axvline(x=contamination, color="green", linestyle="--", alpha=0.5,
               label=f"True contamination ({contamination:.2f})")
    ax.set_xlabel("Contamination Parameter")
    ax.set_ylabel("F1 Score")
    ax.set_title("Isolation Forest — Sensitivity to Contamination Setting")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "contamination_sensitivity.png"), dpi=150)
    plt.close()
    lines.append(f"  [saved] → plots/contamination_sensitivity.png")

    lines += [
        "", "  ── Key Takeaways ──",
        "    • Isolation Forest: fast, scalable, works well with high-dimensional data",
        "    • One-Class SVM: good when normal data has a clear boundary, but slow on large data",
        "    • LOF: density-based, catches local anomalies that global methods miss",
        "    • The contamination parameter is critical — set it based on domain knowledge",
        "    • In production, you rarely know the true contamination — start conservative",
        "", "=" * 65,
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_anomaly_demo()
