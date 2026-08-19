"""
Extreme Imbalance Demo
========================
What happens when only 0.5% of samples are positive (fraud, rare disease, etc.)?

Part 1 — The Accuracy Trap:
    A model that always predicts "negative" gets 99.5% accuracy.
    Show why accuracy is meaningless here.

Part 2 — Techniques:
    Class weights, SMOTE oversampling, threshold tuning.
    Compare using precision@k, recall, PR-AUC (not accuracy).

Part 3 — Two-Stage System:
    Fast recall-focused filter → precise expensive model. How production
    fraud detection actually works.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve, confusion_matrix,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def precision_at_k(y_true, probs, k):
    """Of the top-k scored samples, how many are actually positive?"""
    top_k_idx = np.argsort(-probs)[:k]
    return y_true[top_k_idx].mean()


def run_extreme_imbalance_demo():
    log("EXTREME IMBALANCE DEMO")
    log("=" * 60)

    np.random.seed(42)

    # 0.5% positive rate
    X, y = make_classification(n_samples=20000, n_features=20, n_informative=8,
                               n_redundant=5, weights=[0.995, 0.005],
                               flip_y=0.001, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    n_pos_train = y_train.sum()
    n_pos_test = y_test.sum()

    log(f"\n  Dataset: {len(y)} samples")
    log(f"  Positive rate: {y.mean():.3%}")
    log(f"  Train: {len(y_train)} ({n_pos_train} positives)")
    log(f"  Test:  {len(y_test)} ({n_pos_test} positives)")

    # ═══════════════════════════════════════════════════════
    # Part 1: The Accuracy Trap
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 1: THE ACCURACY TRAP")
    log("=" * 60)

    y_all_negative = np.zeros_like(y_test)
    acc_dummy = accuracy_score(y_test, y_all_negative)
    log(f"\n  'Always predict negative' baseline:")
    log(f"    Accuracy: {acc_dummy:.3f} ({acc_dummy:.1%})")
    log(f"    Precision: {precision_score(y_test, y_all_negative, zero_division=0):.3f}")
    log(f"    Recall:    {recall_score(y_test, y_all_negative, zero_division=0):.3f}")
    log(f"    F1:        {f1_score(y_test, y_all_negative, zero_division=0):.3f}")
    log(f"    → 99.5% accurate but catches ZERO fraud!")

    # ═══════════════════════════════════════════════════════
    # Part 2: Techniques Comparison
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 2: HANDLING IMBALANCE")
    log("=" * 60)

    approaches = {}

    # Vanilla (no handling)
    rf_vanilla = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_vanilla.fit(X_train, y_train)
    probs_vanilla = rf_vanilla.predict_proba(X_test)[:, 1]
    approaches["Vanilla RF"] = probs_vanilla

    # Class weights (built-in)
    rf_weighted = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf_weighted.fit(X_train, y_train)
    probs_weighted = rf_weighted.predict_proba(X_test)[:, 1]
    approaches["RF (class_weight=balanced)"] = probs_weighted

    # Manual oversampling (duplicate minority class)
    pos_mask = y_train == 1
    X_pos = X_train[pos_mask]
    y_pos = y_train[pos_mask]
    oversample_factor = int((y_train == 0).sum() / pos_mask.sum()) // 2
    X_over = np.vstack([X_train, np.tile(X_pos, (oversample_factor, 1))])
    y_over = np.concatenate([y_train, np.tile(y_pos, oversample_factor)])
    rf_over = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_over.fit(X_over, y_over)
    probs_over = rf_over.predict_proba(X_test)[:, 1]
    approaches["RF (oversampled)"] = probs_over

    # Gradient Boosting with scale_pos_weight
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    gb_weighted = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, random_state=42,
    )
    # GradientBoosting doesn't have scale_pos_weight, use sample_weight
    sample_weights = np.where(y_train == 1, ratio, 1.0)
    gb_weighted.fit(X_train, y_train, sample_weight=sample_weights)
    probs_gb = gb_weighted.predict_proba(X_test)[:, 1]
    approaches["GB (weighted)"] = probs_gb

    # Compare
    log(f"\n  {'Method':<30} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6} {'PR-AUC':>7} {'P@50':>6}")
    log(f"  {'-' * 72}")

    results_for_plot = {}
    for name, probs in approaches.items():
        y_pred = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_test, probs)
        p_at_50 = precision_at_k(y_test, probs, 50)

        results_for_plot[name] = {"probs": probs, "pr_auc": pr_auc, "p_at_50": p_at_50}
        log(f"  {name:<30} {acc:>6.3f} {prec:>6.3f} {rec:>7.3f} {f1:>6.3f} {pr_auc:>7.3f} {p_at_50:>6.3f}")

    # ═══════════════════════════════════════════════════════
    # Part 3: Two-Stage System
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 3: TWO-STAGE SYSTEM")
    log("=" * 60)

    # Stage 1: Fast, recall-focused filter (low threshold)
    stage1_threshold = 0.02
    stage1_pass = probs_gb >= stage1_threshold
    stage1_recall = y_test[stage1_pass].sum() / max(y_test.sum(), 1)
    stage1_volume = stage1_pass.sum()

    log(f"\n  Stage 1: Fast filter (threshold={stage1_threshold})")
    log(f"    Passes {stage1_volume} samples ({stage1_volume/len(y_test):.1%} of traffic)")
    log(f"    Recall: {stage1_recall:.1%} (catches {y_test[stage1_pass].sum()}/{int(y_test.sum())} positives)")

    # Stage 2: Precise model on filtered subset
    X_stage2 = X_test[stage1_pass]
    y_stage2 = y_test[stage1_pass]
    probs_stage2 = probs_gb[stage1_pass]
    stage2_threshold = 0.3
    y_pred_stage2 = (probs_stage2 >= stage2_threshold).astype(int)

    if y_pred_stage2.sum() > 0:
        stage2_prec = precision_score(y_stage2, y_pred_stage2, zero_division=0)
        stage2_rec = recall_score(y_stage2, y_pred_stage2, zero_division=0)
        overall_recall = (y_pred_stage2 & (y_stage2 == 1)).sum() / max(y_test.sum(), 1)

        log(f"\n  Stage 2: Precise model (threshold={stage2_threshold})")
        log(f"    Flags {y_pred_stage2.sum()} samples for human review")
        log(f"    Precision: {stage2_prec:.1%} (of flagged, this many are real fraud)")
        log(f"    Overall recall: {overall_recall:.1%}")

    log(f"\n  Why two stages?")
    log(f"    Stage 1 runs on ALL {len(y_test)} transactions (must be fast, <5ms)")
    log(f"    Stage 2 runs on only {stage1_volume} candidates (can be slow, <500ms)")
    log(f"    Human review sees only {y_pred_stage2.sum()} cases (manageable)")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: PR curves
    ax = axes[0, 0]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for (name, res), color in zip(results_for_plot.items(), colors):
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, res["probs"])
        ax.plot(rec_curve, prec_curve, label=f"{name} (AUC={res['pr_auc']:.3f})",
                color=color, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (imbalanced data)")
    ax.legend(fontsize=7)

    # Plot 2: Precision@k
    ax = axes[0, 1]
    ks = [10, 20, 50, 100, 200, 500]
    for (name, res), color in zip(results_for_plot.items(), colors):
        p_at_ks = [precision_at_k(y_test, res["probs"], k) for k in ks]
        ax.plot(ks, p_at_ks, "o-", label=name, color=color, linewidth=2)
    ax.set_xlabel("k (top-k predictions)")
    ax.set_ylabel("Precision@k")
    ax.set_title("Precision@k: How Pure Are the Top Predictions?")
    ax.legend(fontsize=7)

    # Plot 3: Score distributions
    ax = axes[1, 0]
    probs_pos = probs_gb[y_test == 1]
    probs_neg = probs_gb[y_test == 0]
    ax.hist(probs_neg, bins=50, alpha=0.6, color="#3498db", label=f"Negative (n={len(probs_neg)})",
            density=True)
    ax.hist(probs_pos, bins=50, alpha=0.6, color="#e74c3c", label=f"Positive (n={len(probs_pos)})",
            density=True)
    ax.axvline(x=0.5, color="black", linestyle="--", label="Default threshold")
    ax.axvline(x=stage1_threshold, color="#2ecc71", linestyle="--", label=f"Stage 1 ({stage1_threshold})")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Class")
    ax.legend(fontsize=8)

    # Plot 4: Metric comparison
    ax = axes[1, 1]
    method_names = list(results_for_plot.keys())
    pr_aucs = [results_for_plot[n]["pr_auc"] for n in method_names]
    p_at_50s = [results_for_plot[n]["p_at_50"] for n in method_names]
    x = np.arange(len(method_names))
    w = 0.3
    ax.bar(x - w/2, pr_aucs, w, label="PR-AUC", color="#3498db")
    ax.bar(x + w/2, p_at_50s, w, label="Precision@50", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("(", "\n(") for n in method_names], fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("Ranking Metrics Comparison")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "extreme_imbalance.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/extreme_imbalance.png")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_extreme_imbalance_demo()
