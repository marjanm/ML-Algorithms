"""
Ethics, Fairness & Bias in ML Demo
====================================
Shows how ML models can learn and amplify societal bias, and how to detect
and mitigate it.

Part 1 — Biased Dataset:
    Synthetic hiring data where approval rates differ by a protected attribute
    (group A vs group B).  A model trained naively inherits the bias.

Part 2 — Fairness Metrics:
    Demographic parity, equalized odds, disparate impact ratio, calibration
    across groups.

Part 3 — Bias Mitigation:
    Threshold adjustment and reweighting to satisfy fairness constraints
    while preserving as much accuracy as possible.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def create_biased_hiring_data(n=3000, seed=42):
    """Generate synthetic hiring data with embedded bias."""
    np.random.seed(seed)

    # Protected attribute: group (0=majority, 1=minority)
    group = np.random.binomial(1, 0.4, n)  # 40% minority

    # Legitimate features (skill, experience) — same distribution for both groups
    skill = np.random.normal(50, 15, n)
    experience = np.random.normal(10, 5, n).clip(0, 30)
    education = np.random.normal(14, 3, n).clip(8, 22)

    # Historical hiring decision — BIASED against minority group
    # Both groups have same real qualifications, but minority has lower acceptance
    score = 0.4 * skill + 0.3 * experience + 0.2 * education
    threshold = np.percentile(score, 50)
    base_prob = 1 / (1 + np.exp(-0.1 * (score - threshold)))

    # Inject bias: minority group gets a penalty
    bias_penalty = -0.20 * group  # 20% lower acceptance probability for minority
    hired = (np.random.random(n) < (base_prob + bias_penalty).clip(0, 1)).astype(int)

    X = np.column_stack([skill, experience, education])
    feature_names = ["Skill", "Experience", "Education"]
    return X, hired, group, feature_names


def compute_fairness_metrics(y_true, y_pred, group):
    """Compute standard fairness metrics."""
    metrics = {}

    # Positive prediction rates by group
    rate_0 = y_pred[group == 0].mean()
    rate_1 = y_pred[group == 1].mean()
    metrics["acceptance_rate_majority"] = rate_0
    metrics["acceptance_rate_minority"] = rate_1

    # Demographic Parity: P(Y=1|G=0) = P(Y=1|G=1)
    metrics["demographic_parity_diff"] = abs(rate_0 - rate_1)

    # Disparate Impact Ratio: min(rate_0/rate_1, rate_1/rate_0) ≥ 0.8 (4/5 rule)
    metrics["disparate_impact_ratio"] = min(rate_0, rate_1) / max(rate_0, rate_1) if max(rate_0, rate_1) > 0 else 0

    # Equalized Odds: TPR and FPR should be equal across groups
    for g, name in [(0, "majority"), (1, "minority")]:
        mask = group == g
        if mask.sum() == 0:
            continue
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics[f"tpr_{name}"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[f"fpr_{name}"] = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics["equalized_odds_tpr_diff"] = abs(metrics.get("tpr_majority", 0) - metrics.get("tpr_minority", 0))
    metrics["equalized_odds_fpr_diff"] = abs(metrics.get("fpr_majority", 0) - metrics.get("fpr_minority", 0))

    return metrics


def run_fairness_demo():
    log("ETHICS, FAIRNESS & BIAS IN ML DEMO")
    log("=" * 60)

    # --- Part 1: Train on biased data ---
    log("\nPART 1: BIASED DATASET & NAIVE MODEL")
    log("=" * 60)

    X, y, group, feat_names = create_biased_hiring_data()
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, group, test_size=0.3, random_state=42
    )

    log(f"\nDataset: {len(X)} synthetic hiring decisions")
    log(f"  Majority group (G=0): {(group == 0).sum()}")
    log(f"  Minority group (G=1): {(group == 1).sum()}")
    log(f"\nHistorical hiring rates (ground truth — contains bias):")
    log(f"  Majority: {y[group == 0].mean():.1%}")
    log(f"  Minority: {y[group == 1].mean():.1%}")
    log(f"  Gap: {y[group == 0].mean() - y[group == 1].mean():.1%}")

    # Train naive model (includes no fairness constraints)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    log(f"\nNaive model accuracy: {accuracy_score(y_test, y_pred):.3f}")

    # --- Part 2: Measure fairness ---
    log("\n" + "=" * 60)
    log("PART 2: FAIRNESS METRICS")
    log("=" * 60)

    metrics = compute_fairness_metrics(y_test, y_pred, g_test)

    log(f"\nAcceptance rates:")
    log(f"  Majority: {metrics['acceptance_rate_majority']:.1%}")
    log(f"  Minority: {metrics['acceptance_rate_minority']:.1%}")
    log(f"\nDemographic Parity difference: {metrics['demographic_parity_diff']:.3f}")
    log(f"  (ideal: 0.0 — equal acceptance rates)")
    log(f"\nDisparate Impact Ratio: {metrics['disparate_impact_ratio']:.3f}")
    log(f"  (legal threshold: ≥ 0.80, the '4/5 rule')")
    log(f"  {'✓ PASSES' if metrics['disparate_impact_ratio'] >= 0.8 else '✗ FAILS'} the 4/5 rule")
    log(f"\nEqualized Odds:")
    log(f"  TPR difference: {metrics['equalized_odds_tpr_diff']:.3f}")
    log(f"  FPR difference: {metrics['equalized_odds_fpr_diff']:.3f}")
    log(f"  (ideal: both 0.0 — model performs equally regardless of group)")

    # --- Part 3: Mitigation ---
    log("\n" + "=" * 60)
    log("PART 3: BIAS MITIGATION")
    log("=" * 60)

    # Method 1: Threshold adjustment (post-processing)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Find group-specific thresholds that equalize acceptance rates
    target_rate = y_proba.mean()  # overall positive rate as target
    best_t0, best_t1 = 0.5, 0.5
    best_diff = 1.0
    for t0 in np.arange(0.1, 0.9, 0.01):
        for t1 in np.arange(0.1, 0.9, 0.01):
            pred_0 = (y_proba[g_test == 0] >= t0).mean()
            pred_1 = (y_proba[g_test == 1] >= t1).mean()
            diff = abs(pred_0 - pred_1)
            if diff < best_diff:
                best_diff = diff
                best_t0, best_t1 = t0, t1

    y_pred_fair = np.zeros_like(y_pred)
    y_pred_fair[g_test == 0] = (y_proba[g_test == 0] >= best_t0).astype(int)
    y_pred_fair[g_test == 1] = (y_proba[g_test == 1] >= best_t1).astype(int)

    fair_metrics = compute_fairness_metrics(y_test, y_pred_fair, g_test)
    fair_acc = accuracy_score(y_test, y_pred_fair)

    log(f"\nMethod 1: Group-specific threshold adjustment")
    log(f"  Threshold for majority: {best_t0:.2f}")
    log(f"  Threshold for minority: {best_t1:.2f}")
    log(f"  Accuracy: {fair_acc:.3f} (was {accuracy_score(y_test, y_pred):.3f})")
    log(f"  Demographic Parity diff: {fair_metrics['demographic_parity_diff']:.3f} (was {metrics['demographic_parity_diff']:.3f})")
    log(f"  Disparate Impact Ratio: {fair_metrics['disparate_impact_ratio']:.3f} (was {metrics['disparate_impact_ratio']:.3f})")
    log(f"  {'✓ PASSES' if fair_metrics['disparate_impact_ratio'] >= 0.8 else '✗ FAILS'} the 4/5 rule")

    # Method 2: Reweighting (pre-processing)
    # Give higher weight to under-represented group/label combos
    sample_weights = np.ones(len(X_train))
    for g in [0, 1]:
        for label in [0, 1]:
            mask = (g_train == g) & (y_train == label)
            expected = len(X_train) / 4  # uniform expectation
            actual = mask.sum()
            if actual > 0:
                sample_weights[mask] = expected / actual

    model_rw = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_rw.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred_rw = model_rw.predict(X_test)
    rw_metrics = compute_fairness_metrics(y_test, y_pred_rw, g_test)
    rw_acc = accuracy_score(y_test, y_pred_rw)

    log(f"\nMethod 2: Reweighting (sample weights during training)")
    log(f"  Accuracy: {rw_acc:.3f}")
    log(f"  Demographic Parity diff: {rw_metrics['demographic_parity_diff']:.3f}")
    log(f"  Disparate Impact Ratio: {rw_metrics['disparate_impact_ratio']:.3f}")
    log(f"  {'✓ PASSES' if rw_metrics['disparate_impact_ratio'] >= 0.8 else '✗ FAILS'} the 4/5 rule")

    # Summary table
    log(f"\n{'─' * 60}")
    log(f"{'Method':<30} {'Accuracy':>10} {'DP Diff':>10} {'DI Ratio':>10}")
    log(f"{'─' * 60}")
    log(f"{'Naive model':<30} {accuracy_score(y_test, y_pred):>10.3f} {metrics['demographic_parity_diff']:>10.3f} {metrics['disparate_impact_ratio']:>10.3f}")
    log(f"{'Threshold adjustment':<30} {fair_acc:>10.3f} {fair_metrics['demographic_parity_diff']:>10.3f} {fair_metrics['disparate_impact_ratio']:>10.3f}")
    log(f"{'Reweighting':<30} {rw_acc:>10.3f} {rw_metrics['demographic_parity_diff']:>10.3f} {rw_metrics['disparate_impact_ratio']:>10.3f}")
    log(f"{'─' * 60}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Acceptance rates before/after
    ax = axes[0]
    x = np.arange(2)
    w = 0.25
    ax.bar(x - w, [metrics["acceptance_rate_majority"], metrics["acceptance_rate_minority"]],
           w, label="Naive", color="#e74c3c")
    ax.bar(x, [fair_metrics["acceptance_rate_majority"], fair_metrics["acceptance_rate_minority"]],
           w, label="Threshold adj.", color="#2ecc71")
    ax.bar(x + w, [rw_metrics["acceptance_rate_majority"], rw_metrics["acceptance_rate_minority"]],
           w, label="Reweighting", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(["Majority", "Minority"])
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Acceptance Rates by Group")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Plot 2: Fairness-accuracy tradeoff
    ax = axes[1]
    methods_data = [
        ("Naive", accuracy_score(y_test, y_pred), metrics["disparate_impact_ratio"], "#e74c3c"),
        ("Threshold", fair_acc, fair_metrics["disparate_impact_ratio"], "#2ecc71"),
        ("Reweighting", rw_acc, rw_metrics["disparate_impact_ratio"], "#3498db"),
    ]
    for name, acc, di, color in methods_data:
        ax.scatter(di, acc, s=150, c=color, label=name, zorder=3, edgecolors="black")
        ax.annotate(name, (di, acc), textcoords="offset points", xytext=(10, 5), fontsize=9)
    ax.axvline(x=0.8, color="gray", linestyle="--", alpha=0.5, label="4/5 rule threshold")
    ax.set_xlabel("Disparate Impact Ratio (higher = fairer)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fairness vs Accuracy Tradeoff")
    ax.legend(fontsize=8)

    # Plot 3: Score distributions by group
    ax = axes[2]
    ax.hist(y_proba[g_test == 0], bins=30, alpha=0.6, color="#3498db", label="Majority", density=True)
    ax.hist(y_proba[g_test == 1], bins=30, alpha=0.6, color="#e74c3c", label="Minority", density=True)
    ax.axvline(x=0.5, color="black", linestyle="-", label="Default threshold")
    ax.axvline(x=best_t0, color="#3498db", linestyle="--", label=f"Adj. majority ({best_t0:.2f})")
    ax.axvline(x=best_t1, color="#e74c3c", linestyle="--", label=f"Adj. minority ({best_t1:.2f})")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Score Distributions & Thresholds")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fairness_bias.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/fairness_bias.png")

    # Concepts
    log("\n" + "=" * 60)
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
Fairness Definitions (they can conflict with each other!):
  • Demographic Parity   — equal acceptance rates across groups
  • Equalized Odds       — equal TPR and FPR across groups
  • Calibration          — P(Y=1 | score=s, G=g) same for all g
  • Individual Fairness  — similar individuals get similar predictions

Sources of Bias:
  • Historical bias      — past decisions were discriminatory
  • Representation bias  — training data under-represents a group
  • Measurement bias     — features are measured differently per group
  • Aggregation bias     — one model can't serve all populations

Mitigation Strategies:
  ┌────────────────────────┬──────────────────────────────────────┐
  │ Stage                  │ Methods                              │
  ├────────────────────────┼──────────────────────────────────────┤
  │ Pre-processing         │ Reweighting, resampling, fair repr.  │
  │ In-processing          │ Fairness constraints in loss func.   │
  │ Post-processing        │ Threshold adjustment, calibration    │
  └────────────────────────┴──────────────────────────────────────┘

EU AI Act (2024): High-risk AI systems (hiring, credit, healthcare) must
demonstrate fairness testing, provide explanations, and allow human override.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_fairness_demo()
