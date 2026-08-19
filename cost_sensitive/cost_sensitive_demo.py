"""
Cost-Sensitive Learning Demo
==============================
The default 0.5 threshold is almost never optimal in production.
This demo shows how to find the threshold that minimizes total business cost.

Scenario: Fraud detection
  - Missing a fraud (FN) costs $10,000
  - False alarm (FP) costs $50 (analyst reviews it)
  - The optimal threshold is much lower than 0.5

Part 1 — Default threshold vs cost-optimal threshold
Part 2 — Cost curve: total cost as a function of threshold
Part 3 — Expected value framework for individual decisions
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def compute_cost(y_true, y_pred, cost_fn, cost_fp):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return fn * cost_fn + fp * cost_fp, cm


def run_cost_sensitive_demo():
    log("COST-SENSITIVE LEARNING DEMO")
    log("=" * 60)

    np.random.seed(42)

    # Imbalanced fraud-like dataset (5% positive)
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
                               n_redundant=5, weights=[0.95, 0.05],
                               flip_y=0.01, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                         stratify=y, random_state=42)

    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    COST_FN = 10000  # cost of missing a fraud
    COST_FP = 50     # cost of a false alarm

    log(f"\n  Scenario: Fraud detection")
    log(f"  Cost of missed fraud (FN): ${COST_FN:,}")
    log(f"  Cost of false alarm  (FP): ${COST_FP:,}")
    log(f"  Cost ratio: {COST_FN / COST_FP:.0f}:1")
    log(f"  Dataset: {len(y)} samples, {y.mean():.1%} positive (fraud)")

    # ═══════════════════════════════════════════════════════
    # Part 1: Default vs Optimal Threshold
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 1: DEFAULT vs OPTIMAL THRESHOLD")
    log("=" * 60)

    # Default threshold
    y_pred_default = (probs >= 0.5).astype(int)
    cost_default, cm_default = compute_cost(y_test, y_pred_default, COST_FN, COST_FP)
    tn, fp, fn, tp = cm_default.ravel()

    log(f"\n  Default threshold = 0.50:")
    log(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    log(f"    Accuracy: {accuracy_score(y_test, y_pred_default):.3f}")
    log(f"    Missed frauds: {fn}  ×  ${COST_FN:,} = ${fn * COST_FN:,}")
    log(f"    False alarms:  {fp}  ×  ${COST_FP:,} = ${fp * COST_FP:,}")
    log(f"    TOTAL COST: ${cost_default:,}")

    # Sweep thresholds to find optimal
    thresholds = np.arange(0.01, 0.99, 0.01)
    costs = []
    for t in thresholds:
        y_pred_t = (probs >= t).astype(int)
        c, _ = compute_cost(y_test, y_pred_t, COST_FN, COST_FP)
        costs.append(c)
    costs = np.array(costs)

    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    best_cost = costs[best_idx]

    y_pred_optimal = (probs >= best_threshold).astype(int)
    _, cm_optimal = compute_cost(y_test, y_pred_optimal, COST_FN, COST_FP)
    tn_o, fp_o, fn_o, tp_o = cm_optimal.ravel()

    log(f"\n  Optimal threshold = {best_threshold:.2f}:")
    log(f"    TP={tp_o}, FP={fp_o}, FN={fn_o}, TN={tn_o}")
    log(f"    Accuracy: {accuracy_score(y_test, y_pred_optimal):.3f}")
    log(f"    Missed frauds: {fn_o}  ×  ${COST_FN:,} = ${fn_o * COST_FN:,}")
    log(f"    False alarms:  {fp_o}  ×  ${COST_FP:,} = ${fp_o * COST_FP:,}")
    log(f"    TOTAL COST: ${best_cost:,}")

    savings = cost_default - best_cost
    log(f"\n  Savings: ${savings:,} ({savings/cost_default:.0%} reduction)")

    # ═══════════════════════════════════════════════════════
    # Part 2: Multiple Cost Scenarios
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 2: DIFFERENT COST RATIOS → DIFFERENT THRESHOLDS")
    log("=" * 60)

    scenarios = [
        ("Equal costs",       100,   100),
        ("FN 10x more",      1000,   100),
        ("FN 200x more",    10000,    50),
        ("FP 5x more",        100,   500),
    ]

    log(f"\n  {'Scenario':<20} {'Cost FN':>8} {'Cost FP':>8} {'Best t':>7} {'Total cost':>12}")
    log(f"  {'-' * 57}")

    scenario_thresholds = []
    for name, c_fn, c_fp in scenarios:
        scenario_costs = []
        for t in thresholds:
            y_p = (probs >= t).astype(int)
            c, _ = compute_cost(y_test, y_p, c_fn, c_fp)
            scenario_costs.append(c)
        best_t = thresholds[np.argmin(scenario_costs)]
        best_c = min(scenario_costs)
        scenario_thresholds.append((name, best_t))
        log(f"  {name:<20} ${c_fn:>7,} ${c_fp:>7,} {best_t:>7.2f} ${best_c:>11,}")

    log(f"\n  → When FN is expensive, threshold drops (catch more positives)")
    log(f"  → When FP is expensive, threshold rises (fewer false alarms)")

    # ═══════════════════════════════════════════════════════
    # Part 3: Expected Value Framework
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 3: EXPECTED VALUE PER PREDICTION")
    log("=" * 60)

    log(f"\n  For each prediction, compute:")
    log(f"    E[cost if flag]    = P(legit) × cost_FP = (1-p) × ${COST_FP}")
    log(f"    E[cost if ignore]  = P(fraud) × cost_FN = p × ${COST_FN}")
    log(f"    Flag if E[cost if ignore] > E[cost if flag]")
    log(f"    i.e., p × {COST_FN} > (1-p) × {COST_FP}")
    log(f"    i.e., p > {COST_FP} / ({COST_FN} + {COST_FP}) = {COST_FP / (COST_FN + COST_FP):.4f}")
    log(f"\n  Theoretical optimal threshold: {COST_FP / (COST_FN + COST_FP):.4f}")
    log(f"  Empirical optimal threshold:   {best_threshold:.2f}")
    log(f"  (They differ because the model's probabilities aren't perfectly calibrated)")

    # Sample predictions
    log(f"\n  Sample decisions:")
    log(f"  {'Prob':>6} | {'E[cost ignore]':>14} | {'E[cost flag]':>12} | Decision")
    log(f"  {'-' * 52}")
    for p in [0.01, 0.05, 0.10, 0.30, 0.50, 0.80]:
        e_ignore = p * COST_FN
        e_flag = (1 - p) * COST_FP
        decision = "FLAG" if e_ignore > e_flag else "IGNORE"
        log(f"  {p:>6.2f} | ${e_ignore:>13,.0f} | ${e_flag:>11,.0f} | {decision}")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cost curve
    ax = axes[0, 0]
    ax.plot(thresholds, costs, color="#e74c3c", linewidth=2)
    ax.axvline(x=0.5, color="gray", linestyle="--", label=f"Default (${cost_default:,})")
    ax.axvline(x=best_threshold, color="#2ecc71", linestyle="--",
               label=f"Optimal t={best_threshold:.2f} (${best_cost:,})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title(f"Cost Curve (FN=${COST_FN:,}, FP=${COST_FP})")
    ax.legend()

    # Plot 2: Confusion matrices side by side
    ax = axes[0, 1]
    labels = ["Default (t=0.5)", f"Optimal (t={best_threshold:.2f})"]
    cms = [cm_default, cm_optimal]
    bar_data = {
        "True Pos": [cm_default[1, 1], cm_optimal[1, 1]],
        "False Neg": [cm_default[1, 0], cm_optimal[1, 0]],
        "False Pos": [cm_default[0, 1], cm_optimal[0, 1]],
    }
    x = np.arange(len(labels))
    w = 0.25
    colors_bars = ["#2ecc71", "#e74c3c", "#f39c12"]
    for i, (label, vals) in enumerate(bar_data.items()):
        ax.bar(x + i * w - w, vals, w, label=label, color=colors_bars[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Default vs Optimal: What Changes")
    ax.legend()

    # Plot 3: Threshold vs metrics
    ax = axes[1, 0]
    accs, f1s, recalls, precisions = [], [], [], []
    for t in thresholds:
        y_p = (probs >= t).astype(int)
        accs.append(accuracy_score(y_test, y_p))
        f1s.append(f1_score(y_test, y_p, zero_division=0))
        tp_t = ((y_p == 1) & (y_test == 1)).sum()
        fn_t = ((y_p == 0) & (y_test == 1)).sum()
        fp_t = ((y_p == 1) & (y_test == 0)).sum()
        recalls.append(tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0)
        precisions.append(tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0)
    ax.plot(thresholds, accs, label="Accuracy", linewidth=2)
    ax.plot(thresholds, f1s, label="F1", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", linewidth=2)
    ax.plot(thresholds, precisions, label="Precision", linewidth=2)
    ax.axvline(x=best_threshold, color="gray", linestyle="--", alpha=0.5, label=f"Cost-optimal ({best_threshold:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Threshold")
    ax.legend(fontsize=8)

    # Plot 4: Different cost ratios → different thresholds
    ax = axes[1, 1]
    scenario_names = [s[0] for s in scenario_thresholds]
    scenario_ts = [s[1] for s in scenario_thresholds]
    colors_s = ["#3498db", "#f39c12", "#e74c3c", "#2ecc71"]
    bars = ax.barh(scenario_names, scenario_ts, color=colors_s)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Default 0.5")
    ax.set_xlabel("Optimal Threshold")
    ax.set_title("Cost Ratio Determines Threshold")
    for bar, t in zip(bars, scenario_ts):
        ax.text(t + 0.02, bar.get_y() + bar.get_height()/2, f"{t:.2f}",
                va="center", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cost_sensitive.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/cost_sensitive.png")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_cost_sensitive_demo()
