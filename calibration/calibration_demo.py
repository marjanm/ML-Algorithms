"""
Calibration Demo
=================
When a model says "80% confidence", is it actually right 80% of the time?

Part 1 — The Problem:
    Many models are poorly calibrated. Random Forest pushes probabilities
    toward 0 and 1. SVM's probabilities are unreliable. Show reliability
    diagrams (calibration curves) for several models.

Part 2 — The Fix:
    Platt scaling (logistic regression on raw scores) and isotonic regression.
    Show before/after calibration curves and Brier scores.

Part 3 — Why It Matters:
    Decision-making depends on calibrated probabilities, not just rankings.
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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def run_calibration_demo():
    log("CALIBRATION DEMO")
    log("=" * 60)

    np.random.seed(42)
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Split train further for calibration
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # ═══════════════════════════════════════════════════════
    # Part 1: Uncalibrated Models
    # ═══════════════════════════════════════════════════════
    log("\nPART 1: UNCALIBRATED MODELS")
    log("=" * 60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }

    uncalibrated_results = {}

    log(f"\n{'Model':<25} {'Brier ↓':>8} {'Log Loss ↓':>10}")
    log("-" * 45)

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, probs)
        ll = log_loss(y_test, probs)
        uncalibrated_results[name] = {"probs": probs, "brier": brier, "logloss": ll}
        log(f"{name:<25} {brier:>8.4f} {ll:>10.4f}")

    # ═══════════════════════════════════════════════════════
    # Part 2: Apply Calibration
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 2: CALIBRATED MODELS")
    log("=" * 60)

    calibrated_results = {}
    methods = ["sigmoid", "isotonic"]  # Platt scaling and isotonic regression

    for method in methods:
        log(f"\n  --- {method.upper()} calibration ---")
        log(f"  {'Model':<25} {'Brier ↓':>8} {'Improvement':>12}")
        log(f"  {'-' * 47}")

        for name, model in models.items():
            model.fit(X_tr, y_tr)
            cal_model = CalibratedClassifierCV(model, method=method, cv="prefit")
            cal_model.fit(X_cal, y_cal)
            probs = cal_model.predict_proba(X_test)[:, 1]
            brier = brier_score_loss(y_test, probs)
            improvement = uncalibrated_results[name]["brier"] - brier
            key = f"{name} ({method})"
            calibrated_results[key] = {"probs": probs, "brier": brier, "method": method, "base_name": name}
            log(f"  {name:<25} {brier:>8.4f} {improvement:>+11.4f}")

    # ═══════════════════════════════════════════════════════
    # Part 3: Analysis
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 3: ANALYSIS")
    log("=" * 60)

    log(f"\n  Which models need calibration most?")
    for name in models:
        orig = uncalibrated_results[name]["brier"]
        platt = calibrated_results[f"{name} (sigmoid)"]["brier"]
        iso = calibrated_results[f"{name} (isotonic)"]["brier"]
        best_cal = min(platt, iso)
        delta = orig - best_cal
        if delta > 0.005:
            log(f"    {name}: Brier improved by {delta:.4f} — NEEDS calibration")
        else:
            log(f"    {name}: Brier changed by {delta:+.4f} — already well-calibrated")

    log(f"\n  Practical impact example:")
    rf_probs = uncalibrated_results["Random Forest"]["probs"]
    rf_cal_probs = calibrated_results["Random Forest (isotonic)"]["probs"]
    high_conf = rf_probs > 0.8
    if high_conf.sum() > 0:
        actual_rate = y_test[high_conf].mean()
        log(f"    RF says >80% confidence for {high_conf.sum()} samples")
        log(f"    Actual positive rate: {actual_rate:.1%}")
        log(f"    {'Well calibrated' if abs(actual_rate - 0.8) < 0.1 else 'MISCALIBRATED — actual rate differs from 80%'}")
    high_conf_cal = rf_cal_probs > 0.8
    if high_conf_cal.sum() > 0:
        actual_rate_cal = y_test[high_conf_cal].mean()
        log(f"    After isotonic: >80% confidence for {high_conf_cal.sum()} samples")
        log(f"    Actual positive rate: {actual_rate_cal:.1%}")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    # Plot 1: Uncalibrated reliability diagrams
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for (name, res), color in zip(uncalibrated_results.items(), colors):
        prob_true, prob_pred = calibration_curve(y_test, res["probs"], n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", label=f"{name}", color=color, linewidth=2)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Uncalibrated Reliability Diagrams")
    ax.legend(fontsize=7)

    # Plot 2: After Platt (sigmoid) calibration
    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for (name, res), color in zip(uncalibrated_results.items(), colors):
        key = f"{name} (sigmoid)"
        if key in calibrated_results:
            prob_true, prob_pred = calibration_curve(y_test, calibrated_results[key]["probs"],
                                                     n_bins=10, strategy="uniform")
            ax.plot(prob_pred, prob_true, "o-", label=f"{name}", color=color, linewidth=2)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("After Platt (Sigmoid) Calibration")
    ax.legend(fontsize=7)

    # Plot 3: After Isotonic calibration
    ax = axes[0, 2]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for (name, res), color in zip(uncalibrated_results.items(), colors):
        key = f"{name} (isotonic)"
        if key in calibrated_results:
            prob_true, prob_pred = calibration_curve(y_test, calibrated_results[key]["probs"],
                                                     n_bins=10, strategy="uniform")
            ax.plot(prob_pred, prob_true, "o-", label=f"{name}", color=color, linewidth=2)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("After Isotonic Calibration")
    ax.legend(fontsize=7)

    # Plot 4: Brier score comparison
    ax = axes[1, 0]
    model_names = list(models.keys())
    x = np.arange(len(model_names))
    w = 0.25
    orig_briers = [uncalibrated_results[n]["brier"] for n in model_names]
    platt_briers = [calibrated_results[f"{n} (sigmoid)"]["brier"] for n in model_names]
    iso_briers = [calibrated_results[f"{n} (isotonic)"]["brier"] for n in model_names]
    ax.bar(x - w, orig_briers, w, label="Uncalibrated", color="#e74c3c")
    ax.bar(x, platt_briers, w, label="Platt (sigmoid)", color="#3498db")
    ax.bar(x + w, iso_briers, w, label="Isotonic", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=7)
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Brier Score: Before vs After Calibration")
    ax.legend(fontsize=8)

    # Plot 5: Probability distributions before/after
    ax = axes[1, 1]
    ax.hist(uncalibrated_results["Random Forest"]["probs"], bins=30, alpha=0.5,
            color="#e74c3c", density=True, label="RF uncalibrated")
    ax.hist(calibrated_results["Random Forest (sigmoid)"]["probs"], bins=30, alpha=0.5,
            color="#3498db", density=True, label="RF Platt (sigmoid)")
    ax.hist(calibrated_results["Random Forest (isotonic)"]["probs"], bins=30, alpha=0.5,
            color="#2ecc71", density=True, label="RF isotonic")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Random Forest: Probability Distribution")
    ax.legend(fontsize=8)

    # Plot 6: SVM before/after (the "destroyed" case)
    ax = axes[1, 2]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for label, key, color in [("Uncalibrated", "SVM (RBF)", "#9b59b6"),
                               ("Platt", "SVM (RBF) (sigmoid)", "#3498db"),
                               ("Isotonic", "SVM (RBF) (isotonic)", "#2ecc71")]:
        src = uncalibrated_results[key] if key in uncalibrated_results else calibrated_results[key]
        prob_true, prob_pred = calibration_curve(y_test, src["probs"], n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "o-", label=label, color=color, linewidth=2)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("SVM (RBF): Calibration Comparison")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "calibration.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/calibration.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
Calibration methods:
  • Platt scaling (sigmoid) — fits a logistic regression on raw scores.
    Works well when miscalibration is monotonic (just needs rescaling).
  • Isotonic regression — fits a non-parametric monotonic step function.
    More flexible, but needs more calibration data (~1000+ samples).

Brier score = mean((predicted_prob - actual_outcome)²)
  Lower is better. Decomposable into reliability + resolution + uncertainty.

Which models are naturally calibrated?
  • Logistic Regression — usually well-calibrated (it optimizes log-loss)
  • Gradient Boosting — reasonably good, but drifts with many trees
  • Random Forest — pushes probabilities toward 0 and 1 (poor calibration)
  • Naive Bayes — pushes toward extremes due to independence assumption
  • SVM — probabilities are retrofitted (Platt scaling internally), often poor
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_calibration_demo()
