"""
Feature Drift Detection & Silent Model Degradation Demo
=========================================================
Shows how to detect when input data distribution shifts after deployment,
causing model accuracy to silently degrade.

Part 1 — Drift Metrics:
    PSI (Population Stability Index), KS test, KL divergence on synthetic data
    that gradually shifts away from the training distribution.

Part 2 — Silent Degradation Simulation:
    Train a model on stable data, then serve it drifting data over 12 "weeks".
    Track accuracy, drift metrics, and prediction distribution over time.
    Show that the model keeps returning predictions but accuracy tanks.

Part 3 — Retraining Trigger:
    When drift metric crosses a threshold, retrain on recent data and show
    accuracy recovery.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def compute_psi(expected, actual, bins=10):
    """
    Population Stability Index.
    PSI < 0.1  → no significant shift
    PSI 0.1-0.25 → moderate shift
    PSI > 0.25 → significant shift
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_counts = np.histogram(expected, bins=breakpoints)[0] + 1
    actual_counts = np.histogram(actual, bins=breakpoints)[0] + 1
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def compute_kl_divergence(p_data, q_data, bins=30):
    """KL(P || Q) using histogram approximation."""
    lo = min(p_data.min(), q_data.min())
    hi = max(p_data.max(), q_data.max())
    p_hist = np.histogram(p_data, bins=bins, range=(lo, hi))[0].astype(float) + 1
    q_hist = np.histogram(q_data, bins=bins, range=(lo, hi))[0].astype(float) + 1
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return np.sum(p_hist * np.log(p_hist / q_hist))


def generate_drifting_data(n, n_features, drift_magnitude, seed=42):
    """Generate classification data with controllable drift."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    X[:, 0] += drift_magnitude  # shift feature 0
    X[:, 1] += drift_magnitude * 0.5  # shift feature 1 less
    w = np.array([1.5, -1.0] + [0.3] * (n_features - 2))
    logits = X @ w + 0.5
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)
    return X, y


def run_drift_demo():
    log("FEATURE DRIFT DETECTION & SILENT DEGRADATION DEMO")
    log("=" * 60)

    np.random.seed(42)
    N_FEATURES = 5

    # ═══════════════════════════════════════════════════════
    # Part 1: Drift Metrics on Controlled Shifts
    # ═══════════════════════════════════════════════════════
    log("\nPART 1: DRIFT METRICS")
    log("=" * 60)

    X_ref, _ = generate_drifting_data(5000, N_FEATURES, drift_magnitude=0.0, seed=42)

    magnitudes = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0]
    psi_vals, ks_vals, kl_vals = [], [], []

    log(f"\n{'Drift':>6} | {'PSI':>8} | {'KS stat':>8} | {'KS p-val':>10} | {'KL div':>8} | Verdict")
    log("-" * 70)

    for mag in magnitudes:
        X_new, _ = generate_drifting_data(5000, N_FEATURES, drift_magnitude=mag, seed=99)

        psi = compute_psi(X_ref[:, 0], X_new[:, 0])
        ks_stat, ks_p = stats.ks_2samp(X_ref[:, 0], X_new[:, 0])
        kl = compute_kl_divergence(X_ref[:, 0], X_new[:, 0])

        psi_vals.append(psi)
        ks_vals.append(ks_stat)
        kl_vals.append(kl)

        if psi < 0.1:
            verdict = "✓ Stable"
        elif psi < 0.25:
            verdict = "⚠ Moderate"
        else:
            verdict = "✗ DRIFT"

        log(f"{mag:>6.1f} | {psi:>8.4f} | {ks_stat:>8.4f} | {ks_p:>10.2e} | {kl:>8.4f} | {verdict}")

    # ═══════════════════════════════════════════════════════
    # Part 2: Silent Degradation Over Time
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 2: SILENT MODEL DEGRADATION OVER 12 WEEKS")
    log("=" * 60)

    X_train, y_train = generate_drifting_data(3000, N_FEATURES, drift_magnitude=0.0, seed=42)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Simulate 12 weeks of gradually drifting data
    weeks = 12
    weekly_drift = np.linspace(0.0, 3.0, weeks)
    weekly_acc = []
    weekly_psi = []
    weekly_pred_mean = []

    log(f"\n{'Week':>5} | {'Drift':>6} | {'Accuracy':>9} | {'PSI':>8} | {'Pred mean':>10} | Status")
    log("-" * 65)

    for w in range(weeks):
        X_week, y_week = generate_drifting_data(
            500, N_FEATURES, drift_magnitude=weekly_drift[w], seed=100 + w
        )
        y_pred = model.predict(X_week)
        acc = accuracy_score(y_week, y_pred)
        psi = compute_psi(X_train[:, 0], X_week[:, 0])
        pred_mean = y_pred.mean()

        weekly_acc.append(acc)
        weekly_psi.append(psi)
        weekly_pred_mean.append(pred_mean)

        if psi < 0.1:
            status = "✓ OK"
        elif psi < 0.25:
            status = "⚠ Watch"
        else:
            status = "✗ ALERT"

        log(f"{w+1:>5} | {weekly_drift[w]:>6.2f} | {acc:>9.3f} | {psi:>8.4f} | {pred_mean:>10.3f} | {status}")

    log(f"\n  Week 1 accuracy: {weekly_acc[0]:.3f}")
    log(f"  Week 12 accuracy: {weekly_acc[-1]:.3f}")
    log(f"  Drop: {weekly_acc[0] - weekly_acc[-1]:.3f}")
    log(f"  → The model kept serving predictions but accuracy silently degraded")

    # ═══════════════════════════════════════════════════════
    # Part 3: Retraining Trigger
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 3: RETRAINING TRIGGER & RECOVERY")
    log("=" * 60)

    PSI_THRESHOLD = 0.25
    trigger_week = next((w for w, p in enumerate(weekly_psi) if p > PSI_THRESHOLD), None)

    if trigger_week is not None:
        log(f"\n  PSI threshold ({PSI_THRESHOLD}) crossed at week {trigger_week + 1}")
        log(f"  → Triggering retrain on recent data...")

        # Retrain on data from the drifted distribution
        X_retrain, y_retrain = generate_drifting_data(
            3000, N_FEATURES, drift_magnitude=weekly_drift[trigger_week], seed=200
        )
        model_retrained = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        model_retrained.fit(X_retrain, y_retrain)

        # Evaluate on remaining weeks
        retrained_acc = []
        log(f"\n{'Week':>5} | {'Old model':>10} | {'Retrained':>10}")
        log("-" * 35)
        for w in range(trigger_week, weeks):
            X_week, y_week = generate_drifting_data(
                500, N_FEATURES, drift_magnitude=weekly_drift[w], seed=100 + w
            )
            old_acc = accuracy_score(y_week, model.predict(X_week))
            new_acc = accuracy_score(y_week, model_retrained.predict(X_week))
            retrained_acc.append(new_acc)
            log(f"{w+1:>5} | {old_acc:>10.3f} | {new_acc:>10.3f}")

        log(f"\n  Retraining recovered accuracy from {weekly_acc[trigger_week]:.3f} to {retrained_acc[0]:.3f}")
    else:
        log("  No drift threshold crossed (PSI stayed below threshold)")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Drift metrics vs magnitude
    ax = axes[0, 0]
    ax.plot(magnitudes, psi_vals, "o-", label="PSI", color="#e74c3c", linewidth=2)
    ax.plot(magnitudes, ks_vals, "s-", label="KS stat", color="#3498db", linewidth=2)
    ax.plot(magnitudes, kl_vals, "^-", label="KL div", color="#2ecc71", linewidth=2)
    ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.7, label="PSI warn (0.1)")
    ax.axhline(y=0.25, color="red", linestyle="--", alpha=0.7, label="PSI alert (0.25)")
    ax.set_xlabel("Drift Magnitude")
    ax.set_ylabel("Metric Value")
    ax.set_title("Drift Metrics vs Shift Magnitude")
    ax.legend(fontsize=8)

    # Plot 2: Distribution shift visualization
    ax = axes[0, 1]
    X_ref_f0 = X_ref[:, 0]
    for mag, color, alpha in [(0.0, "#3498db", 0.7), (1.0, "#f39c12", 0.5), (3.0, "#e74c3c", 0.5)]:
        X_shifted, _ = generate_drifting_data(5000, N_FEATURES, drift_magnitude=mag, seed=99)
        ax.hist(X_shifted[:, 0], bins=50, alpha=alpha, color=color, density=True,
                label=f"drift={mag:.1f}")
    ax.set_xlabel("Feature 0 value")
    ax.set_ylabel("Density")
    ax.set_title("Feature Distribution Shift")
    ax.legend()

    # Plot 3: Silent degradation over weeks
    ax = axes[1, 0]
    week_nums = range(1, weeks + 1)
    ax2 = ax.twinx()
    l1 = ax.plot(week_nums, weekly_acc, "o-", color="#e74c3c", linewidth=2, label="Accuracy")
    l2 = ax2.plot(week_nums, weekly_psi, "s-", color="#3498db", linewidth=2, label="PSI")
    ax2.axhline(y=PSI_THRESHOLD, color="#3498db", linestyle="--", alpha=0.5)
    if trigger_week is not None:
        ax.axvline(x=trigger_week + 1, color="gray", linestyle=":", alpha=0.7, label="Retrain trigger")
    ax.set_xlabel("Week")
    ax.set_ylabel("Accuracy", color="#e74c3c")
    ax2.set_ylabel("PSI", color="#3498db")
    ax.set_title("Silent Degradation: Accuracy & Drift Over Time")
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=8, loc="center left")

    # Plot 4: Prediction distribution shift
    ax = axes[1, 1]
    for w_idx in [0, 5, 11]:
        X_w, _ = generate_drifting_data(500, N_FEATURES, drift_magnitude=weekly_drift[w_idx], seed=100 + w_idx)
        probs = model.predict_proba(X_w)[:, 1]
        ax.hist(probs, bins=30, alpha=0.5, density=True, label=f"Week {w_idx+1}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Distribution Drift")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "drift_detection.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/drift_detection.png")

    # Key concepts
    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
Drift types:
  • Data drift     — input feature distributions change (new user demographics)
  • Concept drift  — relationship between features and label changes (fraud tactics evolve)
  • Prediction drift — model output distribution shifts (even if inputs look stable)

Metrics:
  ┌──────────┬────────────────────────────────────────────────────┐
  │ PSI      │ Binned distribution comparison. Industry standard │
  │          │ < 0.1 stable, 0.1-0.25 moderate, > 0.25 alert    │
  ├──────────┼────────────────────────────────────────────────────┤
  │ KS test  │ Max distance between CDFs. Returns p-value.       │
  │          │ Non-parametric, works on any distribution shape    │
  ├──────────┼────────────────────────────────────────────────────┤
  │ KL div   │ Information-theoretic. Asymmetric (order matters) │
  │          │ Sensitive to tail differences                      │
  └──────────┴────────────────────────────────────────────────────┘

Monitoring strategy:
  1. Compute drift metrics per feature on a schedule (hourly/daily)
  2. Alert when PSI > threshold on critical features
  3. Retrain on recent data when drift is confirmed
  4. Track business KPIs alongside drift — drift without KPI impact may not need action
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_drift_demo()
