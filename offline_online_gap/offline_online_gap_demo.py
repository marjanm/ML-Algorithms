"""
Offline vs Online Gap — Demo
================================
Your model scores 95% on held-out test data, then underperforms in
production. Why? This demo shows the three main causes of
train-serve skew and how to detect / fix each.

1. Feature skew     — features computed differently at train vs serve time
2. Distribution shift — production data drifts from training data
3. Label leakage     — training data contains future information
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


np.random.seed(42)


def demo_feature_skew():
    """
    Simulate: training uses batch-computed features (exact averages),
    serving uses approximate/stale features (cached, slightly wrong).
    """
    log("=" * 60)
    log("SCENARIO 1: FEATURE SKEW")
    log("=" * 60)
    log("""
  Training pipeline computes user_avg_spend from full history.
  Serving pipeline uses a cached value updated every 24h.
  Result: same model, different feature values → different predictions.
""")

    n = 2000
    # True features at training time (batch-computed, correct)
    age = np.random.normal(35, 10, n)
    avg_spend = np.random.exponential(50, n)
    click_rate = np.random.beta(2, 5, n)

    X_train_true = np.column_stack([age, avg_spend, click_rate])
    y = ((avg_spend > 40) & (click_rate > 0.3)).astype(int)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_train_true, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model.fit(X_tr_s, y_tr)
    offline_acc = accuracy_score(y_te, model.predict(X_te_s))
    offline_auc = roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1])

    # At serving time: avg_spend is stale (24h old), add noise
    skew_levels = [0, 0.05, 0.1, 0.2, 0.4]
    results = []
    for skew in skew_levels:
        X_serve = X_te.copy()
        noise = np.random.normal(0, skew * avg_spend[:len(X_te)].std(), len(X_te))
        X_serve[:, 1] += noise  # corrupt avg_spend
        X_serve_s = scaler.transform(X_serve)
        serve_acc = accuracy_score(y_te, model.predict(X_serve_s))
        serve_auc = roc_auc_score(y_te, model.predict_proba(X_serve_s)[:, 1])
        results.append((skew, serve_acc, serve_auc))

    log(f"  {'Noise Level':>12} | {'Accuracy':>10} | {'AUC':>10} | {'Δ Acc':>10}")
    log(f"  {'-' * 50}")
    for skew, acc, auc in results:
        delta = acc - offline_acc
        label = "offline" if skew == 0 else f"{skew:.0%} noise"
        log(f"  {label:>12} | {acc:>10.4f} | {auc:>10.4f} | {delta:>+10.4f}")

    log("""
  Fix: validate feature distributions between training and serving
  pipelines. Log feature stats at both points and alert on divergence.
""")
    return skew_levels, results, offline_acc


def demo_distribution_shift():
    """
    Simulate: model trained on data from month 1-6,
    deployed into months 7-12 where distribution shifted.
    """
    log("=" * 60)
    log("SCENARIO 2: DISTRIBUTION SHIFT")
    log("=" * 60)
    log("""
  Model trained on Jan-Jun data. Deployed Jul onward.
  User demographics shift (younger users join), spending patterns change.
""")

    n_per_month = 500
    months_acc = []
    months_f1 = []

    # Training data: months 1-6
    X_train_all, y_train_all = [], []
    for m in range(1, 7):
        age = np.random.normal(40, 8, n_per_month)
        spend = np.random.exponential(60, n_per_month)
        x = np.column_stack([age, spend])
        y = ((age < 45) & (spend > 40)).astype(int)
        X_train_all.append(x)
        y_train_all.append(y)

    X_train = np.vstack(X_train_all)
    y_train = np.concatenate(y_train_all)

    model = LogisticRegression(random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model.fit(X_train_s, y_train)

    all_months = list(range(1, 13))
    for m in all_months:
        shift = max(0, (m - 6) * 2)  # gradual shift after month 6
        age = np.random.normal(40 - shift, 8 + shift * 0.3, n_per_month)
        spend = np.random.exponential(60 - shift * 2, n_per_month)
        spend = np.clip(spend, 1, None)
        x = np.column_stack([age, spend])
        y = ((age < 45) & (spend > 40)).astype(int)
        x_s = scaler.transform(x)
        pred = model.predict(x_s)
        months_acc.append(accuracy_score(y, pred))
        months_f1.append(f1_score(y, pred, zero_division=0))

    log(f"  {'Month':>6} | {'Accuracy':>10} | {'F1':>10} | {'Status':>15}")
    log(f"  {'-' * 48}")
    for m, (acc, f1) in enumerate(zip(months_acc, months_f1), 1):
        status = "" if m <= 6 else ("⚠ drifting" if acc > 0.65 else "🔴 degraded")
        log(f"  {m:>6} | {acc:>10.4f} | {f1:>10.4f} | {status:>15}")

    log("""
  Fix: monitor prediction distributions weekly. Retrain when
  PSI > 0.2 or accuracy drops > 5% from baseline (see drift_detection/).
""")
    return all_months, months_acc, months_f1


def demo_label_leakage():
    """
    Simulate: training data accidentally includes a feature that
    encodes the label (future information leak).
    """
    log("=" * 60)
    log("SCENARIO 3: LABEL LEAKAGE")
    log("=" * 60)
    log("""
  Predicting "will customer churn next month?".
  Feature "days_since_last_login" is computed at label time, not prediction time.
  At train time it perfectly separates churners. At serve time it doesn't exist yet.
""")

    n = 3000
    tenure = np.random.exponential(24, n)
    support_tickets = np.random.poisson(2, n)
    satisfaction = np.random.uniform(1, 5, n)

    will_churn = ((satisfaction < 2.5) | (support_tickets > 4)).astype(int)

    # Leaked feature: days_since_last_login (only knowable AFTER the churn period)
    days_since_login = np.where(will_churn == 1,
                                np.random.exponential(30, n),
                                np.random.exponential(3, n))

    X_with_leak = np.column_stack([tenure, support_tickets, satisfaction, days_since_login])
    X_no_leak = np.column_stack([tenure, support_tickets, satisfaction])

    results = {}
    for name, X in [("With leakage", X_with_leak), ("Without leakage", X_no_leak)]:
        X_tr, X_te, y_tr, y_te = train_test_split(X, will_churn, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        m = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        m.fit(X_tr_s, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te_s))
        auc = roc_auc_score(y_te, m.predict_proba(X_te_s)[:, 1])
        results[name] = (acc, auc, m.feature_importances_)

    log(f"\n  {'Scenario':<20} | {'Accuracy':>10} | {'AUC':>10}")
    log(f"  {'-' * 45}")
    for name, (acc, auc, _) in results.items():
        log(f"  {name:<20} | {acc:>10.4f} | {auc:>10.4f}")

    # Feature importance shows leakage
    log(f"\n  Feature importances WITH leakage:")
    feat_names_leak = ["tenure", "support_tickets", "satisfaction", "days_since_login*"]
    for fn, imp in zip(feat_names_leak, results["With leakage"][2]):
        bar = "█" * int(imp * 50)
        log(f"    {fn:<22} {imp:.3f} {bar}")

    log(f"\n  Feature importances WITHOUT leakage:")
    feat_names = ["tenure", "support_tickets", "satisfaction"]
    for fn, imp in zip(feat_names, results["Without leakage"][2]):
        bar = "█" * int(imp * 50)
        log(f"    {fn:<22} {imp:.3f} {bar}")

    log("""
  The leaked feature dominates (importance ≈ 0.9). Without it, the model
  is honest — 70-80% AUC from real features.

  Fix: audit every feature — "could I compute this BEFORE making the prediction?"
  If the answer is no, it's leakage. Common culprits:
    • Aggregates computed after the label period
    • IDs that correlate with outcome (e.g., hospital ID → mortality)
    • Features derived from the target variable
""")
    return results


def run_demo():
    log("OFFLINE vs ONLINE GAP — CASE STUDY")
    log("=" * 60)
    log("Your model works offline but fails online. Three reasons why.\n")

    skew_levels, skew_results, offline_acc = demo_feature_skew()
    months, month_acc, month_f1 = demo_distribution_shift()
    leak_results = demo_label_leakage()

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Feature skew
    ax = axes[0]
    noise = [r[0] for r in skew_results]
    accs = [r[1] for r in skew_results]
    ax.plot(noise, accs, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.axhline(y=offline_acc, color="gray", linestyle="--", alpha=0.5, label="Offline baseline")
    ax.set_xlabel("Feature Noise Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("Feature Skew Impact")
    ax.legend()

    # Plot 2: Distribution shift
    ax = axes[1]
    ax.plot(months, month_acc, "s-", color="#3498db", linewidth=2, label="Accuracy")
    ax.plot(months, month_f1, "^-", color="#2ecc71", linewidth=2, label="F1")
    ax.axvline(x=6.5, color="red", linestyle="--", label="Deploy point")
    ax.set_xlabel("Month")
    ax.set_ylabel("Score")
    ax.set_title("Distribution Shift Over Time")
    ax.legend()

    # Plot 3: Label leakage feature importance
    ax = axes[2]
    feat_names = ["tenure", "tickets", "satisf.", "login*"]
    importances = leak_results["With leakage"][2]
    colors = ["#95a5a6", "#95a5a6", "#95a5a6", "#e74c3c"]
    bars = ax.barh(feat_names, importances, color=colors)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Label Leakage Detection")
    ax.annotate("← LEAKED!", xy=(importances[3], 3), fontsize=12, fontweight="bold", color="red")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "offline_online_gap.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/offline_online_gap.png")

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SUMMARY: CLOSING THE OFFLINE-ONLINE GAP")
    log("=" * 60)
    log("""
  ┌──────────────────┬────────────────────────────────────────┐
  │ Problem          │ Fix                                    │
  ├──────────────────┼────────────────────────────────────────┤
  │ Feature skew     │ Log feature stats at train & serve.    │
  │                  │ Alert on distribution divergence.      │
  │                  │ Use a feature store for consistency.   │
  ├──────────────────┼────────────────────────────────────────┤
  │ Distrib. shift   │ Monitor predictions weekly. Retrain    │
  │                  │ when PSI > 0.2 or acc drops > 5%.      │
  ├──────────────────┼────────────────────────────────────────┤
  │ Label leakage    │ Audit every feature: "could I compute  │
  │                  │ this BEFORE making the prediction?"    │
  │                  │ Check feature importances for suspects.│
  ├──────────────────┼────────────────────────────────────────┤
  │ General          │ Shadow mode: run new model in parallel │
  │                  │ with old one, compare outputs before   │
  │                  │ switching traffic.                     │
  └──────────────────┴────────────────────────────────────────┘

  Golden rule: if offline AUC >> online AUC, suspect one of these three.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
