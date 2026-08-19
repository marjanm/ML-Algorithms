"""
MLflow Model Registry & Monitoring Demo
=========================================
Extends the basic MLflow tracking demo with:

Part 1 — Model Registry Lifecycle:
    Register models, transition through stages (None → Staging → Production → Archived),
    compare versions, load a production model by name.

Part 2 — Monitoring via MLflow:
    Simulate weekly data drift, log PSI/accuracy as MLflow metrics per "week".
    The MLflow UI becomes the monitoring dashboard.

Part 3 — Automated Retraining Trigger:
    When PSI exceeds threshold, retrain, register new version, promote it.

Run:
    python mlflow_registry_monitoring.py
    Then:  mlflow ui  (from this directory)
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_generators.classification_data import generate_synthetic_data

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_DIR = os.path.join(OUTPUT_DIR, "mlruns")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "output_registry.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def compute_psi(expected, actual, bins=10):
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    exp_counts = np.histogram(expected, bins=breakpoints)[0] + 1
    act_counts = np.histogram(actual, bins=breakpoints)[0] + 1
    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def run_registry_monitoring_demo():
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    client = MlflowClient(tracking_uri=f"file:{MLRUNS_DIR}")

    MODEL_NAME = "fraud_detector"
    PSI_THRESHOLD = 0.25

    log("MLFLOW MODEL REGISTRY & MONITORING DEMO")
    log("=" * 60)

    # ═══════════════════════════════════════════════════════
    # Part 1: Model Registry Lifecycle
    # ═══════════════════════════════════════════════════════
    log("\nPART 1: MODEL REGISTRY LIFECYCLE")
    log("=" * 60)

    mlflow.set_experiment("registry_demo")

    X_train, X_test, y_train, y_test, _ = generate_synthetic_data(
        n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
    )

    # Train V1 — a simple model
    with mlflow.start_run(run_name="v1_baseline") as run_v1:
        model_v1 = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
        model_v1.fit(X_train, y_train)
        acc_v1 = accuracy_score(y_test, model_v1.predict(X_test))
        f1_v1 = f1_score(y_test, model_v1.predict(X_test), average="weighted")
        mlflow.log_params({"n_estimators": 50, "max_depth": 3, "version": "v1"})
        mlflow.log_metrics({"accuracy": acc_v1, "f1": f1_v1})
        mlflow.sklearn.log_model(model_v1, "model")
        run_v1_id = run_v1.info.run_id

    log(f"\n  V1 trained: accuracy={acc_v1:.4f}, f1={f1_v1:.4f}")

    # Train V2 — an improved model
    with mlflow.start_run(run_name="v2_improved") as run_v2:
        model_v2 = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        )
        model_v2.fit(X_train, y_train)
        acc_v2 = accuracy_score(y_test, model_v2.predict(X_test))
        f1_v2 = f1_score(y_test, model_v2.predict(X_test), average="weighted")
        mlflow.log_params({"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05, "version": "v2"})
        mlflow.log_metrics({"accuracy": acc_v2, "f1": f1_v2})
        mlflow.sklearn.log_model(model_v2, "model")
        run_v2_id = run_v2.info.run_id

    log(f"  V2 trained: accuracy={acc_v2:.4f}, f1={f1_v2:.4f}")

    # Register both versions
    try:
        client.create_registered_model(MODEL_NAME)
        log(f"\n  Created registered model: '{MODEL_NAME}'")
    except Exception:
        log(f"\n  Registered model '{MODEL_NAME}' already exists")

    mv1 = client.create_model_version(
        name=MODEL_NAME,
        source=f"{MLRUNS_DIR}/{mlflow.get_experiment_by_name('registry_demo').experiment_id}/{run_v1_id}/artifacts/model",
        run_id=run_v1_id,
    )
    log(f"  Registered V1 as version {mv1.version}")

    mv2 = client.create_model_version(
        name=MODEL_NAME,
        source=f"{MLRUNS_DIR}/{mlflow.get_experiment_by_name('registry_demo').experiment_id}/{run_v2_id}/artifacts/model",
        run_id=run_v2_id,
    )
    log(f"  Registered V2 as version {mv2.version}")

    # Lifecycle transitions
    client.transition_model_version_stage(MODEL_NAME, mv1.version, "Staging")
    log(f"\n  V1 → Staging")

    client.transition_model_version_stage(MODEL_NAME, mv1.version, "Production")
    log(f"  V1 → Production (initial deploy)")

    client.transition_model_version_stage(MODEL_NAME, mv2.version, "Staging")
    log(f"  V2 → Staging (validation)")

    # Compare versions
    log(f"\n  Version comparison:")
    log(f"    V1 (Production): accuracy={acc_v1:.4f}")
    log(f"    V2 (Staging):    accuracy={acc_v2:.4f}")

    if acc_v2 > acc_v1:
        client.transition_model_version_stage(MODEL_NAME, mv2.version, "Production")
        client.transition_model_version_stage(MODEL_NAME, mv1.version, "Archived")
        log(f"\n  V2 beats V1 → V2 promoted to Production, V1 archived")
    else:
        log(f"\n  V1 still better → V2 stays in Staging")

    # Show final state
    log(f"\n  Final registry state:")
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        log(f"    Version {mv.version}: stage={mv.current_stage}")

    # ═══════════════════════════════════════════════════════
    # Part 2: Monitoring via MLflow
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 2: MONITORING VIA MLFLOW (simulated 8 weeks)")
    log("=" * 60)

    mlflow.set_experiment("production_monitoring")

    # Use the production model
    prod_model = model_v2 if acc_v2 > acc_v1 else model_v1
    X_ref = X_train  # reference distribution

    weeks = 8
    weekly_accs = []
    weekly_psis = []
    retrain_week = None

    log(f"\n{'Week':>5} | {'Accuracy':>9} | {'F1':>6} | {'PSI':>8} | Action")
    log("-" * 55)

    for w in range(weeks):
        # Gradually drift the data
        drift = w * 0.4
        X_week, _, y_week_train, y_week_test, _ = generate_synthetic_data(
            n_samples=500, n_features=2, n_informative=2, n_redundant=0,
        )
        X_week[:, 0] += drift
        X_week[:, 1] += drift * 0.3
        # Re-derive labels for drifted data
        rng = np.random.RandomState(42 + w)
        w_all = np.array([1.5, -1.0])
        logits = X_week @ w_all
        y_week = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)

        y_pred = prod_model.predict(X_week)
        acc = accuracy_score(y_week, y_pred)
        f1 = f1_score(y_week, y_pred, average="weighted")
        psi_feat0 = compute_psi(X_ref[:, 0], X_week[:, 0])

        weekly_accs.append(acc)
        weekly_psis.append(psi_feat0)

        action = "✓ OK"
        if psi_feat0 > PSI_THRESHOLD and retrain_week is None:
            action = "✗ RETRAIN TRIGGERED"
            retrain_week = w

        with mlflow.start_run(run_name=f"week_{w+1}_monitoring"):
            mlflow.log_params({"week": w + 1, "drift_magnitude": f"{drift:.1f}"})
            mlflow.log_metrics({
                "accuracy": acc, "f1": f1,
                "psi_feature_0": psi_feat0,
                "prediction_positive_rate": float(y_pred.mean()),
            })

        log(f"{w+1:>5} | {acc:>9.3f} | {f1:>6.3f} | {psi_feat0:>8.4f} | {action}")

    # ═══════════════════════════════════════════════════════
    # Part 3: Automated Retrain & Re-register
    # ═══════════════════════════════════════════════════════
    if retrain_week is not None:
        log(f"\n{'=' * 60}")
        log("PART 3: AUTOMATED RETRAIN & VERSION PROMOTION")
        log("=" * 60)

        mlflow.set_experiment("registry_demo")

        # Retrain on recent drifted data
        drift = retrain_week * 0.4
        X_new, _, y_new_train, y_new_test, _ = generate_synthetic_data(
            n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
        )
        X_new[:, 0] += drift
        X_new[:, 1] += drift * 0.3
        w_all = np.array([1.5, -1.0])
        logits = X_new @ w_all
        y_new = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)

        with mlflow.start_run(run_name="v3_retrained") as run_v3:
            model_v3 = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
            )
            model_v3.fit(X_new, y_new)

            # Evaluate on latest week's data
            X_latest = X_new[:500]
            y_latest = y_new[:500]
            acc_v3 = accuracy_score(y_latest, model_v3.predict(X_latest))

            mlflow.log_params({"n_estimators": 200, "max_depth": 5, "version": "v3", "retrain_reason": "drift"})
            mlflow.log_metrics({"accuracy": acc_v3})
            mlflow.sklearn.log_model(model_v3, "model")
            run_v3_id = run_v3.info.run_id

        mv3 = client.create_model_version(
            name=MODEL_NAME,
            source=f"{MLRUNS_DIR}/{mlflow.get_experiment_by_name('registry_demo').experiment_id}/{run_v3_id}/artifacts/model",
            run_id=run_v3_id,
        )

        # Promote V3 to production
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if mv.current_stage == "Production":
                client.transition_model_version_stage(MODEL_NAME, mv.version, "Archived")
        client.transition_model_version_stage(MODEL_NAME, mv3.version, "Production")

        log(f"\n  V3 retrained on drifted data: accuracy={acc_v3:.4f}")
        log(f"  V3 registered as version {mv3.version} → Production")
        log(f"  Previous production model → Archived")

        log(f"\n  Final registry state:")
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
            log(f"    Version {mv.version}: stage={mv.current_stage}")

    # Summary
    log(f"\n{'=' * 60}")
    log("SUMMARY")
    log("=" * 60)
    log(f"""
  What this demo showed:
  1. Registry lifecycle: None → Staging → Production → Archived
  2. Version comparison: promote better model, archive old one
  3. Monitoring: log drift + accuracy as MLflow metrics per time window
  4. Retraining trigger: PSI > {PSI_THRESHOLD} → retrain → new version → promote

  To explore in the MLflow UI:
    cd {OUTPUT_DIR}
    mlflow ui
    open http://localhost:5000

  You'll see:
    - registry_demo experiment: V1, V2, V3 with version transitions
    - production_monitoring experiment: weekly drift + accuracy metrics
    - Model Registry tab: version history with stage transitions
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"→ Output saved to output_registry.txt")


if __name__ == "__main__":
    run_registry_monitoring_demo()
