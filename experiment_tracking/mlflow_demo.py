"""
MLflow Experiment Tracking — Demo
====================================
Trains multiple models, logs everything (params, metrics, artifacts,
plots, saved models) to MLflow, and prints a comparison table.

After running:
    cd experiment_tracking
    mlflow ui
    open http://localhost:5000

Run:
    python mlflow_demo.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_generators.classification_data import generate_synthetic_data

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MLRUNS_DIR = os.path.join(OUTPUT_DIR, "mlruns")


def run_mlflow_demo():
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("model_comparison")

    X_train, X_test, y_train, y_test, _ = generate_synthetic_data(
        n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
    )

    models = [
        (
            "Logistic Regression",
            LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        ),
        (
            "Logistic Regression (strong reg)",
            LogisticRegression(C=0.01, max_iter=1000, random_state=42),
            {"C": 0.01, "max_iter": 1000, "solver": "lbfgs"},
        ),
        (
            "Random Forest (50 trees)",
            RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
            {"n_estimators": 50, "max_depth": 5},
        ),
        (
            "Random Forest (200 trees)",
            RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            {"n_estimators": 200, "max_depth": 10},
        ),
        (
            "Random Forest (deep)",
            RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
            {"n_estimators": 100, "max_depth": "None"},
        ),
    ]

    try:
        from xgboost import XGBClassifier
        models += [
            (
                "XGBoost (lr=0.1)",
                XGBClassifier(n_estimators=100, learning_rate=0.1,
                              max_depth=5, eval_metric="logloss",
                              random_state=42, verbosity=0),
                {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
            ),
            (
                "XGBoost (lr=0.01)",
                XGBClassifier(n_estimators=300, learning_rate=0.01,
                              max_depth=3, eval_metric="logloss",
                              random_state=42, verbosity=0),
                {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 3},
            ),
        ]
    except ImportError:
        pass

    lines = [
        "=" * 75,
        "  MLFLOW EXPERIMENT TRACKING  —  Demo",
        "=" * 75,
        f"  Tracking URI : file:{MLRUNS_DIR}",
        f"  Experiment   : model_comparison",
        f"  Models       : {len(models)}",
        f"  Train size   : {len(y_train)}",
        f"  Test size    : {len(y_test)}",
        "", "  Running experiments...", "",
    ]

    results = []

    for name, model, params in models:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_params(params)

            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "train_time_sec": train_time,
            }
            mlflow.log_metrics(metrics)

            # log confusion matrix as artifact
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(cm, display_labels=["0", "1"]).plot(ax=ax)
            ax.set_title(f"{name}")
            plt.tight_layout()
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close()

            # log the trained model
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id
            results.append((name, metrics, run_id))
            lines.append(
                f"  {name:35s} | acc={metrics['accuracy']:.4f} | "
                f"f1={metrics['f1']:.4f} | auc={metrics['roc_auc']:.4f} | "
                f"time={train_time:.3f}s | run_id={run_id[:8]}"
            )

    # comparison table
    lines += [
        "", "=" * 75,
        "  Comparison (sorted by ROC AUC):",
        "=" * 75,
        f"  {'Model':35s} | {'Acc':>6s} | {'F1':>6s} | {'AUC':>6s} | {'Time':>7s}",
        f"  {'-'*35}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}",
    ]
    for name, m, _ in sorted(results, key=lambda x: -x[1]["roc_auc"]):
        lines.append(
            f"  {name:35s} | {m['accuracy']:6.4f} | {m['f1']:6.4f} | "
            f"{m['roc_auc']:6.4f} | {m['train_time_sec']:6.3f}s"
        )

    lines += [
        "", "=" * 75,
        "  Next steps:",
        f"    cd {OUTPUT_DIR}",
        f"    mlflow ui",
        "    open http://localhost:5000",
        "",
        "  What you'll see in the UI:",
        "    - All runs listed with params and metrics",
        "    - Click any run to see artifacts (confusion matrix, saved model)",
        "    - Use 'Compare' to overlay metrics across runs",
        "    - Each model is reloadable: mlflow.sklearn.load_model(run_uri)",
        "=" * 75,
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_mlflow_demo()
