"""
XGBoost Classifier — fully parameterised
==========================================
Exposes every major XGBClassifier hyper-parameter so you can experiment
with each one.  Returns a trained model plus a results dict.
"""

import os
from typing import Optional

import time
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    # --- booster type ---
    booster: str = "gbtree",              # which booster to use: "gbtree" (tree), "gblinear" (linear), "dart" (dropout trees)
    # --- tree structure ---
    n_estimators: int = 300,              # how many boosting rounds (trees) to build
    max_depth: int = 8,                   # max depth per tree; deeper = more complex, more overfit-prone
    max_leaves: int = 0,                  # max leaf nodes per tree; 0 = no limit (use with lossguide)
    max_bin: int = 256,                   # max bins for continuous features when using "hist" tree method
    grow_policy: str = "depthwise",       # "depthwise" = split level-by-level; "lossguide" = split leaf with highest loss first
    tree_method: str = "hist",            # how to build trees: "exact" (brute-force), "approx" (quantile sketch), "hist" (histogram-based, fastest)
    # --- learning ---
    learning_rate: float = 0.05,          # shrinks each tree's contribution; lower = slower but often better generalisation
    gamma: float = 0.1,                   # minimum loss reduction required to make a split; higher = more conservative
    min_child_weight: float = 3,          # minimum sum of instance weights in a child node; prevents splits on tiny groups
    max_delta_step: float = 0,            # caps the weight update per tree; useful for imbalanced classes (0 = no cap)
    # --- regularisation ---
    reg_alpha: float = 0.1,              # L1 penalty (lasso) — pushes small weights to zero, encourages sparsity
    reg_lambda: float = 1.0,             # L2 penalty (ridge) — penalises large weights, smooths the model
    # --- sub-sampling ---
    subsample: float = 0.8,              # fraction of training rows sampled per tree (1.0 = use all rows)
    colsample_bytree: float = 0.8,       # fraction of features sampled when building each tree
    colsample_bylevel: float = 1.0,      # fraction of features sampled at each depth level
    colsample_bynode: float = 1.0,       # fraction of features sampled at each split node
    # --- objective & eval ---
    objective: str = "binary:logistic",   # loss function to optimise; "binary:logistic" outputs probabilities for 2-class
    eval_metric: str = "logloss",         # metric to track on eval set; "logloss" = log-likelihood loss
    # --- scale / imbalance ---
    scale_pos_weight: float = 1.0,        # ratio of negative/positive samples; >1 upweights the minority class
    base_score: float = 0.5,              # initial prediction (global bias); 0.5 = neutral starting point
    # --- DART-specific (ignored if booster != "dart") ---
    sample_type: str = "uniform",         # how dropped trees are sampled: "uniform" or "weighted" by their contributions
    normalize_type: str = "tree",         # how new tree weights are normalised after dropout: "tree" or "forest"
    rate_drop: float = 0.1,               # probability of dropping a tree during each boosting round
    one_drop: int = 0,                    # if 1, guarantee at least one tree is always dropped
    skip_drop: float = 0.5,              # probability of skipping dropout entirely for a round (keeps full ensemble)
    # --- misc ---
    n_jobs: int = -1,                     # CPU cores to use; -1 = all available cores
    random_state: int = 42,               # seed for reproducibility
    verbosity: int = 0,                   # 0 = silent, 1 = warnings, 2 = info, 3 = debug
    importance_type: str = "gain",        # how to measure feature importance: "gain", "weight" (split count), "cover" (sample count)
    early_stopping_rounds: Optional[int] = 20,  # stop training if eval metric doesn't improve for N rounds
):
    """Train an XGBoost model and return (model, results_dict)."""

    params = dict(
        booster=booster,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_leaves=max_leaves,
        max_bin=max_bin,
        grow_policy=grow_policy,
        tree_method=tree_method,
        learning_rate=learning_rate,
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_delta_step=max_delta_step,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        objective=objective,
        eval_metric=eval_metric,
        scale_pos_weight=scale_pos_weight,
        base_score=base_score,
        n_jobs=n_jobs,
        random_state=random_state,
        verbosity=verbosity,
        importance_type=importance_type,
    )

    if booster == "dart":
        params.update(
            sample_type=sample_type,
            normalize_type=normalize_type,
            rate_drop=rate_drop,
            one_drop=one_drop,
            skip_drop=skip_drop,
        )

    model = XGBClassifier(**params)

    fit_kwargs = {}
    if early_stopping_rounds is not None:
        fit_kwargs["eval_set"] = [(X_test, y_test)]
        fit_kwargs["verbose"] = False

    start = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    predict_time = time.perf_counter() - start

    results = {
        "model_name": "XGBoost",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "train_time_sec": train_time,
        "predict_time_sec": predict_time,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_importances": model.feature_importances_,
    }

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        results["best_iteration"] = best_iter

    lines = [
        "=" * 50,
        "  XGBOOST  —  Results",
        "=" * 50,
        f"  Accuracy   : {results['accuracy']:.4f}",
        f"  Precision  : {results['precision']:.4f}",
        f"  Recall     : {results['recall']:.4f}",
        f"  F1 Score   : {results['f1']:.4f}",
        f"  ROC AUC    : {results['roc_auc']:.4f}",
        f"  Train time : {train_time:.3f}s",
        f"  Predict    : {predict_time:.4f}s",
    ]
    if best_iter is not None:
        lines.append(f"  Best iter  : {best_iter}")
    lines += ["=" * 50, "", "Classification Report:", results["classification_report"]]

    output_text = "\n".join(lines)
    print("\n" + output_text)

    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
    print(f"  [saved] {out_path}")

    return model, results
