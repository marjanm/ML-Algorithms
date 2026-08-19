"""
Model Explainability — SHAP & LIME Demo
=========================================
"Why did the model predict this?" — increasingly required in production
(finance, healthcare, legal).

Demonstrates:
  1. SHAP (SHapley Additive exPlanations) — global & local feature importance
  2. LIME (Local Interpretable Model-agnostic Explanations) — local surrogate
  3. Permutation importance — model-agnostic baseline
  4. Comparison: built-in vs permutation vs SHAP importance

Run:
    python explainability_demo.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURE_NAMES = [
    "income", "age", "credit_score", "debt_ratio",
    "employment_years", "num_accounts", "recent_inquiries",
    "noise_1", "noise_2", "noise_3",
]


def run_explainability_demo():
    lines = [
        "=" * 65,
        "  MODEL EXPLAINABILITY  —  SHAP & LIME Demo",
        "=" * 65, "",
    ]

    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, random_state=42,
    )
    X = pd.DataFrame(X, columns=FEATURE_NAMES)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    lines.append(f"  Model: Gradient Boosting (100 trees, depth=4)")
    lines.append(f"  Test accuracy: {acc:.4f}")
    lines.append(f"  Features: {FEATURE_NAMES}")

    # ── 1. Built-in feature importance ──
    lines += ["", "  ── 1. Built-in Feature Importance (impurity-based) ──"]
    builtin_imp = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    for feat, imp in builtin_imp.items():
        bar = "█" * int(imp * 50)
        lines.append(f"    {feat:20s}: {imp:.4f}  {bar}")

    # ── 2. Permutation importance ──
    lines += ["", "  ── 2. Permutation Importance (model-agnostic) ──"]
    perm_result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)
    perm_imp = pd.Series(perm_result.importances_mean, index=FEATURE_NAMES).sort_values(ascending=False)
    for feat, imp in perm_imp.items():
        bar = "█" * int(imp * 100)
        lines.append(f"    {feat:20s}: {imp:.4f}  {bar}")

    # ── 3. SHAP ──
    try:
        import shap
        shap_available = True
    except ImportError:
        shap_available = False
        lines += [
            "", "  ── 3. SHAP ──",
            "    ⚠ shap not installed. Install with: pip3 install shap",
            "    Skipping SHAP analysis.",
        ]

    if shap_available:
        lines += ["", "  ── 3. SHAP — Global Feature Importance ──"]
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(X_test)

        # GradientBoosting returns a list [class0, class1]; take class-1
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1]
        else:
            shap_values = shap_values_raw

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_imp = pd.Series(mean_abs_shap, index=FEATURE_NAMES).sort_values(ascending=False)
        for feat, imp in shap_imp.items():
            bar = "█" * int(imp * 20)
            lines.append(f"    {feat:20s}: {imp:.4f}  {bar}")

        # SHAP summary plot (beeswarm)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=FEATURE_NAMES, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
        lines.append(f"    [saved] → plots/shap_summary.png")

        # SHAP bar plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values, X_test, feature_names=FEATURE_NAMES,
                          plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
        lines.append(f"    [saved] → plots/shap_bar.png")

        # SHAP local explanation — single prediction
        lines += ["", "  ── 4. SHAP — Local Explanation (single prediction) ──"]
        idx = 0
        pred_class = model.predict(X_test.iloc[[idx]])[0]
        pred_proba = model.predict_proba(X_test.iloc[[idx]])[0, 1]
        lines.append(f"    Sample #{idx}: predicted class={pred_class}, probability={pred_proba:.4f}")
        base_val = explainer.expected_value
        if hasattr(base_val, '__len__'):
            base_val = base_val[1] if len(base_val) > 1 else base_val[0]
        lines.append(f"    Base value (avg prediction): {float(base_val):.4f}")

        sample_shap = shap_values[idx]
        top_features = np.argsort(np.abs(sample_shap))[::-1][:5]
        lines.append(f"    Top contributing features:")
        for fi in top_features:
            direction = "↑" if sample_shap[fi] > 0 else "↓"
            lines.append(f"      {FEATURE_NAMES[fi]:20s}: {sample_shap[fi]:+.4f} {direction} "
                        f"(value={X_test.iloc[idx, fi]:.2f})")

        # waterfall plot for single prediction
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[idx],
            base_values=float(base_val),
            data=X_test.iloc[idx].values,
            feature_names=FEATURE_NAMES,
        ), show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
        lines.append(f"    [saved] → plots/shap_waterfall.png")

    # ── 5. LIME ──
    try:
        import lime
        import lime.lime_tabular
        lime_available = True
    except ImportError:
        lime_available = False
        lines += [
            "", "  ── 5. LIME ──",
            "    ⚠ lime not installed. Install with: pip3 install lime",
            "    Skipping LIME analysis.",
        ]

    if lime_available:
        lines += ["", "  ── 5. LIME — Local Surrogate Explanation ──"]
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=FEATURE_NAMES,
            class_names=["Negative", "Positive"],
            mode="classification",
            random_state=42,
        )

        idx = 0
        lime_exp = lime_explainer.explain_instance(
            X_test.iloc[idx].values,
            model.predict_proba,
            num_features=10,
        )

        lines.append(f"    Sample #{idx} explanation:")
        for feat, weight in lime_exp.as_list():
            direction = "↑" if weight > 0 else "↓"
            lines.append(f"      {feat:40s}: {weight:+.4f} {direction}")

        fig = lime_exp.as_pyplot_figure()
        fig.set_size_inches(10, 5)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "lime_explanation.png"), dpi=150, bbox_inches="tight")
        plt.close("all")
        lines.append(f"    [saved] → plots/lime_explanation.png")

    # ── 6. Importance comparison ──
    lines += ["", "  ── 6. Importance Method Comparison ──"]
    comparison = pd.DataFrame({
        "Built-in": builtin_imp,
        "Permutation": perm_imp,
    })
    if shap_available:
        comparison["SHAP"] = shap_imp

    fig, ax = plt.subplots(figsize=(10, 5))
    comparison_norm = comparison.div(comparison.max())
    comparison_norm.sort_values("Built-in", ascending=True).plot.barh(ax=ax)
    ax.set_xlabel("Normalised Importance")
    ax.set_title("Feature Importance: Built-in vs Permutation vs SHAP")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "importance_comparison.png"), dpi=150)
    plt.close()
    lines.append(f"    [saved] → plots/importance_comparison.png")

    lines += [
        "", "  ── Key Takeaways ──",
        "    • Built-in importance (impurity) can be biased toward high-cardinality features",
        "    • Permutation importance is model-agnostic and measures real predictive contribution",
        "    • SHAP gives both global AND local explanations with theoretical guarantees",
        "    • LIME fits a local linear model around each prediction — intuitive but less consistent",
        "    • Noise features should rank low across all methods (sanity check)",
    ]

    lines += ["", "=" * 65]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_explainability_demo()
