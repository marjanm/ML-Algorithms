"""
Structured Error Slicing — Demo
==================================
Overall accuracy hides failures in subgroups. This demo:

1. Train a model on a dataset with demographic features
2. Slice errors by subgroup (age, income, region)
3. Find the worst-performing slices
4. Show how a "93% accurate" model is only 60% on one subgroup
5. Fix: targeted data collection or per-slice thresholds
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


np.random.seed(42)


def create_biased_dataset(n=5000):
    """
    Loan approval dataset where the model performs worse
    on certain demographic slices due to data imbalance.
    """
    age_group = np.random.choice(["18-25", "26-40", "41-60", "60+"], n,
                                  p=[0.1, 0.4, 0.35, 0.15])
    income_level = np.random.choice(["low", "medium", "high"], n,
                                     p=[0.3, 0.5, 0.2])
    region = np.random.choice(["urban", "suburban", "rural"], n,
                               p=[0.5, 0.35, 0.15])

    age_map = {"18-25": 0, "26-40": 1, "41-60": 2, "60+": 3}
    income_map = {"low": 0, "medium": 1, "high": 2}
    region_map = {"urban": 0, "suburban": 1, "rural": 2}

    age_num = np.array([age_map[a] for a in age_group])
    income_num = np.array([income_map[i] for i in income_level])
    region_num = np.array([region_map[r] for r in region])

    credit_score = np.random.normal(650, 80, n)
    loan_amount = np.random.exponential(30000, n)
    employment_years = np.random.exponential(5, n)

    # Label: approval depends on features, but with slice-specific noise
    base_score = (
        0.3 * (credit_score - 500) / 300 +
        0.2 * (income_num / 2) +
        0.15 * (employment_years / 10) -
        0.1 * (loan_amount / 100000)
    )

    # Inject harder-to-predict patterns for underrepresented groups
    noise = np.random.normal(0, 0.1, n)
    noise[age_group == "18-25"] += np.random.normal(0, 0.3, (age_group == "18-25").sum())
    noise[(age_group == "60+") & (income_level == "low")] += 0.4
    noise[(region == "rural") & (income_level == "low")] += np.random.normal(0, 0.25,
        ((region == "rural") & (income_level == "low")).sum())

    y = (base_score + noise > 0.3).astype(int)

    X = np.column_stack([credit_score, loan_amount, employment_years,
                         age_num, income_num, region_num])
    feature_names = ["credit_score", "loan_amount", "employment_years",
                     "age_group", "income_level", "region"]

    slices = {
        "age": age_group,
        "income": income_level,
        "region": region,
    }
    return X, y, feature_names, slices


def compute_slice_metrics(y_true, y_pred, slice_labels):
    """Compute metrics for each slice value."""
    results = {}
    unique_vals = np.unique(slice_labels)
    for val in unique_vals:
        mask = slice_labels == val
        n = mask.sum()
        if n < 10:
            continue
        results[val] = {
            "n": n,
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "f1": f1_score(y_true[mask], y_pred[mask], zero_division=0),
            "precision": precision_score(y_true[mask], y_pred[mask], zero_division=0),
            "recall": recall_score(y_true[mask], y_pred[mask], zero_division=0),
            "positive_rate": y_true[mask].mean(),
        }
    return results


def run_demo():
    log("STRUCTURED ERROR SLICING — DEMO")
    log("=" * 60)

    X, y, feature_names, slices = create_biased_dataset(5000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Split slice labels accordingly
    slices_test = {}
    _, test_idx = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42)
    for name, labels in slices.items():
        slices_test[name] = labels[test_idx]

    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    overall_f1 = f1_score(y_test, y_pred)
    log(f"\n  Overall accuracy: {overall_acc:.4f}")
    log(f"  Overall F1:       {overall_f1:.4f}")

    # ═══════════════════════════════════════════════════════
    # Slice Analysis
    # ═══════════════════════════════════════════════════════
    all_slice_results = {}

    for slice_name in ["age", "income", "region"]:
        log(f"\n{'=' * 60}")
        log(f"SLICE: {slice_name.upper()}")
        log("=" * 60)

        results = compute_slice_metrics(y_test, y_pred, slices_test[slice_name])
        all_slice_results[slice_name] = results

        log(f"\n  {'Value':<12} {'N':>6} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Pos%':>8}")
        log(f"  {'-' * 62}")
        for val in sorted(results.keys()):
            r = results[val]
            flag = " ◄ UNDERPERFORMING" if r["accuracy"] < overall_acc - 0.05 else ""
            log(f"  {val:<12} {r['n']:>6} {r['accuracy']:>8.4f} {r['f1']:>8.4f} "
                f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['positive_rate']:>8.2f}{flag}")

    # ═══════════════════════════════════════════════════════
    # Intersection slicing (2D)
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("INTERSECTION SLICING (age × income)")
    log("=" * 60)

    log(f"\n  {'Age × Income':<22} {'N':>6} {'Acc':>8} {'F1':>8}")
    log(f"  {'-' * 42}")

    worst_slice = None
    worst_acc = 1.0
    intersection_data = []

    for age_val in sorted(np.unique(slices_test["age"])):
        for inc_val in sorted(np.unique(slices_test["income"])):
            mask = (slices_test["age"] == age_val) & (slices_test["income"] == inc_val)
            n = mask.sum()
            if n < 10:
                continue
            acc = accuracy_score(y_test[mask], y_pred[mask])
            f1 = f1_score(y_test[mask], y_pred[mask], zero_division=0)
            flag = " ◄ WORST" if acc < worst_acc else ""
            if acc < worst_acc:
                worst_acc = acc
                worst_slice = f"{age_val} × {inc_val}"
            log(f"  {age_val + ' × ' + inc_val:<22} {n:>6} {acc:>8.4f} {f1:>8.4f}{flag}")
            intersection_data.append((age_val, inc_val, n, acc, f1))

    log(f"\n  → Worst slice: {worst_slice} (accuracy: {worst_acc:.4f})")
    log(f"  → Overall accuracy: {overall_acc:.4f}")
    log(f"  → Gap: {overall_acc - worst_acc:.4f}")

    # ═══════════════════════════════════════════════════════
    # Error categorization
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("ERROR CATEGORIZATION")
    log("=" * 60)

    errors = y_test != y_pred
    fp = (y_pred == 1) & (y_test == 0)
    fn = (y_pred == 0) & (y_test == 1)

    log(f"\n  Total errors: {errors.sum()} / {len(y_test)} ({errors.mean():.1%})")
    log(f"  False positives (approved but shouldn't): {fp.sum()}")
    log(f"  False negatives (rejected but should approve): {fn.sum()}")

    log(f"\n  False negatives by age group:")
    for val in sorted(np.unique(slices_test["age"])):
        mask = slices_test["age"] == val
        fn_rate = fn[mask].sum() / mask.sum() if mask.sum() > 0 else 0
        bar = "█" * int(fn_rate * 50)
        log(f"    {val:<8} {fn_rate:.3f} {bar}")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy by age group
    ax = axes[0, 0]
    age_results = all_slice_results["age"]
    vals = sorted(age_results.keys())
    accs = [age_results[v]["accuracy"] for v in vals]
    colors = ["#e74c3c" if a < overall_acc - 0.05 else "#2ecc71" for a in accs]
    ax.bar(vals, accs, color=colors)
    ax.axhline(y=overall_acc, color="blue", linestyle="--", label=f"Overall ({overall_acc:.3f})")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Age Group")
    ax.legend()

    # Plot 2: Accuracy by income level
    ax = axes[0, 1]
    inc_results = all_slice_results["income"]
    vals = sorted(inc_results.keys())
    accs = [inc_results[v]["accuracy"] for v in vals]
    sizes = [inc_results[v]["n"] for v in vals]
    colors = ["#e74c3c" if a < overall_acc - 0.05 else "#2ecc71" for a in accs]
    bars = ax.bar(vals, accs, color=colors)
    for b, s in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"n={s}",
               ha="center", fontsize=9)
    ax.axhline(y=overall_acc, color="blue", linestyle="--", label=f"Overall ({overall_acc:.3f})")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Income Level")
    ax.legend()

    # Plot 3: Intersection heatmap
    ax = axes[1, 0]
    age_vals = sorted(np.unique(slices_test["age"]))
    inc_vals = sorted(np.unique(slices_test["income"]))
    heatmap = np.zeros((len(age_vals), len(inc_vals)))
    for d in intersection_data:
        i = age_vals.index(d[0])
        j = inc_vals.index(d[1])
        heatmap[i, j] = d[3]
    im = ax.imshow(heatmap, cmap="RdYlGn", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(len(inc_vals)))
    ax.set_xticklabels(inc_vals)
    ax.set_yticks(range(len(age_vals)))
    ax.set_yticklabels(age_vals)
    ax.set_xlabel("Income")
    ax.set_ylabel("Age Group")
    ax.set_title("Accuracy Heatmap (Age × Income)")
    for i in range(len(age_vals)):
        for j in range(len(inc_vals)):
            ax.text(j, i, f"{heatmap[i,j]:.2f}", ha="center", va="center", fontsize=10,
                   color="white" if heatmap[i,j] < 0.75 else "black")
    plt.colorbar(im, ax=ax)

    # Plot 4: Error distribution
    ax = axes[1, 1]
    categories = ["FP (approve wrong)", "FN (reject wrong)"]
    for slice_name, color in [("age", "#3498db")]:
        for val in sorted(np.unique(slices_test[slice_name])):
            mask = slices_test[slice_name] == val
            fp_rate = fp[mask].sum() / mask.sum()
            fn_rate = fn[mask].sum() / mask.sum()
            ax.scatter(fp_rate, fn_rate, s=mask.sum() / 2, label=val, alpha=0.7)
            ax.annotate(val, (fp_rate, fn_rate), fontsize=9)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.set_title("Error Types by Age Group (size = N)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "error_slicing.png"), dpi=150)
    plt.close()
    log(f"\n→ Plot saved: plots/error_slicing.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
  Why slice?
    Overall metrics hide subgroup failures. A model with 93% accuracy
    might be 60% accurate on the smallest demographic slice — the group
    that needs the model most.

  Workflow:
    1. Pick slicing dimensions (demographics, feature bins, data source)
    2. Compute per-slice metrics (accuracy, F1, FP/FN rates)
    3. Flag underperforming slices (> 5% below overall)
    4. Investigate: is it data quality? data quantity? distribution shift?
    5. Fix: targeted data collection, slice-specific thresholds, or
       separate models for underperforming slices

  Tools:
    • Pandas groupby → manual slicing
    • Scikit-learn: no built-in slicing
    • Google's Slicing Intelligently (slice-finder)
    • Microsoft Responsible AI Toolbox
    • Evidently AI — automated drift & slice reports

  Interview tip: "How would you debug a model that's 95% accurate
  but users are complaining?" → Error slicing is the first answer.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
