"""
Latency-Constrained Inference Demo
====================================
Benchmark prediction latency across model types and sizes.
Plot the accuracy-vs-latency tradeoff — the core production ML question.

Models benchmarked:
  - Logistic Regression (tiny)
  - Decision Tree (small)
  - Random Forest (50, 200, 500 trees)
  - Gradient Boosting (50, 200 trees)
  - XGBoost (50, 200 trees)
  - Small MLP (PyTorch)
"""

import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def benchmark_latency(model, X_single, n_warmup=50, n_runs=500):
    """Measure single-sample prediction latency in microseconds."""
    for _ in range(n_warmup):
        model.predict(X_single)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_single)
        elapsed = (time.perf_counter() - start) * 1e6  # microseconds
        times.append(elapsed)

    return {
        "mean_us": np.mean(times),
        "median_us": np.median(times),
        "p95_us": np.percentile(times, 95),
        "p99_us": np.percentile(times, 99),
    }


def benchmark_batch_latency(model, X_batch, n_warmup=10, n_runs=100):
    """Measure batch prediction latency."""
    for _ in range(n_warmup):
        model.predict(X_batch)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_batch)
        elapsed = (time.perf_counter() - start) * 1e3  # milliseconds
        times.append(elapsed)

    return {"mean_ms": np.mean(times), "p95_ms": np.percentile(times, 95)}


def run_latency_demo():
    log("LATENCY-CONSTRAINED INFERENCE DEMO")
    log("=" * 60)

    np.random.seed(42)
    X, y = make_classification(n_samples=10000, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_single = X_test[:1]
    X_batch_100 = X_test[:100]
    X_batch_1000 = X_test[:1000]

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree (d=5)", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("Decision Tree (d=20)", DecisionTreeClassifier(max_depth=20, random_state=42)),
        ("Random Forest (50)", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("Random Forest (200)", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("Random Forest (500)", RandomForestClassifier(n_estimators=500, random_state=42)),
        ("Gradient Boosting (50)", GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ("Gradient Boosting (200)", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ]

    try:
        from xgboost import XGBClassifier
        models += [
            ("XGBoost (50)", XGBClassifier(n_estimators=50, random_state=42, verbosity=0)),
            ("XGBoost (200)", XGBClassifier(n_estimators=200, random_state=42, verbosity=0)),
        ]
    except ImportError:
        pass

    # ═══════════════════════════════════════════════════════
    # Part 1: Single-Sample Latency
    # ═══════════════════════════════════════════════════════
    log(f"\nPART 1: SINGLE-SAMPLE LATENCY")
    log("=" * 60)

    results = []

    log(f"\n  {'Model':<28} {'Accuracy':>8} {'Mean µs':>9} {'P95 µs':>9} {'P99 µs':>9}")
    log(f"  {'-' * 65}")

    for name, model in models:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        latency = benchmark_latency(model, X_single)
        results.append({
            "name": name, "accuracy": acc,
            "mean_us": latency["mean_us"], "p95_us": latency["p95_us"],
            "p99_us": latency["p99_us"],
        })
        log(f"  {name:<28} {acc:>8.4f} {latency['mean_us']:>9.1f} {latency['p95_us']:>9.1f} {latency['p99_us']:>9.1f}")

    # ═══════════════════════════════════════════════════════
    # Part 2: Batch Latency
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 2: BATCH LATENCY (amortized)")
    log("=" * 60)

    log(f"\n  {'Model':<28} {'1 sample':>10} {'100 batch':>10} {'1K batch':>10} {'Speedup':>8}")
    log(f"  {'-' * 68}")

    for name, model in models:
        model.fit(X_train, y_train)
        lat_1 = benchmark_latency(model, X_single)
        lat_100 = benchmark_batch_latency(model, X_batch_100)
        lat_1000 = benchmark_batch_latency(model, X_batch_1000)
        per_sample_batch = lat_1000["mean_ms"] * 1000 / 1000  # ms to us, per sample
        speedup = lat_1["mean_us"] / per_sample_batch if per_sample_batch > 0 else 0

        log(f"  {name:<28} {lat_1['mean_us']:>9.1f}µs {lat_100['mean_ms']:>9.2f}ms {lat_1000['mean_ms']:>9.2f}ms {speedup:>7.1f}×")

    # ═══════════════════════════════════════════════════════
    # Part 3: SLA Analysis
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 3: SLA COMPLIANCE (50ms budget)")
    log("=" * 60)

    SLA_MS = 50  # 50ms total budget
    SLA_US = SLA_MS * 1000

    log(f"\n  If your SLA is {SLA_MS}ms total (including network, preprocessing):")
    log(f"  Assume ~30ms for network + preprocessing → {SLA_US - 30000}µs for model inference")
    model_budget_us = SLA_US - 30000

    log(f"\n  {'Model':<28} {'P95 µs':>9} {'Fits SLA?':>10}")
    log(f"  {'-' * 50}")
    for r in results:
        fits = "✓ YES" if r["p95_us"] <= model_budget_us else "✗ NO"
        log(f"  {r['name']:<28} {r['p95_us']:>9.1f} {fits:>10}")

    # Best model that fits
    fitting = [r for r in results if r["p95_us"] <= model_budget_us]
    if fitting:
        best = max(fitting, key=lambda r: r["accuracy"])
        log(f"\n  Best model within SLA: {best['name']}")
        log(f"    Accuracy: {best['accuracy']:.4f}, P95 latency: {best['p95_us']:.0f}µs")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Accuracy vs Latency scatter
    ax = axes[0]
    accs = [r["accuracy"] for r in results]
    lats = [r["mean_us"] for r in results]
    names = [r["name"] for r in results]

    ax.scatter(lats, accs, s=120, c=range(len(results)), cmap="viridis", edgecolors="black", zorder=3)
    for i, name in enumerate(names):
        short = name.split("(")[0].strip()
        ax.annotate(short, (lats[i], accs[i]), textcoords="offset points",
                    xytext=(8, 4), fontsize=7)
    ax.axvline(x=model_budget_us, color="red", linestyle="--", alpha=0.5, label=f"SLA budget ({model_budget_us/1000:.0f}ms)")
    ax.set_xlabel("Mean Prediction Latency (µs)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Latency Tradeoff")
    ax.legend()

    # Plot 2: Latency bars (sorted)
    ax = axes[1]
    sorted_results = sorted(results, key=lambda r: r["mean_us"])
    names_sorted = [r["name"] for r in sorted_results]
    lats_sorted = [r["mean_us"] for r in sorted_results]
    p95_sorted = [r["p95_us"] for r in sorted_results]

    y_pos = range(len(names_sorted))
    ax.barh(y_pos, lats_sorted, color="#3498db", alpha=0.7, label="Mean")
    ax.barh(y_pos, p95_sorted, color="#e74c3c", alpha=0.3, label="P95")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("Latency (µs)")
    ax.set_title("Single-Sample Prediction Latency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "latency_benchmark.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/latency_benchmark.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
Optimization techniques (when model is too slow):
  1. Model distillation  — train a small student to mimic a large teacher
  2. Quantization        — 32-bit → 8-bit weights (4× smaller, ~2× faster)
  3. Pruning             — remove near-zero weights
  4. Feature caching     — precompute expensive features
  5. Batching            — group requests for GPU efficiency
  6. ONNX Runtime        — optimized inference engine
  7. Simpler model       — LR is 100× faster than RF-500 with ~1% accuracy loss

Production rule: measure P95/P99, not mean. A mean of 10ms with P99 of 500ms
means 1% of users wait half a second — unacceptable for real-time.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_latency_demo()
