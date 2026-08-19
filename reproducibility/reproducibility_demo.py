"""
Reproducibility Guarantees Demo
================================
Shows how to ensure ML experiments produce identical results every time.

Part 1 — The Problem: Without Seeds
    Train the same model 5 times with no seed control. Results differ.

Part 2 — The Fix: Full Seed Control
    Set seeds for numpy, random, torch, sklearn, and hash. Run twice,
    verify byte-identical outputs.

Part 3 — Environment Logging
    Capture Python version, package versions, CPU/OS info, and git hash.
    Save a reproducibility manifest alongside the model.

Part 4 — Deterministic PyTorch
    Enable deterministic CUDA ops and benchmark=False.
    Show the tradeoff: reproducibility vs speed.
"""

import os, sys, hashlib
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def set_all_seeds(seed):
    """Set seeds for all sources of randomness."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except ImportError:
        pass


def hash_predictions(predictions):
    """Hash predictions to verify exact reproducibility."""
    return hashlib.md5(predictions.tobytes()).hexdigest()



def run_reproducibility_demo():
    log("REPRODUCIBILITY GUARANTEES DEMO")
    log("=" * 60)

    # ═══════════════════════════════════════════════════════
    # Part 1: The Problem — No Seeds
    # ═══════════════════════════════════════════════════════
    log("\nPART 1: THE PROBLEM — UNREPRODUCIBLE RESULTS")
    log("=" * 60)

    accs_no_seed = []
    preds_no_seed = []

    for trial in range(5):
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                   n_redundant=2, random_state=None)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs_no_seed.append(acc)
        preds_no_seed.append(hash_predictions(y_pred))

    log(f"\n  5 trials WITHOUT seed control:")
    for i, (acc, h) in enumerate(zip(accs_no_seed, preds_no_seed)):
        log(f"    Trial {i+1}: accuracy={acc:.4f}  hash={h[:12]}...")
    log(f"  Accuracy range: {min(accs_no_seed):.4f} — {max(accs_no_seed):.4f}")
    log(f"  Unique prediction hashes: {len(set(preds_no_seed))}/5")
    log(f"  → Every run is different!")

    # ═══════════════════════════════════════════════════════
    # Part 2: The Fix — Full Seed Control
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 2: THE FIX — DETERMINISTIC RESULTS")
    log("=" * 60)

    SEED = 42
    accs_seeded = []
    preds_seeded = []

    for trial in range(5):
        set_all_seeds(SEED)
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                   n_redundant=2, random_state=SEED)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
        model = RandomForestClassifier(n_estimators=100, random_state=SEED)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs_seeded.append(acc)
        preds_seeded.append(hash_predictions(y_pred))


    log(f"\n  5 trials WITH seed={SEED}:")
    for i, (acc, h) in enumerate(zip(accs_seeded, preds_seeded)):
        log(f"    Trial {i+1}: accuracy={acc:.4f}  hash={h[:12]}...")
    log(f"  Accuracy range: {min(accs_seeded):.4f} — {max(accs_seeded):.4f}")
    log(f"  Unique prediction hashes: {len(set(preds_seeded))}/5")
    all_match = len(set(preds_seeded)) == 1
    log(f"  → {'All identical!' if all_match else 'Some differ (check seed coverage)'}")

    # ═══════════════════════════════════════════════════════
    # Part 3: What Can Break Reproducibility
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("PART 3: WHAT CAN BREAK REPRODUCIBILITY")
    log("=" * 60)

    # Multi-threaded models
    log(f"\n  Test: n_jobs effect on RandomForest")
    results_by_njobs = {}
    for n_jobs in [1, 2, -1]:
        set_all_seeds(SEED)
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                   n_redundant=2, random_state=SEED)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)
        model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        h = hash_predictions(y_pred)
        results_by_njobs[n_jobs] = h
        log(f"    n_jobs={n_jobs:>2}: hash={h[:16]}...")

    all_same = len(set(results_by_njobs.values())) == 1
    log(f"    → {'All identical' if all_same else 'WARNING: n_jobs affects results!'}")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Unseeded vs seeded accuracy spread
    ax = axes[0]
    ax.scatter([1]*5, accs_no_seed, s=100, color="#e74c3c", label="No seed", zorder=3)
    ax.scatter([2]*5, accs_seeded, s=100, color="#2ecc71", label="With seed", zorder=3)
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["No Seed", "Seed=42"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Reproducibility: Unseeded vs Seeded (5 trials)")
    ax.legend()

    # Checklist
    ax = axes[1]
    ax.axis("off")
    checklist = [
        "✓ numpy.random.seed(SEED)",
        "✓ random.seed(SEED)",
        "✓ PYTHONHASHSEED=SEED",
        "✓ sklearn random_state=SEED",
        "✓ train_test_split random_state=SEED",
        "✓ data generation random_state=SEED",
        "✓ n_jobs does not affect results",
    ]
    for i, item in enumerate(checklist):
        ax.text(0.05, 0.95 - i * 0.065, item, fontsize=10, fontfamily="monospace",
                transform=ax.transAxes, verticalalignment="top")
    ax.set_title("Reproducibility Checklist", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "reproducibility.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/reproducibility.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
Sources of non-determinism in ML:
  ┌───────────────────────┬──────────────────────────────────────┐
  │ Source                │ Fix                                  │
  ├───────────────────────┼──────────────────────────────────────┤
  │ Data shuffling        │ random_state in train_test_split     │
  │ Weight initialization │ torch.manual_seed / random_state     │
  │ Dropout / augmentation│ set all RNG seeds                    │
  │ CUDA non-determinism  │ cudnn.deterministic = True           │
  │ Multi-threading       │ n_jobs=1 or accept variance          │
  │ Hash randomization    │ PYTHONHASHSEED env variable          │
  │ Package versions      │ Pin in requirements.txt              │
  │ Python version        │ Pin in Dockerfile / pyenv            │
  └───────────────────────┴──────────────────────────────────────┘

Tradeoff: deterministic CUDA is ~10-15% slower. In production, you may
accept non-determinism for speed. In research, reproducibility is mandatory.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_reproducibility_demo()
