"""
Causal Inference Demo
=====================
Shows the difference between correlation and causation with concrete examples.

Part 1 — Simpson's Paradox:
    A treatment appears harmful overall but is beneficial within every subgroup.
    The confounder (disease severity) reverses the effect when not accounted for.

Part 2 — Propensity Score Matching:
    Observational data where treatment assignment is biased (sicker patients get
    the drug more often).  Naive comparison is confounded.  Propensity scores
    rebalance the groups so we recover the true treatment effect.

Part 3 — Instrumental Variables (conceptual):
    When you can't observe the confounder, use an instrument correlated with
    treatment but not with outcome except through treatment.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


def simpsons_paradox():
    """Demonstrate Simpson's Paradox with a drug trial."""
    log("=" * 60)
    log("PART 1: SIMPSON'S PARADOX")
    log("=" * 60)

    # Mild cases: drug works well
    mild_treat_n, mild_treat_recover = 100, 85   # 85% recovery
    mild_ctrl_n, mild_ctrl_recover   = 400, 300   # 75% recovery

    # Severe cases: drug also works well
    sev_treat_n, sev_treat_recover = 400, 200     # 50% recovery
    sev_ctrl_n, sev_ctrl_recover   = 100, 30      # 30% recovery

    # But overall (Simpson's reversal!)
    total_treat_n = mild_treat_n + sev_treat_n      # 500
    total_treat_r = mild_treat_recover + sev_treat_recover  # 285  → 57%
    total_ctrl_n  = mild_ctrl_n + sev_ctrl_n        # 500
    total_ctrl_r  = mild_ctrl_recover + sev_ctrl_recover    # 330  → 66%

    log(f"\nOverall (MISLEADING):")
    log(f"  Drug group:    {total_treat_r}/{total_treat_n} = {total_treat_r/total_treat_n:.0%} recovered")
    log(f"  Control group: {total_ctrl_r}/{total_ctrl_n}  = {total_ctrl_r/total_ctrl_n:.0%} recovered")
    log(f"  → Drug looks WORSE! (57% vs 66%)")

    log(f"\nStratified by severity (CORRECT):")
    log(f"  Mild   — Drug: {mild_treat_recover}/{mild_treat_n} = {mild_treat_recover/mild_treat_n:.0%}  |  Control: {mild_ctrl_recover}/{mild_ctrl_n} = {mild_ctrl_recover/mild_ctrl_n:.0%}")
    log(f"  Severe — Drug: {sev_treat_recover}/{sev_treat_n} = {sev_treat_recover/sev_treat_n:.0%}  |  Control: {sev_ctrl_recover}/{sev_ctrl_n} = {sev_ctrl_recover/sev_ctrl_n:.0%}")
    log(f"  → Drug is BETTER in BOTH subgroups!")

    log(f"\nWhy the reversal?")
    log(f"  Severe patients were more likely to get the drug ({sev_treat_n} vs {sev_ctrl_n})")
    log(f"  Severity is a confounder — it affects both treatment assignment and outcome")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall
    ax = axes[0]
    rates = [total_treat_r/total_treat_n, total_ctrl_r/total_ctrl_n]
    bars = ax.bar(["Drug", "Control"], rates, color=["#e74c3c", "#3498db"], width=0.5)
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Overall (Misleading)")
    ax.set_ylim(0, 1)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width()/2, r + 0.02, f"{r:.0%}", ha="center", fontweight="bold")

    # Stratified
    ax = axes[1]
    x = np.arange(2)
    w = 0.3
    drug_rates = [mild_treat_recover/mild_treat_n, sev_treat_recover/sev_treat_n]
    ctrl_rates = [mild_ctrl_recover/mild_ctrl_n, sev_ctrl_recover/sev_ctrl_n]
    b1 = ax.bar(x - w/2, drug_rates, w, label="Drug", color="#e74c3c")
    b2 = ax.bar(x + w/2, ctrl_rates, w, label="Control", color="#3498db")
    ax.set_xticks(x)
    ax.set_xticklabels(["Mild", "Severe"])
    ax.set_ylabel("Recovery Rate")
    ax.set_title("Stratified by Severity (Correct)")
    ax.set_ylim(0, 1)
    ax.legend()
    for bars_group in [b1, b2]:
        for b in bars_group:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                    f"{b.get_height():.0%}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "simpsons_paradox.png"), dpi=150)
    plt.close()
    log("\n→ Plot saved: plots/simpsons_paradox.png")


def propensity_score_matching():
    """Show how propensity scores correct for confounded treatment assignment."""
    log("\n" + "=" * 60)
    log("PART 2: PROPENSITY SCORE MATCHING")
    log("=" * 60)

    np.random.seed(42)
    n = 2000

    # Confounders: age and severity (both affect treatment AND outcome)
    age = np.random.normal(50, 15, n).clip(20, 80)
    severity = np.random.normal(5, 2, n).clip(0, 10)

    # Treatment assignment is biased: sicker/older patients more likely to get drug
    treat_prob = 1 / (1 + np.exp(-(0.03 * (age - 50) + 0.3 * (severity - 5))))
    treatment = (np.random.random(n) < treat_prob).astype(int)

    # True causal effect of treatment: +15 percentage points recovery
    TRUE_EFFECT = 0.15
    base_recovery_prob = 1 / (1 + np.exp(0.05 * (age - 50) + 0.4 * (severity - 5)))
    outcome_prob = base_recovery_prob + treatment * TRUE_EFFECT
    outcome = (np.random.random(n) < outcome_prob.clip(0, 1)).astype(int)

    # Naive comparison (confounded)
    naive_treat = outcome[treatment == 1].mean()
    naive_ctrl = outcome[treatment == 0].mean()
    naive_effect = naive_treat - naive_ctrl

    log(f"\nTrue causal effect: +{TRUE_EFFECT:.0%}")
    log(f"\nNaive comparison (BIASED):")
    log(f"  Treated:   {naive_treat:.3f}")
    log(f"  Control:   {naive_ctrl:.3f}")
    log(f"  Estimated effect: {naive_effect:+.3f}  (true: +{TRUE_EFFECT:.3f})")
    log(f"  → Bias because sicker patients get treatment more often")

    # Step 1: Estimate propensity scores
    X = np.column_stack([age, severity])
    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(X, treatment)
    propensity = ps_model.predict_proba(X)[:, 1]

    # Step 2: Inverse Propensity Weighting (IPW)
    # weight treated by 1/p(treat), control by 1/(1-p(treat))
    weights = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))
    weights = weights / weights.sum()  # normalize

    ipw_effect = (
        np.sum(weights * treatment * outcome) / np.sum(weights * treatment)
        - np.sum(weights * (1 - treatment) * outcome) / np.sum(weights * (1 - treatment))
    )

    log(f"\nInverse Propensity Weighting (IPW):")
    log(f"  Estimated effect: {ipw_effect:+.3f}  (true: +{TRUE_EFFECT:.3f})")

    # Step 3: Stratified matching (bin by propensity quintiles)
    quintiles = np.percentile(propensity, [20, 40, 60, 80])
    bins = np.digitize(propensity, quintiles)
    strat_effects = []
    for b in range(5):
        mask = bins == b
        if treatment[mask].sum() > 5 and (1 - treatment[mask]).sum() > 5:
            eff = outcome[mask & (treatment == 1)].mean() - outcome[mask & (treatment == 0)].mean()
            strat_effects.append(eff)
    strat_effect = np.mean(strat_effects)

    log(f"\nStratified by propensity quintiles:")
    log(f"  Estimated effect: {strat_effect:+.3f}  (true: +{TRUE_EFFECT:.3f})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Propensity score distributions
    ax = axes[0]
    ax.hist(propensity[treatment == 1], bins=30, alpha=0.6, label="Treated", color="#e74c3c", density=True)
    ax.hist(propensity[treatment == 0], bins=30, alpha=0.6, label="Control", color="#3498db", density=True)
    ax.set_xlabel("Propensity Score P(treatment)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Distributions")
    ax.legend()

    # Confounder balance before/after
    ax = axes[1]
    methods = ["Naive", "IPW", "Stratified", "True"]
    effects = [naive_effect, ipw_effect, strat_effect, TRUE_EFFECT]
    colors = ["#e74c3c", "#2ecc71", "#f39c12", "#3498db"]
    bars = ax.bar(methods, effects, color=colors, width=0.5)
    ax.axhline(y=TRUE_EFFECT, color="#3498db", linestyle="--", alpha=0.7, label=f"True effect ({TRUE_EFFECT:+.2f})")
    ax.set_ylabel("Estimated Treatment Effect")
    ax.set_title("Effect Estimates: Naive vs Corrected")
    ax.legend()
    for b, e in zip(bars, effects):
        ax.text(b.get_x() + b.get_width()/2, e + 0.005, f"{e:+.3f}", ha="center", fontsize=9)

    # Severity vs treatment (showing the confounding)
    ax = axes[2]
    ax.scatter(age[treatment == 1], severity[treatment == 1], alpha=0.15, c="#e74c3c", label="Treated", s=10)
    ax.scatter(age[treatment == 0], severity[treatment == 0], alpha=0.15, c="#3498db", label="Control", s=10)
    ax.set_xlabel("Age")
    ax.set_ylabel("Severity")
    ax.set_title("Confounding: Sicker Patients Get Treatment")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "propensity_matching.png"), dpi=150)
    plt.close()
    log("→ Plot saved: plots/propensity_matching.png")


def causal_concepts():
    """Explain key causal inference concepts."""
    log("\n" + "=" * 60)
    log("PART 3: KEY CAUSAL CONCEPTS")
    log("=" * 60)

    log("""
Correlation ≠ Causation — Three ways correlation arises without causation:
  1. Confounding:  Z → X and Z → Y  (ice cream sales ↔ drowning, confounder: hot weather)
  2. Reverse causation:  Y → X  (hospitals have high death rates — sick people go there)
  3. Collider bias:  conditioning on a common effect creates spurious association

The Causal Toolbox:
  ┌─────────────────────────┬──────────────────────────────────────────┐
  │ Method                  │ When to use                              │
  ├─────────────────────────┼──────────────────────────────────────────┤
  │ Randomized experiment   │ Gold standard. You control treatment.    │
  │ Propensity matching     │ Observational data, measured confounders │
  │ Instrumental variables  │ Unmeasured confounders, valid instrument │
  │ Difference-in-diff      │ Before/after with control group          │
  │ Regression discontinuity│ Sharp cutoff determines treatment        │
  └─────────────────────────┴──────────────────────────────────────────┘

Do-Calculus (Pearl):
  P(Y | do(X=1)) ≠ P(Y | X=1)
  "do" means intervening (setting X), not just observing X.
  The adjustment formula: P(Y|do(X)) = Σ_z P(Y|X,Z) P(Z)
  where Z is the set of confounders to adjust for.
""")


def run_causal_demo():
    log("CAUSAL INFERENCE DEMO")
    log("=" * 60)
    simpsons_paradox()
    propensity_score_matching()
    causal_concepts()

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved to output.txt")


if __name__ == "__main__":
    run_causal_demo()
