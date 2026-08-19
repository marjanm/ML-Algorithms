"""
Probabilistic Modeling Intuition — Demo
==========================================
How Bayesian thinking differs from frequentist ML:

1. Prior → Likelihood → Posterior (coin flip example)
2. Conjugate priors (Beta-Binomial, Normal-Normal)
3. Posterior predictive distribution
4. Credible intervals vs confidence intervals
5. Bayesian A/B testing: "probability that B is better than A"
"""

import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output.txt")

lines = []
def log(msg=""):
    print(msg)
    lines.append(str(msg))


np.random.seed(42)


def beta_binomial_demo():
    """
    Classic Bayesian updating: coin flips.
    Prior: Beta(α, β) — our belief about fairness before seeing data.
    Likelihood: Binomial — probability of observed heads.
    Posterior: Beta(α + heads, β + tails) — updated belief.
    """
    log("=" * 60)
    log("DEMO 1: BAYESIAN UPDATING (is this coin fair?)")
    log("=" * 60)

    true_p = 0.65  # coin is slightly biased
    prior_alpha, prior_beta = 1, 1  # uniform prior (no opinion)

    observations = np.random.binomial(1, true_p, 100)

    checkpoints = [0, 1, 5, 10, 25, 50, 100]
    posteriors = []

    log(f"\n  True probability: {true_p}")
    log(f"  Prior: Beta({prior_alpha}, {prior_beta}) — uniform, no prior knowledge")
    log(f"\n  {'N flips':>8} {'Heads':>6} {'Post α':>8} {'Post β':>8} {'Post Mean':>10} {'95% CI':>20}")
    log(f"  {'-' * 65}")

    for n in checkpoints:
        heads = observations[:n].sum() if n > 0 else 0
        tails = n - heads
        a = prior_alpha + heads
        b = prior_beta + tails
        mean = a / (a + b)
        ci = stats.beta.ppf([0.025, 0.975], a, b)
        posteriors.append((n, a, b))
        log(f"  {n:>8} {heads:>6} {a:>8} {b:>8} {mean:>10.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    log(f"\n  → As we see more data, the posterior tightens around the true value.")
    log(f"  → The prior washes out: with 100 observations, prior barely matters.")
    return posteriors


def normal_normal_demo():
    """
    Estimating a population mean with uncertainty.
    Prior: N(μ₀, σ₀²)
    Likelihood: N(μ, σ²/n)
    Posterior: N(μ_post, σ_post²)
    """
    log(f"\n{'=' * 60}")
    log("DEMO 2: ESTIMATING A MEAN (Normal-Normal conjugate)")
    log("=" * 60)

    true_mean = 170  # true height in cm
    true_std = 10
    prior_mean = 165  # our prior guess
    prior_std = 15    # how uncertain we are

    sample_sizes = [0, 1, 5, 10, 30, 100]
    all_data = np.random.normal(true_mean, true_std, 100)

    log(f"\n  True mean: {true_mean} cm")
    log(f"  Prior: N({prior_mean}, {prior_std}²)")
    log(f"\n  {'N obs':>6} {'Post μ':>10} {'Post σ':>10} {'95% Credible Interval':>30}")
    log(f"  {'-' * 60}")

    post_data = []
    for n in sample_sizes:
        if n == 0:
            post_mean = prior_mean
            post_std = prior_std
        else:
            data = all_data[:n]
            data_mean = data.mean()
            # Posterior for known variance case
            post_var = 1 / (1/prior_std**2 + n/true_std**2)
            post_mean = post_var * (prior_mean/prior_std**2 + n*data_mean/true_std**2)
            post_std = np.sqrt(post_var)

        ci = (post_mean - 1.96*post_std, post_mean + 1.96*post_std)
        log(f"  {n:>6} {post_mean:>10.2f} {post_std:>10.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
        post_data.append((n, post_mean, post_std))

    return post_data


def bayesian_ab_test():
    """
    Bayesian A/B testing: "what is P(B > A)?"
    Instead of p-values, we get a direct probability.
    """
    log(f"\n{'=' * 60}")
    log("DEMO 3: BAYESIAN A/B TESTING")
    log("=" * 60)

    # Simulate: A has 5% CTR, B has 6% CTR
    n_a, n_b = 1000, 1000
    true_ctr_a, true_ctr_b = 0.05, 0.06

    clicks_a = np.random.binomial(n_a, true_ctr_a)
    clicks_b = np.random.binomial(n_b, true_ctr_b)

    log(f"\n  Variant A: {clicks_a}/{n_a} clicks ({clicks_a/n_a:.2%} CTR)")
    log(f"  Variant B: {clicks_b}/{n_b} clicks ({clicks_b/n_b:.2%} CTR)")

    # Posterior: Beta(1 + clicks, 1 + no-clicks)
    post_a = stats.beta(1 + clicks_a, 1 + n_a - clicks_a)
    post_b = stats.beta(1 + clicks_b, 1 + n_b - clicks_b)

    # Monte Carlo: P(B > A)
    n_samples = 100000
    samples_a = post_a.rvs(n_samples)
    samples_b = post_b.rvs(n_samples)
    p_b_better = (samples_b > samples_a).mean()
    expected_lift = ((samples_b - samples_a) / samples_a).mean()

    log(f"\n  P(B > A) = {p_b_better:.4f}")
    log(f"  Expected lift = {expected_lift:.2%}")
    log(f"  95% credible interval for lift: "
        f"[{np.percentile((samples_b-samples_a)/samples_a, 2.5):.2%}, "
        f"{np.percentile((samples_b-samples_a)/samples_a, 97.5):.2%}]")

    log(f"\n  Frequentist would say: 'p = 0.03, reject null at α=0.05'")
    log(f"  Bayesian says: 'there is a {p_b_better:.0%} chance B is better,")
    log(f"  with an expected lift of {expected_lift:.2%}'")
    log(f"  → Much more interpretable for stakeholders!")

    return samples_a, samples_b, post_a, post_b


def run_demo():
    log("PROBABILISTIC MODELING INTUITION — DEMO")
    log("=" * 60)

    posteriors = beta_binomial_demo()
    post_data = normal_normal_demo()
    samples_a, samples_b, post_a, post_b = bayesian_ab_test()

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Posterior evolution (Beta-Binomial)
    ax = axes[0, 0]
    x = np.linspace(0, 1, 200)
    colors = plt.cm.viridis(np.linspace(0, 1, len(posteriors)))
    for (n, a, b), c in zip(posteriors, colors):
        y = stats.beta.pdf(x, a, b)
        ax.plot(x, y, color=c, label=f"n={n}", linewidth=2)
    ax.axvline(x=0.65, color="red", linestyle="--", alpha=0.5, label="True p=0.65")
    ax.set_xlabel("Probability of Heads")
    ax.set_ylabel("Density")
    ax.set_title("Posterior Evolution (coin fairness)")
    ax.legend(fontsize=8)

    # Plot 2: Normal posterior evolution
    ax = axes[0, 1]
    for n, mu, sigma in post_data:
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        y = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, y, label=f"n={n}", linewidth=2)
    ax.axvline(x=170, color="red", linestyle="--", alpha=0.5, label="True μ=170")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior for Mean Height")
    ax.legend(fontsize=8)

    # Plot 3: A/B test posteriors
    ax = axes[1, 0]
    x = np.linspace(0.02, 0.10, 200)
    ax.plot(x, post_a.pdf(x), label="Variant A", color="#e74c3c", linewidth=2)
    ax.plot(x, post_b.pdf(x), label="Variant B", color="#2ecc71", linewidth=2)
    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Density")
    ax.set_title("A/B Test: Posterior CTR Distributions")
    ax.legend()

    # Plot 4: Lift distribution
    ax = axes[1, 1]
    lift = (samples_b - samples_a) / samples_a * 100
    ax.hist(lift, bins=80, color="#3498db", alpha=0.7, density=True)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No effect")
    ax.axvline(x=np.median(lift), color="green", linestyle="--", linewidth=2, label=f"Median: {np.median(lift):.1f}%")
    p_positive = (lift > 0).mean()
    ax.set_xlabel("Lift (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"Lift Distribution — P(B>A) = {p_positive:.1%}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "probabilistic_modeling.png"), dpi=150)
    plt.close()
    log(f"\n→ Plot saved: plots/probabilistic_modeling.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
  Bayesian vs Frequentist:
  ┌──────────────────┬───────────────────────────────────────┐
  │ Frequentist      │ Bayesian                              │
  ├──────────────────┼───────────────────────────────────────┤
  │ P-value: P(data  │ Posterior: P(hypothesis | data)       │
  │ | null)          │ → direct probability of hypothesis    │
  │ Point estimate   │ Full distribution over parameter      │
  │ Confidence       │ Credible interval: "95% probability   │
  │ interval: "95%   │ the parameter is in this range"       │
  │ of intervals     │                                       │
  │ contain truth"   │                                       │
  └──────────────────┴───────────────────────────────────────┘

  When to go Bayesian:
    • Small data (prior helps regularize)
    • Need uncertainty quantification
    • A/B testing (P(B>A) is more useful than p-values)
    • Sequential decisions (update beliefs as data arrives)

  Conjugate pairs (closed-form posteriors):
    • Beta-Binomial    → conversion rates, coin flips
    • Normal-Normal    → estimating means
    • Dirichlet-Multinomial → category distributions
    • Gamma-Poisson    → count data (events per time)
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
