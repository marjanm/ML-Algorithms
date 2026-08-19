"""
A/B Testing & Statistical Tests — Demo
========================================
How to know if your model is actually better, not just lucky.

Demonstrates:
  1. Simulating an A/B test (conversion rates)
  2. Two-sample t-test & z-test
  3. Chi-squared test for proportions
  4. Confidence intervals
  5. Statistical power & sample size calculation
  6. Pitfall: peeking (early stopping bias)
  7. Multiple comparisons correction (Bonferroni)

Run:
    python ab_testing_demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

np.random.seed(42)


def simulate_ab_test(n_a=5000, n_b=5000, rate_a=0.10, rate_b=0.12):
    """Simulate click/conversion data for two variants."""
    conversions_a = np.random.binomial(1, rate_a, n_a)
    conversions_b = np.random.binomial(1, rate_b, n_b)
    return conversions_a, conversions_b


def z_test_proportions(conv_a, conv_b):
    """Two-proportion z-test."""
    n_a, n_b = len(conv_a), len(conv_b)
    p_a, p_b = conv_a.mean(), conv_b.mean()
    # Pooled proportion under H₀ (no difference): shared baseline for the SE estimate
    p_pool = (conv_a.sum() + conv_b.sum()) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value, p_a, p_b, se


def confidence_interval(p, n, confidence=0.95):
    """Wilson score interval for a proportion."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
    return centre - margin, centre + margin


def required_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """Minimum sample size per group for a two-proportion z-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1, p2 = baseline_rate, baseline_rate + mde
    p_avg = (p1 + p2) / 2
    n = ((z_alpha * np.sqrt(2 * p_avg * (1 - p_avg))
          + z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2))) / mde) ** 2
    return int(np.ceil(n))


def peeking_simulation(n_total=10000, rate_a=0.10, rate_b=0.10, n_peeks=20):
    """Show how repeated peeking inflates false positive rate.x
    Both groups have the SAME rate (no real effect) — any significance is a false positive.
    """
    conv_a = np.random.binomial(1, rate_a, n_total)
    conv_b = np.random.binomial(1, rate_b, n_total)

    peek_points = np.linspace(100, n_total, n_peeks, dtype=int)
    p_values = []
    for n in peek_points:
        _, pv, _, _, _ = z_test_proportions(conv_a[:n], conv_b[:n])
        p_values.append(pv)
    return peek_points, p_values


def run_ab_demo():
    lines = [
        "=" * 65,
        "  A/B TESTING & STATISTICAL TESTS  —  Demo",
        "=" * 65, "",
    ]

    # ── 1. Basic A/B test ──
    lines += ["  ── 1. Simulated A/B Test ──"]
    conv_a, conv_b = simulate_ab_test(n_a=5000, n_b=5000, rate_a=0.10, rate_b=0.12)
    z_stat, p_val, p_a, p_b, se = z_test_proportions(conv_a, conv_b)

    lines += [
        f"    Group A (control): n={len(conv_a)}, conversions={conv_a.sum()}, rate={p_a:.4f}",
        f"    Group B (variant):  n={len(conv_b)}, conversions={conv_b.sum()}, rate={p_b:.4f}",
        f"    Observed lift: {(p_b - p_a) / p_a:+.2%}",
        f"",
        f"    Z-statistic: {z_stat:.4f}",
        f"    P-value:     {p_val:.6f}",
        f"    Significant at α=0.05? {'YES' if p_val < 0.05 else 'NO'}",
    ]

    # ── 2. Chi-squared test ──
    lines += ["", "  ── 2. Chi-Squared Test ──"]
    table = np.array([
        [conv_a.sum(), len(conv_a) - conv_a.sum()],
        [conv_b.sum(), len(conv_b) - conv_b.sum()],
    ])
    chi2, chi2_p, dof, expected = stats.chi2_contingency(table)
    lines += [
        f"    Chi² statistic: {chi2:.4f}",
        f"    P-value:        {chi2_p:.6f}",
        f"    Degrees of freedom: {dof}",
        f"    Agrees with z-test? {'YES' if (chi2_p < 0.05) == (p_val < 0.05) else 'NO'}",
    ]

    # ── 3. Confidence intervals ──
    lines += ["", "  ── 3. Confidence Intervals (95%) ──"]
    ci_a = confidence_interval(p_a, len(conv_a))
    ci_b = confidence_interval(p_b, len(conv_b))
    lines += [
        f"    Group A: {p_a:.4f}  [{ci_a[0]:.4f}, {ci_a[1]:.4f}]",
        f"    Group B: {p_b:.4f}  [{ci_b[0]:.4f}, {ci_b[1]:.4f}]",
        f"    Overlap? {'YES — not conclusive' if ci_a[1] > ci_b[0] else 'NO — strong signal'}",
    ]

    fig, ax = plt.subplots(figsize=(7, 3))
    for i, (name, p, ci) in enumerate([("A (control)", p_a, ci_a), ("B (variant)", p_b, ci_b)]):
        ax.errorbar(p, i, xerr=[[p - ci[0]], [ci[1] - p]], fmt="o", capsize=5,
                    color=["steelblue", "coral"][i], markersize=8)
        ax.text(ci[1] + 0.002, i, f"{p:.4f}", va="center")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["A (control)", "B (variant)"])
    ax.set_xlabel("Conversion Rate")
    ax.set_title("95% Confidence Intervals")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confidence_intervals.png"), dpi=150)
    plt.close()
    lines.append(f"    [saved] → plots/confidence_intervals.png")

    # ── 4. Power analysis & sample size ──
    lines += ["", "  ── 4. Power Analysis ──"]
    mde_levels = [0.005, 0.01, 0.02, 0.03, 0.05]
    lines.append(f"    Baseline rate: 10%")
    lines.append(f"    α = 0.05, power = 0.80")
    lines.append(f"")
    lines.append(f"    {'MDE':>6s} | {'Sample size/group':>18s} | {'Total':>10s}")
    lines.append(f"    {'-'*6}-+-{'-'*18}-+-{'-'*10}")
    sample_sizes = []
    for mde in mde_levels:
        n = required_sample_size(0.10, mde)
        sample_sizes.append(n)
        lines.append(f"    {mde:6.3f} | {n:18,d} | {2*n:10,d}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([m*100 for m in mde_levels], sample_sizes, marker="o", color="coral")
    ax.set_xlabel("Minimum Detectable Effect (percentage points)")
    ax.set_ylabel("Required Sample Size per Group")
    ax.set_title("Sample Size vs Effect Size (α=0.05, power=80%)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sample_size.png"), dpi=150)
    plt.close()
    lines.append(f"    [saved] → plots/sample_size.png")

    # ── 5. Peeking pitfall ──
    lines += ["", "  ── 5. Pitfall: Peeking (Repeated Testing) ──",
              "    Both groups have the SAME rate (10%) — no real effect.",
              "    Checking p-value repeatedly inflates false positives."]

    n_simulations = 500
    false_positives = 0
    for _ in range(n_simulations):
        ca = np.random.binomial(1, 0.10, 10000)
        cb = np.random.binomial(1, 0.10, 10000)
        peek_pts = np.linspace(500, 10000, 20, dtype=int)
        for n in peek_pts:
            _, pv, _, _, _ = z_test_proportions(ca[:n], cb[:n])
            if pv < 0.05:
                false_positives += 1
                break

    fpr = false_positives / n_simulations
    lines.append(f"    False positive rate with peeking: {fpr:.1%} (should be 5%)")

    peek_pts, peek_pvals = peeking_simulation()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(peek_pts, peek_pvals, marker=".", color="steelblue")
    ax.axhline(y=0.05, color="red", linestyle="--", label="α = 0.05")
    ax.set_xlabel("Sample Size (at time of peek)")
    ax.set_ylabel("P-value")
    ax.set_title("P-value Trajectory with Repeated Peeking (no real effect)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "peeking_pitfall.png"), dpi=150)
    plt.close()
    lines.append(f"    [saved] → plots/peeking_pitfall.png")

    # ── 6. Multiple comparisons ──
    lines += ["", "  ── 6. Multiple Comparisons (Bonferroni) ──",
              "    Testing 5 variants at once → higher chance of false positive."]
    n_variants = 5
    p_values_multi = []
    for v in range(n_variants):
        cv = np.random.binomial(1, 0.10, 5000)
        _, pv, _, _, _ = z_test_proportions(conv_a, cv)
        p_values_multi.append(pv)

    alpha_corrected = 0.05 / n_variants
    lines.append(f"    Raw α = 0.05, Bonferroni-corrected α = {alpha_corrected:.3f}")
    lines.append(f"")
    for v, pv in enumerate(p_values_multi):
        sig_raw = "SIG" if pv < 0.05 else "   "
        sig_corr = "SIG" if pv < alpha_corrected else "   "
        lines.append(f"    Variant {v+1}: p={pv:.4f}  raw: {sig_raw}  corrected: {sig_corr}")

    lines += [
        "", "  ── Key Concepts ──",
        "    P-value:          'Is the difference real or noise?'",
        "    Lift / effect size: 'How big is the difference?'",
        "    MDE (minimum lift): 'How many samples do I need to detect a lift this small?'",
        "",
        "    → P-value alone is not enough. Always check the lift too.",
        "    → A tiny lift can be 'significant' with enough data, but not worth acting on.",
    ]

    lines += [
        "", "  ── When to Use Which Test ──",
        "    • Z-test:     large samples, comparing proportions (CTR, conversion)",
        "    • T-test:     continuous outcomes (revenue, time-on-page), small samples OK",
        "    • Chi-squared: categorical outcomes, contingency tables",
        "    • Always: decide sample size BEFORE the test (power analysis)",
        "    • Never:  peek at results and stop early (inflates false positives)",
        "    • Multiple variants: use Bonferroni or Holm correction",
        "", "=" * 65,
    ]

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_ab_demo()
