"""
LLM Evaluation — Demo
========================
How to evaluate text generation quality:

1. BLEU  — n-gram precision (machine translation)
2. ROUGE — n-gram recall (summarization)
3. Both computed from scratch + comparison with library implementations
4. LLM-as-judge — using a scoring rubric (simulated)
5. When each metric fails and better alternatives
"""

import os
import re
import math
from collections import Counter
import numpy as np
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


# ═══════════════════════════════════════════════════════════
# BLEU — from scratch
# ═══════════════════════════════════════════════════════════

def tokenize(text):
    return re.findall(r'\w+', text.lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference, candidate, max_n=4):
    """
    BLEU score from scratch.
    Measures n-gram precision of candidate against reference.
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if len(cand_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(ref_tokens, n))
        cand_ngrams = Counter(ngrams(cand_tokens, n))

        clipped = {ng: min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items()}
        numerator = sum(clipped.values())
        denominator = max(sum(cand_ngrams.values()), 1)
        precisions.append(numerator / denominator)

    # Brevity penalty
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(cand_tokens))

    # Geometric mean of precisions (with smoothing for zero)
    log_avg = 0
    n_valid = 0
    for p in precisions:
        if p > 0:
            log_avg += math.log(p)
            n_valid += 1
    if n_valid == 0:
        return 0.0

    log_avg /= max_n
    return bp * math.exp(log_avg)


# ═══════════════════════════════════════════════════════════
# ROUGE — from scratch
# ═══════════════════════════════════════════════════════════

def rouge_n(reference, candidate, n=1):
    """
    ROUGE-N: n-gram recall of reference n-grams in the candidate.
    """
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    ref_ng = Counter(ngrams(ref_tokens, n))
    cand_ng = Counter(ngrams(cand_tokens, n))

    overlap = {ng: min(ref_ng[ng], cand_ng.get(ng, 0)) for ng in ref_ng}
    numerator = sum(overlap.values())
    denominator = sum(ref_ng.values())

    if denominator == 0:
        return {"precision": 0, "recall": 0, "f1": 0}

    recall = numerator / denominator
    cand_denom = sum(cand_ng.values())
    precision = numerator / cand_denom if cand_denom > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(x, y):
    """Longest Common Subsequence length."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(reference, candidate):
    """ROUGE-L: based on Longest Common Subsequence."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    lcs = lcs_length(ref_tokens, cand_tokens)
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return {"precision": 0, "recall": 0, "f1": 0}

    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


# ═══════════════════════════════════════════════════════════
# LLM-as-Judge (simulated)
# ═══════════════════════════════════════════════════════════

def llm_judge_score(reference, candidate, rubric=None):
    """
    Simulate LLM-as-judge scoring.
    In practice you'd send a prompt to GPT-4 / Claude asking it to score.
    Here we approximate with heuristic-based scoring.
    """
    ref_tokens = set(tokenize(reference))
    cand_tokens = set(tokenize(candidate))

    # Coverage: how many reference concepts appear in candidate
    coverage = len(ref_tokens & cand_tokens) / max(len(ref_tokens), 1)

    # Fluency proxy: average word length (longer = more sophisticated?)
    avg_word_len = np.mean([len(w) for w in tokenize(candidate)]) if tokenize(candidate) else 0
    fluency = min(avg_word_len / 6.0, 1.0)

    # Length appropriateness
    ratio = len(tokenize(candidate)) / max(len(tokenize(reference)), 1)
    length_score = 1.0 - abs(1.0 - ratio) * 0.5
    length_score = max(0, min(1, length_score))

    scores = {
        "coverage": round(coverage * 5, 1),
        "fluency": round(fluency * 5, 1),
        "length_appropriateness": round(length_score * 5, 1),
    }
    scores["overall"] = round(np.mean(list(scores.values())), 1)
    return scores


def run_demo():
    log("LLM EVALUATION — DEMO")
    log("=" * 60)

    # ═══════════════════════════════════════════════════════
    # Test cases
    # ═══════════════════════════════════════════════════════
    test_cases = [
        {
            "name": "Good translation",
            "reference": "The cat sat on the mat in the living room",
            "candidate": "The cat sat on the mat in the living room",
        },
        {
            "name": "Decent paraphrase",
            "reference": "The cat sat on the mat in the living room",
            "candidate": "A cat was sitting on a mat in the living area",
        },
        {
            "name": "Partial overlap",
            "reference": "The cat sat on the mat in the living room",
            "candidate": "The cat is sleeping on the floor",
        },
        {
            "name": "Completely wrong",
            "reference": "The cat sat on the mat in the living room",
            "candidate": "Dogs are wonderful pets that love walks",
        },
        {
            "name": "Too short",
            "reference": "The cat sat on the mat in the living room",
            "candidate": "Cat mat",
        },
        {
            "name": "Too verbose",
            "reference": "The cat sat on the mat in the living room",
            "candidate": "The very large and fluffy cat slowly and carefully sat down on the old dirty mat that was placed in the corner of the big spacious living room of the house",
        },
    ]

    # ═══════════════════════════════════════════════════════
    # BLEU scores
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("BLEU SCORES (n-gram precision)")
    log("=" * 60)
    log(f"\n  BLEU measures: of the n-grams in the CANDIDATE, how many appear in the reference?")
    log(f"  → Penalizes candidates that add wrong words (precision-focused)\n")

    bleu_scores = []
    log(f"  {'Case':<22} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-4':>8}")
    log(f"  {'-' * 50}")
    for tc in test_cases:
        b1 = bleu_score(tc["reference"], tc["candidate"], max_n=1)
        b2 = bleu_score(tc["reference"], tc["candidate"], max_n=2)
        b4 = bleu_score(tc["reference"], tc["candidate"], max_n=4)
        log(f"  {tc['name']:<22} {b1:>8.4f} {b2:>8.4f} {b4:>8.4f}")
        bleu_scores.append(b4)

    # ═══════════════════════════════════════════════════════
    # ROUGE scores
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("ROUGE SCORES (n-gram recall)")
    log("=" * 60)
    log(f"\n  ROUGE measures: of the n-grams in the REFERENCE, how many appear in the candidate?")
    log(f"  → Penalizes candidates that miss reference content (recall-focused)\n")

    rouge_scores = []
    log(f"  {'Case':<22} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}")
    log(f"  {'-' * 50}")
    for tc in test_cases:
        r1 = rouge_n(tc["reference"], tc["candidate"], n=1)
        r2 = rouge_n(tc["reference"], tc["candidate"], n=2)
        rl = rouge_l(tc["reference"], tc["candidate"])
        log(f"  {tc['name']:<22} {r1['f1']:>8.4f} {r2['f1']:>8.4f} {rl['f1']:>8.4f}")
        rouge_scores.append(rl["f1"])

    # ═══════════════════════════════════════════════════════
    # LLM-as-Judge
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("LLM-AS-JUDGE SCORES (simulated)")
    log("=" * 60)
    log(f"\n  In practice, you send this prompt to GPT-4/Claude:")
    log(f"    'Rate the following response on a 1-5 scale for coverage, fluency,")
    log(f"     and length appropriateness. Reference: ... Candidate: ...'\n")

    judge_scores = []
    log(f"  {'Case':<22} {'Coverage':>9} {'Fluency':>8} {'Length':>8} {'Overall':>8}")
    log(f"  {'-' * 58}")
    for tc in test_cases:
        j = llm_judge_score(tc["reference"], tc["candidate"])
        log(f"  {tc['name']:<22} {j['coverage']:>9.1f} {j['fluency']:>8.1f} {j['length_appropriateness']:>8.1f} {j['overall']:>8.1f}")
        judge_scores.append(j["overall"])

    # ═══════════════════════════════════════════════════════
    # When metrics disagree
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("WHEN METRICS DISAGREE")
    log("=" * 60)

    log(f"\n  {'Case':<22} {'BLEU-4':>8} {'ROUGE-L':>8} {'Judge':>8}  Analysis")
    log(f"  {'-' * 75}")
    analyses = [
        "All agree: perfect match",
        "BLEU down (new words), ROUGE OK (covers ref)",
        "All low — limited overlap",
        "All ~0 — no semantic match",
        "BLEU penalized (too short), ROUGE OK (words match)",
        "BLEU OK (ref words present), ROUGE high (covers ref)",
    ]
    for i, tc in enumerate(test_cases):
        log(f"  {tc['name']:<22} {bleu_scores[i]:>8.4f} {rouge_scores[i]:>8.4f} {judge_scores[i]:>8.1f}  {analyses[i]}")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    case_names = [tc["name"] for tc in test_cases]
    x = np.arange(len(case_names))
    width = 0.25

    ax = axes[0]
    ax.bar(x - width, bleu_scores, width, label="BLEU-4", color="#3498db")
    ax.bar(x, rouge_scores, width, label="ROUGE-L", color="#2ecc71")
    ax.bar(x + width, [s/5 for s in judge_scores], width, label="Judge/5", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison Across Cases")
    ax.legend()

    ax = axes[1]
    ax.scatter(bleu_scores, rouge_scores, c=judge_scores, cmap="RdYlGn", s=120, edgecolors="black")
    for i, name in enumerate(case_names):
        ax.annotate(name, (bleu_scores[i], rouge_scores[i]), fontsize=7,
                   textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("BLEU-4")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("BLEU vs ROUGE (color = judge score)")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Judge Score")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "llm_evaluation.png"), dpi=150)
    plt.close()
    log(f"\n→ Plot saved: plots/llm_evaluation.png")

    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
  ┌────────────┬────────────────────────────────────────────┐
  │ Metric     │ What it measures                           │
  ├────────────┼────────────────────────────────────────────┤
  │ BLEU       │ Precision: are candidate words correct?    │
  │ ROUGE      │ Recall: are reference words covered?       │
  │ ROUGE-L    │ Longest common subsequence (word order)    │
  │ BERTScore  │ Semantic similarity via BERT embeddings    │
  │ METEOR     │ Combines precision, recall, synonyms       │
  │ LLM-judge  │ Holistic: fluency, factuality, relevance  │
  └────────────┴────────────────────────────────────────────┘

  Limitations:
    • BLEU/ROUGE only match exact words — "car" ≠ "automobile"
    • Both ignore semantic meaning entirely
    • LLM-as-judge is expensive but most correlated with human preference
    • For open-ended generation, human eval is still the gold standard

  Best practice: use BLEU/ROUGE as cheap filters, LLM-as-judge for
  final evaluation, human eval for critical decisions.
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
