"""
TF-IDF — FROM SCRATCH
=======================
No sklearn. Hand-coded Term Frequency, Inverse Document Frequency, and
TF-IDF matrix computation.

TF-IDF answers: "How important is this word to THIS document relative
to the whole corpus?"

  TF(t,d)  = count(t in d) / total_words_in_d
  IDF(t)   = log(N / df(t))    where df(t) = docs containing term t
  TF-IDF   = TF × IDF

High TF-IDF = word is frequent in THIS doc but rare across docs (distinctive).
Low TF-IDF  = word is common everywhere ("the", "is", "and").

Run:
    python tfidf_scratch.py
"""

import os
import numpy as np
import math

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

DOCUMENTS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
    "the cat chased the dog around the house",
    "machine learning is a branch of artificial intelligence",
    "deep learning uses neural networks for machine learning tasks",
    "natural language processing deals with text and speech",
    "the transformer architecture revolutionized natural language processing",
]


def tokenize(doc):
    """Lowercase and split on whitespace."""
    return doc.lower().split()


def compute_tf(doc_tokens):
    """Term Frequency: count(word) / total_words."""
    total = len(doc_tokens)
    tf = {}
    for word in doc_tokens:
        tf[word] = tf.get(word, 0) + 1
    return {word: count / total for word, count in tf.items()}


def compute_idf(corpus_tokens):
    """Inverse Document Frequency: log(N / df(t))."""
    N = len(corpus_tokens)
    df = {}
    for doc_tokens in corpus_tokens:
        unique_words = set(doc_tokens)
        for word in unique_words:
            df[word] = df.get(word, 0) + 1
    return {word: math.log(N / count) for word, count in df.items()}


def compute_tfidf(documents):
    """Compute the full TF-IDF matrix."""
    corpus_tokens = [tokenize(doc) for doc in documents]
    idf = compute_idf(corpus_tokens)
    vocab = sorted(idf.keys())
    vocab_idx = {word: i for i, word in enumerate(vocab)}

    tfidf_matrix = np.zeros((len(documents), len(vocab)))
    tf_list = []

    for doc_i, doc_tokens in enumerate(corpus_tokens):
        tf = compute_tf(doc_tokens)
        tf_list.append(tf)
        for word, tf_val in tf.items():
            col = vocab_idx[word]
            tfidf_matrix[doc_i, col] = tf_val * idf[word]

    return tfidf_matrix, vocab, tf_list, idf


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def run_tfidf_demo():
    tfidf_matrix, vocab, tf_list, idf = compute_tfidf(DOCUMENTS)

    lines = [
        "=" * 70, "  TF-IDF — FROM SCRATCH", "=" * 70,
        f"  Documents  : {len(DOCUMENTS)}",
        f"  Vocabulary : {len(vocab)} unique words", "",
    ]

    # show TF for first doc
    lines.append(f"  Example — Doc 0: \"{DOCUMENTS[0]}\"")
    lines.append(f"  TF (term frequency):")
    for word, tf_val in sorted(tf_list[0].items(), key=lambda x: -x[1]):
        lines.append(f"    {word:15s} : {tf_val:.4f}")

    # show IDF for selected words
    lines += ["", "  IDF (inverse document frequency) — selected words:"]
    sample_words = ["the", "cat", "dog", "machine", "learning", "transformer", "friends"]
    for word in sample_words:
        if word in idf:
            lines.append(f"    {word:15s} : {idf[word]:.4f}  (appears in {int(len(DOCUMENTS)/math.exp(idf[word]))} docs)")

    # show top TF-IDF words per document
    lines += ["", "  Top TF-IDF words per document (most distinctive):"]
    for i, doc in enumerate(DOCUMENTS):
        row = tfidf_matrix[i]
        top_idx = np.argsort(row)[-3:][::-1]
        top_words = [(vocab[j], row[j]) for j in top_idx if row[j] > 0]
        top_str = ", ".join(f"{w}={s:.3f}" for w, s in top_words)
        lines.append(f"    Doc {i}: \"{doc[:50]:50s}\" → {top_str}")

    # document similarity using TF-IDF vectors
    lines += ["", "  Document similarity (cosine of TF-IDF vectors):"]
    pairs = [(0, 1), (0, 4), (4, 5), (6, 7), (0, 7)]
    for i, j in pairs:
        sim = cosine_sim(tfidf_matrix[i], tfidf_matrix[j])
        lines.append(f"    Doc {i} ↔ Doc {j} : {sim:.4f}  (\"{DOCUMENTS[i][:30]}\" vs \"{DOCUMENTS[j][:30]}\")")

    lines += [
        "", "  Key formulas:",
        "    TF(t,d)  = count(t in d) / total_words_in_d",
        "    IDF(t)   = log(N / docs_containing_t)",
        "    TF-IDF   = TF × IDF",
        "    High TF-IDF → word is distinctive to this document",
        "    Low TF-IDF  → word is common across all documents (\"the\", \"is\")",
        "=" * 70,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_tfidf_demo()
