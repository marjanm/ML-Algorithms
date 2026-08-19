"""
Tokenization — visual demo
============================
How text gets split into pieces (tokens) before a model sees it.

This demo compares four tokenisation strategies on the same sentences:
  1. Whitespace  — split on spaces (simplest, naive)
  2. Word-level  — split on word boundaries + punctuation
  3. BPE (Byte-Pair Encoding) — GPT-2's tokeniser (sub-word)
  4. WordPiece   — BERT's tokeniser (sub-word)

Sub-word tokenisers break rare words into known pieces:
  "unhappiness" → ["un", "##happiness"] (WordPiece)
  "unhappiness" → ["un", "happ", "iness"] (BPE)

Run:
    python tokenization_demo.py
"""

import os
from transformers import AutoTokenizer

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization is surprisingly important!",
    "unhappiness is not the same as sadness",
    "GPT-4 uses byte-pair encoding (BPE) for tokenization.",
    "Pneumonoultramicroscopicsilicovolcanoconiosis is a real word.",
    "I love New York 🗽",
    "Machine learning models don't see text — they see numbers.",
]


def run_tokenization_demo():
    tokenizers = {
        "GPT-2 (BPE)": AutoTokenizer.from_pretrained("gpt2"),
        "BERT (WordPiece)": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "DistilBERT (WordPiece)": AutoTokenizer.from_pretrained("distilbert-base-uncased"),
    }

    lines = [
        "=" * 80, "  TOKENIZATION  —  Demo", "=" * 80, "",
        "  Comparing how different tokenisers break the same text.", "",
    ]

    for sentence in SENTENCES:
        lines.append(f"  Input: \"{sentence}\"")
        lines.append(f"  {'—' * 70}")

        # whitespace baseline
        ws_tokens = sentence.split()
        lines.append(f"  {'Whitespace':20s} | {len(ws_tokens):3d} tokens | {ws_tokens}")

        for name, tok in tokenizers.items():
            tokens = tok.tokenize(sentence)
            ids = tok.encode(sentence)
            lines.append(f"  {name:20s} | {len(tokens):3d} tokens | {tokens}")

        lines.append("")

    # vocabulary size comparison
    lines.append("  Vocabulary sizes:")
    for name, tok in tokenizers.items():
        lines.append(f"    {name:20s} : {tok.vocab_size:,} tokens")

    lines += [
        "", "  Key concepts:",
        "    - Whitespace tokenisation: simple but can't handle unknown words",
        "    - WordPiece (BERT): splits unknowns into sub-words with ## prefix",
        "    - BPE (GPT-2): merges frequent byte-pairs into single tokens",
        "    - Sub-word tokenisers have a fixed vocabulary but can represent ANY text",
        "    - Longer words = more tokens = slower + more expensive to process",
        "    - Token count affects cost: GPT-4 charges per token, not per word",
        "=" * 80,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_tokenization_demo()
