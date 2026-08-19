"""
BERT Demo — Hugging Face Transformers (no fine-tuning)
=======================================================
Runs a pre-trained BERT model on sample text for several tasks:
  1. Fill-Mask        — predict a missing word
  2. Sentiment        — classify positive / negative
  3. NER              — named entity recognition
  4. Text Similarity  — compare sentence embeddings

Run:
    python bert_demo.py
"""

import os
import io
import sys
from transformers import pipeline

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def demo_fill_mask():
    """BERT's native pre-training task: predict a masked token."""
    print("━" * 60)
    print("  1.  FILL-MASK  (bert-base-uncased)")
    print("━" * 60)

    filler = pipeline("fill-mask", model="bert-base-uncased")

    sentences = [
        "Artificial intelligence will [MASK] the world.",
        "The capital of France is [MASK].",
        "Machine learning is a subset of [MASK] intelligence.",
        "The [MASK] is the powerhouse of the cell.",
    ]

    for sentence in sentences:
        print(f"\n  Input : {sentence}")
        results = filler(sentence, top_k=3)
        for r in results:
            print(f"    -> {r['token_str']:<15}  (score: {r['score']:.4f})")


def demo_sentiment():
    """Sentiment analysis using a fine-tuned DistilBERT."""
    print("\n" + "━" * 60)
    print("  2.  SENTIMENT ANALYSIS  (distilbert-base-uncased)")
    print("━" * 60)

    classifier = pipeline("sentiment-analysis")

    texts = [
        "I absolutely love this product, it changed my life!",
        "This is the worst experience I have ever had.",
        "The movie was okay, nothing special but not terrible either.",
        "Machine learning is fascinating and full of possibilities.",
        "I'm frustrated with all the bugs in this software.",
    ]

    for text in texts:
        result = classifier(text)[0]
        print(f"\n  Text  : {text}")
        print(f"  Label : {result['label']}  (confidence: {result['score']:.4f})")


def demo_ner():
    """Named Entity Recognition — detect people, orgs, locations."""
    print("\n" + "━" * 60)
    print("  3.  NAMED ENTITY RECOGNITION")
    print("━" * 60)

    ner = pipeline("ner", aggregation_strategy="simple")

    texts = [
        "Elon Musk founded SpaceX in Hawthorne, California.",
        "Google was started by Larry Page and Sergey Brin at Stanford University.",
        "The Eiffel Tower in Paris was built by Gustave Eiffel in 1889.",
    ]

    for text in texts:
        print(f"\n  Text: {text}")
        entities = ner(text)
        if not entities:
            print("    (no entities found)")
        for ent in entities:
            print(f"    [{ent['entity_group']:<5}] {ent['word']:<25}  "
                  f"(score: {ent['score']:.4f})")


def demo_zero_shot():
    """Classify text into arbitrary categories without training."""
    print("\n" + "━" * 60)
    print("  4.  ZERO-SHOT CLASSIFICATION")
    print("━" * 60)

    classifier = pipeline("zero-shot-classification")

    examples = [
        {
            "text": "The stock market crashed after the Fed raised interest rates.",
            "labels": ["finance", "sports", "technology", "politics"],
        },
        {
            "text": "The new iPhone features a faster chip and better camera.",
            "labels": ["finance", "sports", "technology", "politics"],
        },
        {
            "text": "The team won the championship after a thrilling overtime.",
            "labels": ["finance", "sports", "technology", "politics"],
        },
    ]

    for ex in examples:
        result = classifier(ex["text"], candidate_labels=ex["labels"])
        print(f"\n  Text: {ex['text']}")
        for label, score in zip(result["labels"], result["scores"]):
            bar = "█" * int(score * 30)
            print(f"    {label:<12} {score:.3f}  {bar}")


def main():
    capture = io.StringIO()

    class Tee:
        """Write to both stdout and a StringIO buffer."""
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, capture)

    print("\n" + "=" * 60)
    print("   BERT / Transformers Demo  —  Pre-trained (no fine-tuning)")
    print("=" * 60)

    demo_fill_mask()
    demo_sentiment()
    demo_ner()
    demo_zero_shot()

    print("\n" + "=" * 60)
    print("   All demos complete.")
    print("=" * 60 + "\n")

    sys.stdout = original_stdout

    out_path = os.path.join(OUTPUT_DIR, "output.txt")
    with open(out_path, "w") as f:
        f.write(capture.getvalue())
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
