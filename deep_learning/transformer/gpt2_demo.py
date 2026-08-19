"""
GPT-2 Text Generation — Hugging Face (no fine-tuning)
=======================================================
Uses the pre-trained GPT-2 model to generate text, demonstrating
how a real LLM (Large Language Model) works in practice.

GPT-2 is a decoder-only transformer trained to predict the next token.
It generates text autoregressively: predict one token, append it, repeat.

Run:
    python gpt2_demo.py
"""

import os
import io
import sys
from transformers import pipeline, set_seed

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def demo_text_generation():
    """Generate text continuations from prompts."""
    print("━" * 60)
    print("  1.  TEXT GENERATION  (GPT-2)")
    print("━" * 60)

    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)

    prompts = [
        "Artificial intelligence will",
        "The future of machine learning is",
        "In a world where robots",
        "The most important thing about deep learning is",
    ]

    for prompt in prompts:
        result = generator(
            prompt,
            max_new_tokens=60,             # how many tokens to generate after the prompt
            num_return_sequences=1,        # how many different completions to generate
            temperature=0.8,               # controls randomness: lower = more focused, higher = more creative
            top_k=50,                      # only sample from the top 50 most likely tokens
            top_p=0.9,                     # nucleus sampling: sample from tokens whose cumulative probability < 0.9
            do_sample=True,                # enable sampling (vs greedy decoding)
            repetition_penalty=1.2,        # penalise repeating the same token
        )
        print(f"\n  Prompt: \"{prompt}\"")
        print(f"  Output: {result[0]['generated_text']}")


def demo_temperature_comparison():
    """Show how temperature affects output creativity."""
    print("\n" + "━" * 60)
    print("  2.  TEMPERATURE COMPARISON")
    print("━" * 60)
    print("  (same prompt, different temperature values)")

    generator = pipeline("text-generation", model="gpt2")
    prompt = "The key to understanding neural networks is"

    for temp in [0.3, 0.7, 1.0, 1.5]:
        set_seed(42)
        result = generator(
            prompt,
            max_new_tokens=40,
            temperature=temp,
            do_sample=True,
            top_k=50,
        )
        print(f"\n  temp={temp}: {result[0]['generated_text']}")


def demo_fill_in_context():
    """Show how GPT-2 can complete different styles of text."""
    print("\n" + "━" * 60)
    print("  3.  STYLE COMPLETION")
    print("━" * 60)

    generator = pipeline("text-generation", model="gpt2")
    set_seed(42)

    prompts = [
        "Dear Sir or Madam, I am writing to",
        "def fibonacci(n):\n    \"\"\"",
        "Once upon a time in a galaxy far, far away,",
        "Breaking news: Scientists have discovered",
    ]

    for prompt in prompts:
        result = generator(
            prompt,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
        )
        print(f"\n  Prompt: \"{prompt[:50]}...\"")
        text = result[0]["generated_text"].replace("\n", "\n         ")
        print(f"  Output: {text}")


def main():
    capture = io.StringIO()

    class Tee:
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
    print("   GPT-2  —  Text Generation Demo (no fine-tuning)")
    print("=" * 60)
    print("\n  Model: gpt2 (124M parameters, decoder-only transformer)")
    print("  Source: Hugging Face (pre-trained by OpenAI)")
    print("  This model predicts the next token autoregressively.\n")

    demo_text_generation()
    demo_temperature_comparison()
    demo_fill_in_context()

    print("\n" + "=" * 60)
    print("   Key LLM Concepts Demonstrated:")
    print("=" * 60)
    print("  - Autoregressive generation: predict one token at a time")
    print("  - Temperature: controls randomness (0 = deterministic, >1 = creative)")
    print("  - Top-k sampling: only consider the k most likely next tokens")
    print("  - Top-p (nucleus): sample from smallest set of tokens summing to p")
    print("  - Repetition penalty: discourage repeating the same phrases")
    print("  - The model was never trained on these specific prompts")
    print("    — it generalises from patterns seen in its training data")
    print("=" * 60 + "\n")

    sys.stdout = original_stdout

    out_path = os.path.join(OUTPUT_DIR, "output_gpt2.txt")
    with open(out_path, "w") as f:
        f.write(capture.getvalue())
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
