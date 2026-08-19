"""
Fine-tuning vs Transfer Learning — demo
==========================================
Take a pre-trained model and adapt it to your specific task.

Transfer Learning = use a pre-trained model's knowledge as-is (no training)
Fine-tuning      = retrain some/all layers on your own data

This demo:
  1. Uses DistilBERT zero-shot (no fine-tuning) on sentiment data
  2. Fine-tunes DistilBERT on the same data for a few epochs
  3. Compares accuracy before vs after fine-tuning

Run:
    python fine_tuning_demo.py
"""

import os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    pipeline,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_TEXTS = [
    ("This movie was absolutely fantastic!", 1),
    ("I loved every minute of it", 1),
    ("Great acting and wonderful storyline", 1),
    ("The film exceeded all my expectations", 1),
    ("A masterpiece of modern cinema", 1),
    ("Brilliant performance by the lead actor", 1),
    ("Highly recommend this to everyone", 1),
    ("One of the best films I have ever seen", 1),
    ("Incredible visuals and amazing soundtrack", 1),
    ("A delightful and heartwarming story", 1),
    ("This was terrible and boring", 0),
    ("Worst movie I have ever watched", 0),
    ("Complete waste of time and money", 0),
    ("The plot made absolutely no sense", 0),
    ("Awful acting and terrible direction", 0),
    ("I could not wait for it to end", 0),
    ("Disappointing from start to finish", 0),
    ("A total disaster of a film", 0),
    ("Boring and predictable storyline", 0),
    ("I regret watching this movie", 0),
]

TEST_TEXTS = [
    ("An outstanding and moving film", 1),
    ("I really enjoyed this movie", 1),
    ("Absolutely dreadful experience", 0),
    ("Would not recommend to anyone", 0),
    ("Beautiful cinematography and great script", 1),
    ("The worst film of the year", 0),
    ("A truly magical experience", 1),
    ("So bad I walked out halfway", 0),
]


def run_fine_tuning_demo():
    device = torch.device("cpu")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    lines = [
        "=" * 70, "  FINE-TUNING vs TRANSFER LEARNING  —  Demo", "=" * 70,
        f"  Base model: {model_name}",
        f"  Task: binary sentiment classification",
        f"  Train: {len(TRAIN_TEXTS)} examples,  Test: {len(TEST_TEXTS)} examples", "",
    ]

    # --- 1. Zero-shot (transfer learning, no fine-tuning) ---
    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    correct = 0
    lines.append("  1. Zero-shot (pre-trained SST-2 model, no fine-tuning):")
    for text, label in TEST_TEXTS:
        pred = clf(text)[0]
        pred_label = 1 if pred["label"] == "POSITIVE" else 0
        match = pred_label == label
        correct += int(match)
        lines.append(f"     {'✓' if match else '✗'} \"{text[:45]:45s}\" → {pred['label']} ({pred['score']:.3f})")
    zs_acc = correct / len(TEST_TEXTS)
    lines += [f"     Accuracy: {zs_acc:.2%}", ""]

    # --- 2. Fine-tune from scratch ---
    lines.append("  2. Fine-tuning DistilBERT (3 epochs on our tiny dataset):")
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_texts_only = [t for t, _ in TRAIN_TEXTS]
    train_labels = [l for _, l in TRAIN_TEXTS]
    enc = tokenizer(train_texts_only, padding=True, truncation=True, max_length=64, return_tensors="pt")
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(train_labels))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    start = time.perf_counter()
    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        lines.append(f"     Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")
    ft_time = time.perf_counter() - start

    # evaluate fine-tuned model
    model.eval()
    correct = 0
    lines.append(f"     Training time: {ft_time:.1f}s")
    lines.append("     Test results:")
    for text, label in TEST_TEXTS:
        enc = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc.to(device)).logits
        pred_label = torch.argmax(logits, dim=-1).item()
        match = pred_label == label
        correct += int(match)
        conf = torch.softmax(logits, dim=-1).max().item()
        pred_str = "POSITIVE" if pred_label == 1 else "NEGATIVE"
        lines.append(f"     {'✓' if match else '✗'} \"{text[:45]:45s}\" → {pred_str} ({conf:.3f})")
    ft_acc = correct / len(TEST_TEXTS)
    lines += [f"     Accuracy: {ft_acc:.2%}", ""]

    lines += [
        "  Summary:",
        f"    Zero-shot (SST-2 pre-trained) : {zs_acc:.2%}",
        f"    Fine-tuned (3 epochs, 20 ex.) : {ft_acc:.2%}",
        "",
        "  Key concepts:",
        "    - Transfer learning: use a model trained on task A for task B (no retraining)",
        "    - Fine-tuning: retrain the model's weights on your specific data",
        "    - Even 20 examples + 3 epochs can adapt a pre-trained model",
        "    - Pre-trained models already understand language — you just teach them your task",
        "=" * 70,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_fine_tuning_demo()
