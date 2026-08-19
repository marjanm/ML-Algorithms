"""
LoRA — Low-Rank Adaptation demo
==================================
Fine-tune only a TINY fraction of model weights instead of all of them.

LoRA inserts small trainable matrices (rank-decomposed) next to the frozen
pre-trained weights.  This means:
  - Full fine-tuning: update ALL 66M parameters of DistilBERT
  - LoRA fine-tuning: update ~0.3M parameters (< 0.5%)

Same accuracy, 100× fewer trained params, 10× less memory.

This demo:
  1. Counts parameters in a full fine-tuning setup
  2. Applies LoRA and counts trainable params (shows the massive reduction)
  3. Fine-tunes with LoRA on a tiny sentiment dataset
  4. Compares the two approaches

Run:
    pip install peft
    python lora_demo.py
"""

import os, time
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_TEXTS = [
    ("This movie was absolutely fantastic!", 1),
    ("I loved every minute of it", 1),
    ("Great acting and wonderful storyline", 1),
    ("The film exceeded all my expectations", 1),
    ("Brilliant performance by the lead actor", 1),
    ("This was terrible and boring", 0),
    ("Worst movie I have ever watched", 0),
    ("Complete waste of time and money", 0),
    ("The plot made absolutely no sense", 0),
    ("Awful acting and terrible direction", 0),
]

TEST_TEXTS = [
    ("An outstanding and moving film", 1),
    ("I really enjoyed this movie", 1),
    ("Absolutely dreadful experience", 0),
    ("Would not recommend to anyone", 0),
]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_lora_demo():
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        msg = "Install peft:  pip install peft"
        print(msg)
        with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
            f.write(msg + "\n")
        return

    device = torch.device("cpu")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    lines = [
        "=" * 70, "  LoRA  —  Low-Rank Adaptation Demo", "=" * 70,
        f"  Base model: {model_name}", "",
    ]

    # --- 1. Full model parameter count ---
    full_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    total, trainable = count_params(full_model)
    lines += [
        "  1. Full fine-tuning (all params trainable):",
        f"     Total params     : {total:>12,}",
        f"     Trainable params : {trainable:>12,}",
        f"     Trainable %      : {100 * trainable / total:.2f}%", "",
    ]
    del full_model

    # --- 2. LoRA model ---
    base_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                         # rank of the low-rank matrices (lower = fewer params)
        lora_alpha=16,               # scaling factor; effective lr ≈ lora_alpha / r
        lora_dropout=0.1,            # dropout on LoRA layers
        target_modules=["q_lin", "v_lin"],  # which layers to apply LoRA to (attention Q and V)
    )
    lora_model = get_peft_model(base_model, lora_config)
    total, trainable = count_params(lora_model)
    lines += [
        "  2. LoRA fine-tuning (only low-rank adapters trainable):",
        f"     Total params     : {total:>12,}",
        f"     Trainable params : {trainable:>12,}",
        f"     Trainable %      : {100 * trainable / total:.2f}%",
        f"     LoRA rank (r)    : {lora_config.r}",
        f"     Target modules   : {lora_config.target_modules}",
        "",
    ]

    # --- 3. Train with LoRA ---
    train_texts_only = [t for t, _ in TRAIN_TEXTS]
    train_labels = [l for _, l in TRAIN_TEXTS]
    enc = tokenizer(train_texts_only, padding=True, truncation=True, max_length=64, return_tensors="pt")
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(train_labels))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    lora_model.to(device)
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=5e-4)

    lora_model.train()
    start = time.perf_counter()
    lines.append("  3. Training with LoRA (3 epochs):")
    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        lines.append(f"     Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")
    ft_time = time.perf_counter() - start
    lines.append(f"     Time: {ft_time:.1f}s")

    # --- 4. Evaluate ---
    lora_model.eval()
    correct = 0
    lines.append("\n  4. Test results:")
    for text, label in TEST_TEXTS:
        enc = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = lora_model(**enc).logits
        pred = torch.argmax(logits, dim=-1).item()
        match = pred == label
        correct += int(match)
        lines.append(f"     {'✓' if match else '✗'} \"{text}\" → {'POS' if pred == 1 else 'NEG'}")
    acc = correct / len(TEST_TEXTS)
    lines += [
        f"     Accuracy: {acc:.2%}", "",
        "  Key concepts:",
        "    - LoRA freezes all original weights and adds tiny trainable matrices",
        f"    - We trained {trainable:,} params instead of ~66M (>{100*trainable//total}× reduction)",
        "    - Same accuracy potential, fraction of the memory and compute",
        "    - QLoRA goes further: quantises the frozen weights to 4-bit",
        "    - In practice, LoRA lets you fine-tune 7B+ models on a single GPU",
        "=" * 70,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_lora_demo()
