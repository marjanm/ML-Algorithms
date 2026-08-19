"""
Multi-modal Models (CLIP) — Demo
==================================
CLIP (Contrastive Language-Image Pretraining) by OpenAI learns a shared
embedding space for images AND text. This enables:

1. Zero-shot classification — classify images using text descriptions
2. Image-text similarity — rank images by relevance to a query
3. Cross-modal retrieval — find images from text, or text from images

Uses the HuggingFace transformers + PIL for CLIP inference.
Falls back to a simulated demo if CLIP model is too large to download.
"""

import os
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


def try_real_clip():
    """Try loading real CLIP model. Returns False if dependencies missing."""
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import torch

        log("  Loading CLIP model (openai/clip-vit-base-patch32)...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

        # Create simple synthetic images (colored squares)
        colors = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
        }
        images = []
        image_names = []
        for name, rgb in colors.items():
            img = Image.new("RGB", (224, 224), rgb)
            images.append(img)
            image_names.append(f"{name} square")

        # Text queries
        text_queries = [
            "a photo of something red",
            "a blue colored object",
            "the color of grass",
            "sunset colors",
            "royal purple color",
        ]

        log(f"  Computing image-text similarities...")

        inputs = processor(text=text_queries, images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_text  # (n_texts, n_images)
            probs = logits.softmax(dim=-1).numpy()

        log(f"\n  Image-Text Similarity Matrix (rows=text, cols=image):")
        header = "  " + " " * 30 + "".join(f"{n:>12}" for n in image_names)
        log(header)
        log(f"  {'-' * (30 + 12 * len(image_names))}")
        for i, query in enumerate(text_queries):
            row = f"  {query:<30}" + "".join(f"{probs[i,j]:>12.3f}" for j in range(len(images)))
            log(row)

        # Zero-shot classification
        log(f"\n  Zero-Shot Classification:")
        classify_texts = ["a photo of a cat", "a photo of a dog", "a photo of a car",
                          "a photo of a flower", "a photo of food"]
        for img, name in zip(images, image_names):
            inputs = processor(text=classify_texts, images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs_cls = outputs.logits_per_image.softmax(dim=-1).numpy()[0]
            best = classify_texts[probs_cls.argmax()]
            log(f"    {name:<15} → {best} ({probs_cls.max():.3f})")

        return True, probs, text_queries, image_names

    except Exception as e:
        log(f"  CLIP model unavailable ({type(e).__name__}: {e})")
        log(f"  Running simulated demo instead.\n")
        return False, None, None, None


def simulated_clip_demo():
    """Demonstrate CLIP concepts with simulated embeddings."""
    log("  [Simulated] Creating mock image and text embeddings...")

    np.random.seed(42)
    dim = 512

    image_categories = ["cat", "dog", "car", "sunset", "pizza", "guitar"]
    text_queries = [
        "a photo of a cute cat",
        "a fluffy puppy",
        "a red sports car",
        "beautiful sunset over ocean",
        "delicious pizza with cheese",
        "acoustic guitar on stage",
    ]

    # Simulate: similar concepts get similar embeddings
    base_embeddings = np.random.randn(len(image_categories), dim)
    image_embeddings = base_embeddings + np.random.randn(len(image_categories), dim) * 0.1
    text_embeddings = base_embeddings + np.random.randn(len(text_queries), dim) * 0.15

    # Add cross-category similarity (dog ↔ cat are more similar)
    image_embeddings[0] += image_embeddings[1] * 0.3  # cat ↔ dog
    text_embeddings[0] += text_embeddings[1] * 0.3

    # Normalize
    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    # Similarity matrix
    sim_matrix = text_embeddings @ image_embeddings.T

    # Softmax for probabilities
    temp = 0.07  # CLIP uses a learned temperature
    logits = sim_matrix / temp
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    return probs, text_queries, image_categories


def run_demo():
    log("MULTI-MODAL MODELS (CLIP) — DEMO")
    log("=" * 60)

    log("""
  CLIP learns a shared embedding space where images and text
  that describe the same thing are close together.

  Training: contrastive learning on 400M image-text pairs from the internet.
  The model learns to match images with their captions.
""")

    # Try real CLIP first
    success, probs, text_queries, image_names = try_real_clip()

    if not success:
        probs, text_queries, image_names = simulated_clip_demo()

    # ═══════════════════════════════════════════════════════
    # Similarity Matrix
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("IMAGE-TEXT SIMILARITY MATRIX")
    log("=" * 60)

    header = "  " + " " * 25 + "".join(f"{n:>12}" for n in image_names)
    log(header)
    log(f"  {'-' * (25 + 12 * len(image_names))}")
    for i, query in enumerate(text_queries):
        row = f"  {query[:24]:<25}" + "".join(f"{probs[i,j]:>12.3f}" for j in range(len(image_names)))
        best_match = image_names[probs[i].argmax()]
        row += f"  ← {best_match}"
        log(row)

    # ═══════════════════════════════════════════════════════
    # Zero-shot classification demo
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("ZERO-SHOT CLASSIFICATION")
    log("=" * 60)
    log(f"\n  For each image, CLIP ranks text descriptions by similarity.")
    log(f"  No training on these specific classes — just text descriptions!\n")

    for j, img_name in enumerate(image_names):
        scores = probs[:, j]
        ranking = np.argsort(-scores)
        log(f"  Image: {img_name}")
        for rank, idx in enumerate(ranking[:3], 1):
            log(f"    #{rank}: {text_queries[idx][:40]:<40} score={scores[idx]:.3f}")
        log()

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Similarity heatmap
    ax = axes[0]
    im = ax.imshow(probs, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(image_names)))
    ax.set_xticklabels([n[:10] for n in image_names], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(text_queries)))
    ax.set_yticklabels([q[:25] for q in text_queries], fontsize=9)
    for i in range(len(text_queries)):
        for j in range(len(image_names)):
            ax.text(j, i, f"{probs[i,j]:.2f}", ha="center", va="center", fontsize=8,
                   color="white" if probs[i,j] > 0.5 else "black")
    ax.set_title("Text-Image Similarity (CLIP)")
    plt.colorbar(im, ax=ax)

    # Plot 2: Best match scores
    ax = axes[1]
    best_scores = probs.max(axis=1)
    best_idx = probs.argmax(axis=1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(text_queries)))
    bars = ax.barh(range(len(text_queries)), best_scores, color=colors)
    ax.set_yticks(range(len(text_queries)))
    ax.set_yticklabels([q[:30] for q in text_queries], fontsize=9)
    for i, (score, idx) in enumerate(zip(best_scores, best_idx)):
        ax.text(score + 0.01, i, f"→ {image_names[idx]}", va="center", fontsize=8)
    ax.set_xlabel("Similarity Score")
    ax.set_title("Best Image Match per Text Query")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "clip_multimodal.png"), dpi=150)
    plt.close()
    log(f"\n→ Plot saved: plots/clip_multimodal.png")

    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
  ┌──────────────────┬────────────────────────────────────────┐
  │ Concept          │ Description                            │
  ├──────────────────┼────────────────────────────────────────┤
  │ Contrastive      │ Push matching pairs together,          │
  │ Learning         │ non-matching pairs apart in embedding  │
  │                  │ space. InfoNCE loss.                   │
  │ Zero-shot        │ Classify without training on those     │
  │                  │ classes. Just provide text descriptions.│
  │ Shared embedding │ Images and text live in the SAME       │
  │                  │ vector space. Cosine similarity works. │
  │ Temperature      │ Learned scalar that sharpens the       │
  │                  │ softmax distribution.                  │
  └──────────────────┴────────────────────────────────────────┘

  Multi-modal model landscape:
    • CLIP      — image ↔ text (OpenAI, 2021)
    • ALIGN     — image ↔ text (Google, noisier data)
    • LLaVA     — image + text → text (visual QA)
    • GPT-4V    — image + text → text (OpenAI)
    • Gemini    — image + text + audio + video (Google)
    • DALL-E    — text → image (generation, not matching)

  Applications:
    • Image search by natural language query
    • Content moderation (match images to policy descriptions)
    • Product recommendation (match product images to user queries)
    • Accessibility (describe images for visually impaired users)
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
