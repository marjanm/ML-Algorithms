"""
RAG — Retrieval Augmented Generation demo
============================================
Give an LLM access to external documents so it can answer questions
based on YOUR data, not just what it memorised during pre-training.

Pipeline:
  1. EMBED documents into vectors  (sentence-transformers)
  2. RETRIEVE the most relevant chunks for a query  (cosine similarity)
  3. GENERATE an answer using the retrieved context  (GPT-2 or any LM)

This demo runs entirely locally — no API keys needed.

Run:
    pip install sentence-transformers
    python rag_demo.py
"""

import os
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

DOCUMENTS = [
    "Python was created by Guido van Rossum and first released in 1991. It emphasises code readability.",
    "PyTorch is an open-source machine learning framework developed by Meta AI. It uses dynamic computation graphs.",
    "TensorFlow was developed by Google Brain. It was released in 2015 and supports static computation graphs.",
    "Transformers were introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017.",
    "BERT stands for Bidirectional Encoder Representations from Transformers. It was developed by Google in 2018.",
    "GPT stands for Generative Pre-trained Transformer. GPT-2 was released by OpenAI in 2019.",
    "RAG combines retrieval from a knowledge base with text generation to produce grounded answers.",
    "Fine-tuning adjusts a pre-trained model's weights on task-specific data. LoRA makes this more efficient.",
    "Cosine similarity measures the angle between two vectors. It ranges from -1 (opposite) to 1 (identical).",
    "The attention mechanism lets the model focus on relevant parts of the input for each output token.",
]

QUERIES = [
    "Who created Python?",
    "What is the difference between PyTorch and TensorFlow?",
    "When were transformers introduced?",
    "What does BERT stand for?",
    "How does RAG work?",
]


def run_rag_demo():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install sentence-transformers:  pip install sentence-transformers")
        with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
            f.write("Run: pip install sentence-transformers\n")
        return

    from transformers import pipeline as hf_pipeline

    lines = [
        "=" * 75, "  RAG  —  Retrieval Augmented Generation Demo", "=" * 75, "",
        f"  Knowledge base: {len(DOCUMENTS)} documents",
        f"  Queries       : {len(QUERIES)}", "",
    ]

    # --- STEP 1: Embed documents ---
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(DOCUMENTS)
    lines.append("  Step 1: Embedded all documents into 384-dim vectors.")

    # --- STEP 2: For each query, retrieve top-k documents ---
    lines.append("  Step 2: Retrieve most relevant documents per query.")
    lines.append("  Step 3: Generate answer using retrieved context.\n")

    generator = hf_pipeline("text-generation", model="gpt2", max_new_tokens=60)

    for query in QUERIES:
        query_emb = embedder.encode([query])
        similarities = np.dot(doc_embeddings, query_emb.T).squeeze()
        top_k_idx = np.argsort(similarities)[-3:][::-1]

        lines.append(f"  Q: {query}")
        lines.append(f"  Retrieved (top 3):")
        for rank, idx in enumerate(top_k_idx, 1):
            lines.append(f"    {rank}. [{similarities[idx]:.3f}] {DOCUMENTS[idx][:80]}...")

        context = " ".join([DOCUMENTS[i] for i in top_k_idx])
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        answer = generator(prompt, do_sample=False)[0]["generated_text"]
        answer_only = answer.split("Answer:")[-1].strip().split("\n")[0]
        lines.append(f"  Generated answer: {answer_only[:120]}")
        lines.append("")

    lines += [
        "  Key concepts:",
        "    - Embedding: convert text to dense vectors that capture meaning",
        "    - Retrieval: find relevant docs via cosine similarity / dot product",
        "    - Generation: feed retrieved context + query into an LLM",
        "    - RAG lets LLMs answer about YOUR documents without retraining",
        "    - Production RAG uses vector databases (Pinecone, Chroma, FAISS)",
        "=" * 75,
    ]
    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(os.path.join(OUTPUT_DIR, "output.txt"), "w") as f:
        f.write(output_text)


if __name__ == "__main__":
    run_rag_demo()
