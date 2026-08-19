"""
Vector Databases & Approximate Nearest Neighbor Search — Demo
================================================================
Shows how vector search works under the hood:

1. Exact (brute-force) search — cosine similarity on all vectors
2. FAISS Flat index           — exact but optimized (BLAS)
3. FAISS IVF index            — approximate, partitioned search
4. FAISS HNSW index           — approximate, graph-based search

Benchmarks: recall@k vs query speed as index type changes.
Uses sentence-transformer embeddings on a small text corpus.
"""

import os
import time
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


def cosine_similarity_manual(query, vectors):
    """Brute-force cosine similarity."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    vectors_norm = vectors / norms  # unit-length vectors so dot product = cosine similarity
    return vectors_norm @ query_norm


def create_synthetic_embeddings(n_vectors=10000, dim=128):
    """Create synthetic embeddings with clustered structure."""
    np.random.seed(42)
    n_clusters = 20
    centers = np.random.randn(n_clusters, dim) * 3
    cluster_ids = np.random.randint(0, n_clusters, n_vectors)
    vectors = centers[cluster_ids] + np.random.randn(n_vectors, dim) * 0.5
    vectors = vectors.astype(np.float32)
    return vectors, cluster_ids


def benchmark_search(index_builder, vectors, queries, k=10, name=""):
    """Benchmark an index: measure build time, query time, recall."""
    # Build
    t0 = time.perf_counter()
    search_fn = index_builder(vectors)
    build_time = time.perf_counter() - t0

    # Ground truth (brute force)
    gt_results = []
    for q in queries:
        sims = cosine_similarity_manual(q, vectors)
        gt_results.append(np.argsort(-sims)[:k])

    # Query
    t0 = time.perf_counter()
    pred_results = []
    for q in queries:
        pred_results.append(search_fn(q, k))
    query_time = (time.perf_counter() - t0) / len(queries)

    # Recall@k
    recalls = []
    for gt, pred in zip(gt_results, pred_results):
        overlap = len(set(gt) & set(pred))
        recalls.append(overlap / k)
    avg_recall = np.mean(recalls)

    return {
        "name": name,
        "build_time": build_time,
        "query_time_ms": query_time * 1000,
        "recall_at_k": avg_recall,
    }


def run_demo():
    log("VECTOR DATABASES & ANN SEARCH — DEMO")
    log("=" * 60)

    n_vectors = 20000
    dim = 128
    n_queries = 50
    k = 10

    vectors, cluster_ids = create_synthetic_embeddings(n_vectors, dim)
    query_idx = np.random.choice(n_vectors, n_queries, replace=False)
    queries = vectors[query_idx]

    log(f"\n  Dataset: {n_vectors} vectors, {dim} dimensions")
    log(f"  Queries: {n_queries}, retrieving top-{k}")

    # ═══════════════════════════════════════════════════════
    # Method 1: Brute Force (NumPy)
    # ═══════════════════════════════════════════════════════
    def build_brute_force(vecs):
        def search(query, k):
            sims = cosine_similarity_manual(query, vecs)  # 1D array: one similarity score per vector, shape (n_vectors,)
            return np.argsort(-sims)[:k]
        return search

    # ═══════════════════════════════════════════════════════
    # Method 2-4: FAISS indices
    # ═══════════════════════════════════════════════════════
    import faiss

    def build_flat(vecs):
        index = faiss.IndexFlatIP(dim)  # inner product (cosine on normalized)
        faiss.normalize_L2(vecs)
        index.add(vecs.copy())
        def search(query, k):
            q = query.reshape(1, -1).copy()
            faiss.normalize_L2(q)
            _, I = index.search(q, k)
            return I[0]
        return search

    def build_ivf(vecs, nlist=50, nprobe=10):
        # nlist=50: partition all vectors into 50 clusters via k-means
        # nprobe=10: at query time, only search the 10 closest clusters (skip the other 40)
        quantizer = faiss.IndexFlatIP(dim)          # brute-force index used to find closest cluster centers
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        vecs_copy = vecs.copy()
        faiss.normalize_L2(vecs_copy)               # normalize so dot product = cosine similarity
        index.train(vecs_copy)                      # run k-means to learn 50 cluster centers
        index.add(vecs_copy)                        # assign each vector to its nearest cluster
        index.nprobe = nprobe                       # how many clusters to search (higher = better recall, slower)
        def search(query, k):
            q = query.reshape(1, -1).copy()
            faiss.normalize_L2(q)
            _, I = index.search(q, k)               # find query's nearest clusters, search only those
            return I[0]
        return search

    def build_hnsw(vecs, M=32, ef_search=64):
        # M=32: each node connects to 32 neighbors in the graph
        # ef_search=64: explore 64 candidates during search (higher = better recall, slower)
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = ef_search
        vecs_copy = vecs.copy()
        faiss.normalize_L2(vecs_copy)
        index.add(vecs_copy)                        # builds the graph: each vector connects to its M nearest neighbors
        def search(query, k):
            q = query.reshape(1, -1).copy()
            faiss.normalize_L2(q)
            _, I = index.search(q, k)               # hop through graph from coarse to fine layers
            return I[0]
        return search

    # ═══════════════════════════════════════════════════════
    # Benchmarks
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("BENCHMARK RESULTS")
    log("=" * 60)

    results = []
    results.append(benchmark_search(build_brute_force, vectors.copy(), queries.copy(), k, "Brute Force (NumPy)"))
    results.append(benchmark_search(build_flat, vectors.copy(), queries.copy(), k, "FAISS Flat (exact)"))
    results.append(benchmark_search(build_ivf, vectors.copy(), queries.copy(), k, "FAISS IVF (approx)"))
    results.append(benchmark_search(build_hnsw, vectors.copy(), queries.copy(), k, "FAISS HNSW (approx)"))

    log(f"\n  {'Index Type':<25} {'Build (s)':>10} {'Query (ms)':>12} {'Recall@{}'.format(k):>12}")
    log(f"  {'-' * 62}")
    for r in results:
        log(f"  {r['name']:<25} {r['build_time']:>10.3f} {r['query_time_ms']:>12.3f} {r['recall_at_k']:>12.3f}")

    # ═══════════════════════════════════════════════════════
    # IVF nprobe sweep (if FAISS available)
    # ═══════════════════════════════════════════════════════
    nprobe_results = []
    log(f"\n{'=' * 60}")
    log("IVF nprobe SWEEP — Recall vs Speed Tradeoff")
    log("=" * 60)

    nprobes = [1, 2, 5, 10, 20, 50]
    for np_val in nprobes:
        def build_ivf_sweep(vecs, nprobe=np_val):
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 50, faiss.METRIC_INNER_PRODUCT)
            vecs_copy = vecs.copy()
            faiss.normalize_L2(vecs_copy)
            index.train(vecs_copy)
            index.add(vecs_copy)
            index.nprobe = nprobe
            def search(query, k):
                q = query.reshape(1, -1).copy()
                faiss.normalize_L2(q)
                _, I = index.search(q, k)
                return I[0]
            return search

        r = benchmark_search(build_ivf_sweep, vectors.copy(), queries.copy(), k, f"IVF nprobe={np_val}")
        nprobe_results.append(r)

    log(f"\n  {'nprobe':>8} {'Query (ms)':>12} {'Recall@{}'.format(k):>12}")
    log(f"  {'-' * 35}")
    for r in nprobe_results:
        log(f"  {r['name'].split('=')[1]:>8} {r['query_time_ms']:>12.3f} {r['recall_at_k']:>12.3f}")

    # ═══════════════════════════════════════════════════════
    # Semantic Search Demo
    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("SEMANTIC SEARCH DEMO (simulated embeddings)")
    log("=" * 60)

    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Neural networks are inspired by the human brain",
        "Random forests use ensemble of decision trees",
        "Python is popular for data science and ML",
        "Transfer learning reuses pretrained model weights",
        "Gradient descent optimizes model parameters",
        "Transformers use self-attention mechanisms",
        "BERT is a bidirectional language model",
        "Convolutional networks excel at image recognition",
        "Reinforcement learning trains agents via rewards",
    ]

    np.random.seed(123)
    doc_embeddings = np.random.randn(len(documents), dim).astype(np.float32)
    # Make semantically similar docs have similar embeddings
    doc_embeddings[0] += doc_embeddings[1] * 0.5  # ML ↔ neural nets
    doc_embeddings[6] += doc_embeddings[7] * 0.5  # transformers ↔ BERT
    doc_embeddings[2] += doc_embeddings[5] * 0.3  # trees ↔ gradient descent

    query_text = "How do language models work?"
    query_emb = (doc_embeddings[6] + doc_embeddings[7]) / 2 + np.random.randn(dim).astype(np.float32) * 0.1

    sims = cosine_similarity_manual(query_emb, doc_embeddings)
    ranked = np.argsort(-sims)

    log(f"\n  Query: \"{query_text}\"")
    log(f"\n  {'Rank':>4} {'Score':>8}  Document")
    log(f"  {'-' * 65}")
    for rank, idx in enumerate(ranked, 1):
        marker = " ◄" if rank <= 3 else ""
        log(f"  {rank:>4} {sims[idx]:>8.4f}  {documents[idx]}{marker}")

    # ═══════════════════════════════════════════════════════
    # Plots
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Index comparison
    ax = axes[0]
    names = [r["name"] for r in results]
    times = [r["query_time_ms"] for r in results]
    recalls = [r["recall_at_k"] for r in results]
    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"][:len(results)]
    ax.barh(names, times, color=colors)
    for i, (t, r) in enumerate(zip(times, recalls)):
        ax.text(t + 0.01, i, f"recall={r:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Query Time (ms)")
    ax.set_title("Index Type: Speed Comparison")

    # Plot 2: nprobe tradeoff
    ax = axes[1]
    if nprobe_results:
        np_times = [r["query_time_ms"] for r in nprobe_results]
        np_recalls = [r["recall_at_k"] for r in nprobe_results]
        ax.plot(np_times, np_recalls, "o-", color="#e74c3c", linewidth=2, markersize=8)
        for t, r, np_val in zip(np_times, np_recalls, nprobes):
            ax.annotate(f"nprobe={np_val}", (t, r), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Query Time (ms)")
        ax.set_ylabel(f"Recall@{k}")
        ax.set_title("IVF: Recall vs Speed Tradeoff")
    else:
        ax.text(0.5, 0.5, "FAISS not installed\n(pip install faiss-cpu)", 
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("IVF nprobe Sweep (unavailable)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "vector_db_benchmark.png"), dpi=150)
    plt.close()
    log(f"\n→ Plot saved: plots/vector_db_benchmark.png")

    # ═══════════════════════════════════════════════════════
    log(f"\n{'=' * 60}")
    log("KEY CONCEPTS")
    log("=" * 60)
    log("""
  ┌────────────────────┬──────────────────────────────────────┐
  │ Index Type         │ Tradeoff                             │
  ├────────────────────┼──────────────────────────────────────┤
  │ Flat (brute force) │ 100% recall, slow for large N       │
  │ IVF (inverted file)│ Partitions space into clusters,      │
  │                    │ only searches nearby partitions.      │
  │                    │ nprobe controls recall/speed.        │
  │ HNSW (graph)       │ Navigable small-world graph.         │
  │                    │ Fast queries, larger memory.          │
  │ PQ (product quant) │ Compresses vectors. Very fast,       │
  │                    │ lower recall. Good for billions.      │
  └────────────────────┴──────────────────────────────────────┘

  Vector DB tools:
    • FAISS       — Meta's library. CPU/GPU. The standard.
    • Chroma      — Python-native, good for prototyping.
    • Pinecone    — Managed service, zero-ops.
    • Weaviate    — Open source, supports hybrid search.
    • Milvus      — Distributed, handles billions of vectors.
    • Qdrant      — Rust-based, fast, filtering support.

  When to use:
    • RAG pipelines (retrieve relevant docs for LLM context)
    • Semantic search (find similar items by meaning)
    • Recommendation (find similar users/items)
    • Deduplication (find near-duplicate documents)
""")

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"\n→ Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_demo()
