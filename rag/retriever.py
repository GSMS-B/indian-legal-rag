"""
rag/retriever.py
----------------
Handles vector retrieval from ChromaDB with per-Act minimum guarantees,
and CrossEncoder reranking.

Models are loaded ONCE at module level — never inside a per-query function.
"""

import os
from typing import Optional

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── constants ────────────────────────────────────────────────────────────────
COLLECTION_NAME = "indian_criminal_law"
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ACT_NAMES = ["BNS 2023", "BNSS 2023", "BSA 2023"]
PER_ACT_K = 4  # retrieve top-4 from each Act -> 12 candidates total
SCORE_THRESHOLD = -4.0  # drop reranked results below this score

# ── module-level model loading (runs once on first import) ───────────────────
_embed_model: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_collection: Optional[chromadb.Collection] = None


def _load_models():
    """Lazy-load models and ChromaDB collection once."""
    global _embed_model, _reranker, _collection

    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)

    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(name=COLLECTION_NAME)


def get_collection_count() -> int:
    """Return the number of documents in the collection, or 0 if not ready."""
    try:
        _load_models()
        return _collection.count()
    except Exception:
        return 0


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Full retrieval pipeline:
      1. Embed query with BGE prefix
      2. Query ChromaDB 3× (one per Act, top-4 each) → 12 candidates
      3. Rerank with CrossEncoder, apply score threshold → return top-k

    Returns a list of dicts with keys:
        text, act, section_number, section_title, chapter,
        source_label, chunk_id, rerank_score
    """
    _load_models()

    # ── Step 1: embed the query ──────────────────────────────────────────
    query_embedding = _embed_model.encode(
        BGE_QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    # ── Step 2: per-Act retrieval ────────────────────────────────────────
    candidates: list[dict] = []
    seen_ids: set[str] = set()

    for act in ACT_NAMES:
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=PER_ACT_K,
            where={"act": act},
        )

        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    **results["metadatas"][0][i],
                }
            )

    # ── Step 3: CrossEncoder reranking ───────────────────────────────────
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    scores = _reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    # Sort descending by rerank score
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # ── Score-threshold filtering ────────────────────────────────────
    # Drop low-scoring irrelevant sections so they never reach the LLM.
    filtered = [r for r in reranked if r["rerank_score"] > SCORE_THRESHOLD]

    if len(filtered) >= top_k:
        return filtered[:top_k]
    else:
        # Not enough passed threshold — return top_k by score regardless
        return reranked[:top_k]
