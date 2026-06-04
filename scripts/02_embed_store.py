"""
scripts/02_embed_store.py
-------------------------
Loads data/chunks.json, embeds every chunk with BGE-base-en-v1.5,
and stores them in a ChromaDB persistent collection.

Run once:  python scripts/02_embed_store.py
"""

import json
import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.json")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── BGE query prefix (used ONLY on queries, never on documents) ──────────────
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── embedding model ──────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# ── ChromaDB collection name ────────────────────────────────────────────────
COLLECTION_NAME = "indian_criminal_law"


def main():
    force = "--force" in sys.argv
    # 1. Load chunks ──────────────────────────────────────────────────────────
    if not os.path.isfile(CHUNKS_PATH):
        print(f"ERROR: Chunks file not found -> {CHUNKS_PATH}")
        print("Run scripts/01_chunk.py first.")
        sys.exit(1)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)

    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    # 2. Load embedding model ─────────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBED_MODEL_NAME} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print("Model loaded.")

    # 3. Initialize ChromaDB ──────────────────────────────────────────────────
    print(f"\nInitializing ChromaDB at {CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Create or get collection with COSINE distance (not default L2)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Check for existing documents
    existing_count = collection.count()
    if existing_count > 0:
        print(f"\nWARNING: Collection already has {existing_count} documents.")
        if not force:
            answer = input("Re-embed and overwrite? (yes/no): ").strip().lower()
            if answer != "yes":
                print("Aborting. Existing embeddings preserved.")
                sys.exit(0)
        # Delete existing collection and recreate
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("Collection cleared.")

    # 4. Embed all chunk texts ────────────────────────────────────────────────
    texts = [c["text"] for c in chunks]
    print(f"\nEmbedding {len(texts)} chunks (batch_size=32) ...")
    # NOTE: Do NOT add BGE query prefix to documents
    embeddings = embed_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print("Embedding complete.")

    # 5. Store in ChromaDB ────────────────────────────────────────────────────
    print("\nStoring chunks in ChromaDB ...")

    # ChromaDB has a batch limit; add in batches of 500
    batch_size = 500
    for start in tqdm(range(0, len(chunks), batch_size), desc="Storing"):
        end = min(start + batch_size, len(chunks))
        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end].tolist()

        ids = [c["chunk_id"] for c in batch_chunks]
        documents = [c["text"] for c in batch_chunks]
        metadatas = [
            {
                "act": c["act"],
                "section_number": c["section_number"],
                "section_title": c["section_title"],
                "chapter": c["chapter"],
                "source_label": c["source_label"],
            }
            for c in batch_chunks
        ]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )

    final_count = collection.count()
    print(f"\nStored {final_count} chunks in ChromaDB.")

    # 6. Validation query ─────────────────────────────────────────────────────
    print("\n-- Validation Query --")
    test_query = "what is the punishment for murder"
    print(f'  Query: "{test_query}"')

    # Embed with BGE prefix (queries only!)
    query_embedding = embed_model.encode(
        BGE_QUERY_PREFIX + test_query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
    )

    print("  Top 5 results:")
    found_murder = False
    for i, (doc_id, meta) in enumerate(
        zip(results["ids"][0], results["metadatas"][0])
    ):
        label = meta["source_label"]
        print(f"    {i+1}. [{doc_id}] {label}")
        if meta["section_number"] in ("101", "103") and "BNS" in meta["act"]:
            found_murder = True

    if found_murder:
        print("\n  OK: Murder section (101 or 103) found - embeddings are working.")
    else:
        print("\n  WARNING: Neither Section 101 nor 103 from BNS appeared "
              "in top 5. Review embeddings.")


if __name__ == "__main__":
    main()
