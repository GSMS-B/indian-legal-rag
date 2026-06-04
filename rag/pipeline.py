"""
rag/pipeline.py
---------------
Orchestrates the full RAG pipeline:
    user query → expand → retrieve → generate answer

Exposes a single function: answer(user_query) → dict
"""

from rag.generator import expand_query, generate_answer
from rag.retriever import retrieve


def answer(user_query: str) -> dict:
    """
    Run the full RAG pipeline for a user query.

    Returns:
        dict with keys:
            - answer   (str):  The LLM-generated cited answer
            - sources  (list): List of chunk metadata dicts used
            - expanded_query (str): The legal-terminology-expanded query
    """
    # Guard: if ChromaDB is not initialized, return a helpful message
    try:
        from rag.retriever import get_collection_count
        count = get_collection_count()
        if count == 0:
            return {
                "answer": (
                    "⚠️ Database not initialized. "
                    "Please run `python scripts/02_embed_store.py` first."
                ),
                "sources": [],
                "expanded_query": "",
            }
    except Exception:
        return {
            "answer": (
                "⚠️ Database not initialized. "
                "Please run `python scripts/02_embed_store.py` first."
            ),
            "sources": [],
            "expanded_query": "",
        }

    # Step 1: Expand query with legal terminology
    expanded_query = expand_query(user_query)

    # Step 2: Retrieve relevant sections using the EXPANDED query
    retrieved_chunks = retrieve(expanded_query, top_k=4)

    # Step 3: Generate answer using the ORIGINAL query (user sees their
    #         own words answered) and the retrieved context
    llm_result = generate_answer(user_query, retrieved_chunks)

    # Build source metadata for display (strip the full text to save space)
    sources = []
    for chunk in retrieved_chunks:
        sources.append(
            {
                "source_label": chunk.get("source_label", ""),
                "section_number": chunk.get("section_number", ""),
                "section_title": chunk.get("section_title", ""),
                "chapter": chunk.get("chapter", ""),
                "act": chunk.get("act", ""),
                "rerank_score": chunk.get("rerank_score", 0.0),
                "text_preview": chunk.get("text", "")[:400],
            }
        )

    return {
        "answer": llm_result["text"],
        "sources": sources,
        "expanded_query": expanded_query,
        "provider_used": llm_result["provider_used"],
    }
