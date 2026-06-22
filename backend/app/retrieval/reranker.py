"""
Re-ranking (spec §3 Stage 6 Strategy A step 4).

Cross-encoder reranking via BGE-reranker-v2-m3. Cheap, lazy-loaded, optional:
falls back to vector scores if the model can't load.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def rerank_chunks(
    query: str,
    chunks: List[dict],
    top_k: Optional[int] = None,
) -> List[dict]:
    """Re-rank a list of chunk dicts by cross-encoder score against the query.

    Each chunk dict must have 'text' and 'chunk_id'. Returns a new sorted list
    with an updated 'score' field. Falls back to original order on failure.
    """
    if not chunks:
        return []

    try:
        from app.vector.embedder import get_reranker
        reranker = get_reranker()
        docs = [c.get("text", "") for c in chunks]
        ranked = reranker.rank(query, docs, top_k=top_k)
        return [
            {**chunks[idx], "score": float(score)}
            for idx, score in ranked
        ]
    except Exception as exc:  # noqa: BLE001 — reranking is best-effort
        logger.warning("Reranker unavailable, using vector scores: %s", exc)
        ordered = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
        return ordered[:top_k] if top_k else ordered
