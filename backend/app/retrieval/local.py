"""
Local search — entity-centric retrieval (spec §3 Stage 6 Strategy A).

Flow:
  1. Extract entities from the query (LLM or lightweight keyword match).
  2. For each query entity, traverse the graph (1-2 hops) to neighbors.
  3. Fetch the full text of evidence chunks attached to those entities/edges.
  4. Re-rank chunks with the cross-encoder and return the top-K.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from app.config import settings
from app.llm.client import LLMError, llm
from app.retrieval.reranker import rerank_chunks

logger = logging.getLogger(__name__)


class LocalRetriever:
    """Entity-centric retrieval over the knowledge graph."""

    def __init__(self, embedder, qdrant, neo4j) -> None:
        self.embedder = embedder
        self.qdrant = qdrant
        self.neo4j = neo4j

    def search(self, query: str, top_k: Optional[int] = None) -> dict:
        top_k = top_k or settings.rerank_top_k
        # 1. Extract entities from the query.
        query_entities = self._extract_query_entities(query)
        if not query_entities:
            logger.info("Local: no entities found in query, falling back to vector search")
            query_vec = self.embedder.encode([query])[0]
            chunks = self.qdrant.search_chunks(query_vec, top_k=top_k)
            return {"chunks": chunks, "entities": [], "relationships": [], "query_entities": []}

        # 2. Traverse the graph for each entity; gather chunks + relationships.
        all_chunks: dict[str, dict] = {}
        entities_out, rels_out = [], []
        for ent_name in query_entities:
            result = self.neo4j.get_neighbors(
                ent_name,
                depth=settings.local_traversal_depth,
                limit=settings.local_neighbor_limit,
            )
            for chunk_id in result.get("chunk_ids", []):
                if chunk_id and chunk_id not in all_chunks:
                    all_chunks[chunk_id] = {"chunk_id": chunk_id}
            entities_out.extend(result.get("entities", []))
            rels_out.extend(result.get("relationships", []))

        # 3. Fetch full text for the gathered chunk ids.
        chunk_ids = list(all_chunks.keys())
        fetched = self.qdrant.fetch_chunks(chunk_ids) if chunk_ids else []
        # Merge any missing chunks from a vector fallback if graph found nothing.
        if not fetched:
            query_vec = self.embedder.encode([query])[0]
            fetched = self.qdrant.search_chunks(query_vec, top_k=top_k * 2)

        # 4. Re-rank by cross-encoder score.
        ranked = rerank_chunks(query, fetched, top_k=top_k)
        return {
            "chunks": ranked,
            "entities": _dedupe_entities(entities_out),
            "relationships": rels_out,
            "query_entities": query_entities,
        }

    # ------------------------------------------------------------------
    def _extract_query_entities(self, query: str) -> List[str]:
        """Use the LLM to pull entity names out of the query; fall back to keywords."""
        if llm.is_configured:
            try:
                data = llm.extract_json(
                    system="Extract the named entities from the user's question. "
                           "Return JSON {\"entities\": [\"name\", ...]}.",
                    user=query,
                )
                ents = data.get("entities", []) if isinstance(data, dict) else []
                if ents:
                    return [str(e).strip() for e in ents if str(e).strip()]
            except LLMError as exc:
                logger.warning("LLM entity extraction failed, using keyword fallback: %s", exc)

        # Keyword fallback: capitalized tokens / acronyms.
        words = re.findall(r"\b([A-Z][a-zA-Z]+|[A-Z]{2,})\b", query)
        return list(dict.fromkeys(words))  # dedupe, preserve order


def _dedupe_entities(entities: List[dict]) -> List[dict]:
    seen, out = set(), []
    for e in entities:
        key = e.get("name", "").lower()
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    return out
