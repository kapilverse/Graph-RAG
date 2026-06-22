"""
Global search — community-centric retrieval (spec §3 Stage 6 Strategy B).

Flow:
  1. Embed the query, find the most relevant community summaries (top-K).
  2. For each community, pull its key entities and their chunks.
  3. Return community summaries + entities for narrative context.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class GlobalRetriever:
    """Community-centric retrieval over community summaries."""

    def __init__(self, embedder, qdrant, neo4j) -> None:
        self.embedder = embedder
        self.qdrant = qdrant
        self.neo4j = neo4j

    def search(self, query: str, top_k: Optional[int] = None) -> dict:
        top_k = top_k or settings.community_top_k
        query_vec = self.embedder.encode([query])[0]

        # 1. Vector search over community summaries.
        communities = self.qdrant.search_communities(query_vec, top_k=top_k)
        if not communities:
            logger.info("Global: no communities indexed")
            return {"communities": [], "entities": []}

        # 2. Pull key entities per community from the graph.
        community_ids = [c["community_id"] for c in communities if c.get("community_id")]
        enriched = self.neo4j.get_entity_communities(community_ids) if community_ids else []
        # Merge graph entity data into the community hits.
        by_id = {c["community_id"]: c for c in enriched}
        for comm in communities:
            extra = by_id.get(comm["community_id"])
            if extra:
                comm["entities"] = [e.get("name", "") for e in extra.get("entities", [])]
                if not comm.get("summary"):
                    comm["summary"] = extra.get("summary", "")

        all_entities = []
        for comm in communities:
            for name in comm.get("entities", []):
                if name:
                    all_entities.append({"name": name})

        return {"communities": communities, "entities": all_entities}
