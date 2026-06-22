"""
Qdrant vector store.

Stores chunk embeddings (for semantic retrieval) and community-summary embeddings
(for global search). Each chunk carries metadata that lets retrieval jump straight
to the graph: entity_ids[], community_ids[], document_id (spec §3 Stage 4).

Two collections:
- graphrag_chunks        — one point per Chunk
- graphrag_communities   — one point per community summary
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from app.config import settings
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class QdrantStore:
    """Wrapper over qdrant_client for chunk + community collections."""

    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        self._Distance = Distance
        self._VectorParams = VectorParams
        self.client = QdrantClient(url=settings.qdrant_url, timeout=30)
        self._chunks = settings.qdrant_collection_chunks
        self._communities = settings.qdrant_collection_communities

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------
    def init_collections(self, vector_dim: int) -> None:
        """Create collections if they don't exist. Call after embedder is ready."""
        for name in (self._chunks, self._communities):
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=self._VectorParams(size=vector_dim, distance=self._Distance.COSINE),
                )
                logger.info("Created Qdrant collection '%s' (dim=%d)", name, vector_dim)

    def reset_collections(self) -> None:
        """Drop + recreate collections (used by re-ingest)."""
        for name in (self._chunks, self._communities):
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
                logger.info("Dropped Qdrant collection '%s'", name)

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------
    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        from qdrant_client.http.models import PointStruct

        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")
            points.append(
                PointStruct(
                    id=_stable_id(chunk.id),
                    vector=chunk.embedding,
                    payload={
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                        "source_doc": chunk.source_doc,
                        "section_header": chunk.section_header,
                        "page_num": chunk.page_num,
                        "chunk_index": chunk.chunk_index,
                        "entity_ids": chunk.entity_ids,
                        "community_ids": chunk.community_ids,
                        "document_id": chunk.source_doc,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self._chunks, points=points)
        logger.info("Upserted %d chunks into Qdrant", len(points))

    def search_chunks(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_entity_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Vector search over chunks. Optionally filter by entity ids."""
        from qdrant_client.http.models import FieldCondition, Filter, MatchAny

        query_filter = None
        if filter_entity_ids:
            query_filter = Filter(
                must=[FieldCondition(key="entity_ids", match=MatchAny(any=filter_entity_ids))]
            )

        hits = self.client.query_points(
            collection_name=self._chunks,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points
        return [
            {
                "chunk_id": h.payload.get("chunk_id"),
                "text": h.payload.get("text", ""),
                "score": h.score,
                "source_doc": h.payload.get("source_doc", ""),
                "page_num": h.payload.get("page_num"),
                "section_header": h.payload.get("section_header", ""),
                "entity_ids": h.payload.get("entity_ids", []),
                "community_ids": h.payload.get("community_ids", []),
            }
            for h in hits
        ]

    def fetch_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch chunks by their chunk_id (not the Qdrant point id)."""
        if not chunk_ids:
            return []
        from qdrant_client.http.models import FieldCondition, Filter, MatchAny

        points = self.client.scroll(
            collection_name=self._chunks,
            scroll_filter=Filter(
                must=[FieldCondition(key="chunk_id", match=MatchAny(any=chunk_ids))]
            ),
            limit=len(chunk_ids),
            with_payload=True,
            with_vectors=False,
        )[0]
        return [
            {
                "chunk_id": p.payload.get("chunk_id"),
                "text": p.payload.get("text", ""),
                "source_doc": p.payload.get("source_doc", ""),
                "page_num": p.payload.get("page_num"),
                "section_header": p.payload.get("section_header", ""),
                "entity_ids": p.payload.get("entity_ids", []),
            }
            for p in points
        ]

    # ------------------------------------------------------------------
    # Communities
    # ------------------------------------------------------------------
    def upsert_community(self, community_id: str, summary: str, embedding: List[float],
                         entity_names: List[str]) -> None:
        from qdrant_client.http.models import PointStruct

        self.client.upsert(
            collection_name=self._communities,
            points=[
                PointStruct(
                    id=_stable_id(community_id),
                    vector=embedding,
                    payload={
                        "community_id": community_id,
                        "summary": summary,
                        "entities": entity_names,
                    },
                )
            ],
        )

    def search_communities(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        hits = self.client.query_points(
            collection_name=self._communities,
            query=query_vector,
            limit=top_k,
        ).points
        return [
            {
                "community_id": h.payload.get("community_id"),
                "summary": h.payload.get("summary", ""),
                "entities": h.payload.get("entities", []),
                "score": h.score,
            }
            for h in hits
        ]


def _stable_id(key: str) -> str:
    """Qdrant needs an int or UUID id. Hash the chunk/community id to a stable UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
