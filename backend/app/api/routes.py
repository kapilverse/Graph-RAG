"""
FastAPI routes for the Graph RAG API.

Endpoints:
  POST /ingest          — ingest a file or directory
  POST /query           — query the knowledge graph
  POST /graph/explore   — fetch a subgraph around an entity (for D3)
  GET  /documents       — list ingested documents
  GET  /communities     — list detected communities
  GET  /stats           — graph + vector store statistics
  GET  /health          — service health check
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    CommunityInfo,
    DocumentInfo,
    GraphExploreRequest,
    GraphExploreResponse,
    GraphLink,
    GraphNode,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    StatsResponse,
)
from app.config import settings
from app.llm.client import llm
from app.llm.schemas import QueryResult
from app.pipeline import GraphRAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level pipeline singleton (lazy init on first request).
_pipeline: Optional[GraphRAGPipeline] = None


def _get_pipeline() -> GraphRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = GraphRAGPipeline(connect=True)
    return _pipeline


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health():
    p = _get_pipeline()
    neo4j_ok = False
    qdrant_ok = False
    try:
        neo4j_ok = p.neo4j.ping()
    except Exception:  # noqa: BLE001
        pass
    try:
        neo4j_ok = neo4j_ok and bool(p.qdrant.client.list_collections())
        qdrant_ok = True
    except Exception:  # noqa: BLE001
        pass
    return HealthResponse(
        neo4j=neo4j_ok,
        qdrant=qdrant_ok,
        llm=llm.is_configured,
        llm_provider=llm.primary_provider_name if llm.is_configured else "none",
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
@router.get("/stats", response_model=StatsResponse)
async def stats():
    p = _get_pipeline()
    s = p.stats()
    return StatsResponse(**s)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    p = _get_pipeline()
    try:
        from pathlib import Path
        path = Path(req.path)
        if path.is_dir():
            results = p.ingest_dir(str(path), reset=req.reset)
        elif path.is_file():
            results = [p.ingest(str(path), reset=req.reset)]
        else:
            raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
        return IngestResponse(files=results)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResult)
async def query(req: QueryRequest):
    p = _get_pipeline()
    try:
        result = p.query(req.question, strategy=req.strategy, top_k=req.top_k)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Graph explore (subgraph for D3 visualization)
# ---------------------------------------------------------------------------
@router.post("/graph/explore", response_model=GraphExploreResponse)
async def graph_explore(req: GraphExploreRequest):
    p = _get_pipeline()
    try:
        data = p.neo4j.explore_neighborhood(req.entity_name, limit=req.limit)
        nodes = [
            GraphNode(
                id=n.get("id", n.get("name", "")),
                name=n.get("name", ""),
                type=n.get("type", ""),
                description=n.get("description", ""),
                label=n.get("name", ""),
            )
            for n in data.get("nodes", [])
        ]
        links = [
            GraphLink(source=l.get("source", ""), target=l.get("target", ""), type=l.get("type", "RELATED_TO"))
            for l in data.get("links", [])
        ]
        return GraphExploreResponse(nodes=nodes, links=links)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph explore error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    p = _get_pipeline()
    try:
        stats = p.neo4j.stats()
        # Return basic info; if needed, fetch from graph directly.
        return [DocumentInfo(source_doc="all", num_chunks=stats.get("chunk", 0))]
    except Exception as exc:  # noqa: BLE001
        logger.exception("List documents error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------
@router.get("/communities", response_model=list[CommunityInfo])
async def list_communities(top_k: int = 20):
    p = _get_pipeline()
    try:
        # Search with a dummy vector to list all communities.
        dim = p.embedder.dim
        dummy = [0.0] * dim
        hits = p.qdrant.search_communities(dummy, top_k=top_k * 5)
        # Enrich with graph data.
        cids = [h["community_id"] for h in hits if h.get("community_id")]
        graph_data = p.neo4j.get_entity_communities(cids) if cids else []
        by_id = {g["community_id"]: g for g in graph_data}
        communities = []
        for h in hits:
            cid = h.get("community_id", "")
            extra = by_id.get(cid, {})
            communities.append(CommunityInfo(
                community_id=cid,
                summary=h.get("summary") or extra.get("summary", ""),
                entities=h.get("entities") or [e["name"] for e in extra.get("entities", [])],
                score=h.get("score"),
            ))
        return communities[:top_k]
    except Exception as exc:  # noqa: BLE001
        logger.exception("List communities error")
        raise HTTPException(status_code=500, detail=str(exc))
