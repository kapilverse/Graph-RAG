"""
FastAPI request/response schemas for the Graph RAG API.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    """Request body for /ingest."""
    path: str = Field(..., description="File path or directory to ingest")
    reset: bool = Field(False, description="Clear existing data before ingest")


class IngestResponse(BaseModel):
    success: bool = True
    files: List[dict] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """Request body for /query."""
    question: str = Field(..., min_length=1)
    strategy: str = Field("hybrid", description="local | global | hybrid")
    top_k: Optional[int] = Field(None, description="Override default top-K")


# ---------------------------------------------------------------------------
# Graph exploration (D3 subgraph)
# ---------------------------------------------------------------------------
class GraphExploreRequest(BaseModel):
    entity_name: str = Field(..., description="Center entity name")
    limit: int = Field(20, ge=1, le=100)


class GraphNode(BaseModel):
    id: str
    name: str = ""
    type: str = ""
    description: str = ""
    label: str = ""


class GraphLink(BaseModel):
    source: str
    target: str
    type: str = "RELATED_TO"


class GraphExploreResponse(BaseModel):
    nodes: List[GraphNode] = []
    links: List[GraphLink] = []


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class DocumentInfo(BaseModel):
    source_doc: str
    num_chunks: Optional[int] = None


# ---------------------------------------------------------------------------
# Communities
# ---------------------------------------------------------------------------
class CommunityInfo(BaseModel):
    community_id: str
    summary: str = ""
    entities: List[str] = []
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
class StatsResponse(BaseModel):
    neo4j: dict = {}
    qdrant: dict = {}
    llm_provider: str = "none"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    neo4j: bool = False
    qdrant: bool = False
    llm: bool = False
    llm_provider: str = "none"
