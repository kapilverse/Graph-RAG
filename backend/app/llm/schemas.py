"""
Pydantic schemas for LLM structured output and internal data flow.

These define the contracts between extraction, graph construction, and retrieval.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Extraction output (what the LLM returns per chunk)
# ---------------------------------------------------------------------------
class ExtractedEntity(BaseModel):
    name: str
    type: str = Field(..., description="Person, Organization, Location, Product, Technology, Concept, Event")
    aliases: List[str] = Field(default_factory=list)
    description: str = ""


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    type: str = Field(..., description="SCREAMING_SNAKE relation type, e.g. CEO_OF, COMPETES_WITH")
    evidence: str = ""
    confidence: float = 0.8


class ExtractionResult(BaseModel):
    """Full extraction output for one chunk."""

    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.relationships


# ---------------------------------------------------------------------------
# Question generation (eval)
# ---------------------------------------------------------------------------
class GeneratedQuestion(BaseModel):
    question: str
    answer: str
    hops: int = 1
    source_chunk_ids: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval / API output
# ---------------------------------------------------------------------------
class ChunkHit(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_doc: str = ""
    page_num: Optional[int] = None
    section_header: str = ""


class EntityHit(BaseModel):
    name: str
    type: str
    description: str = ""


class RelationshipHit(BaseModel):
    source: str
    target: str
    type: str
    confidence: float
    evidence_chunk_ids: List[str] = Field(default_factory=list)


class CommunityHit(BaseModel):
    community_id: str
    summary: str
    entities: List[str] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Full answer payload returned by the query API."""

    answer: str
    strategy: str
    chunks: List[ChunkHit] = Field(default_factory=list)
    entities: List[EntityHit] = Field(default_factory=list)
    relationships: List[RelationshipHit] = Field(default_factory=list)
    communities: List[CommunityHit] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    insufficient_information: bool = False
