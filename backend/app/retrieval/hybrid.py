"""
Hybrid retrieval + answer generation (spec §3 Stage 6 Strategy C + Stage 7).

Combines local (entity-centric) and global (community-centric) results, assembles a
structured context, and calls the LLM to generate a cited answer.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import settings
from app.llm.client import LLMError, llm
from app.llm.prompts import GENERATION_SYSTEM, GENERATION_USER
from app.llm.schemas import QueryResult
from app.retrieval.global_search import GlobalRetriever
from app.retrieval.local import LocalRetriever

logger = logging.getLogger(__name__)

_STRATEGIES = ("local", "global", "hybrid")
_INSUFFICIENT_MARKER = "I don't have enough information"


class HybridRetriever:
    """The default query engine: local + global, merged, then LLM generation."""

    def __init__(self, embedder, qdrant, neo4j) -> None:
        self.local = LocalRetriever(embedder, qdrant, neo4j)
        self.global_ = GlobalRetriever(embedder, qdrant, neo4j)

    def query(self, question: str, strategy: str = "hybrid",
              top_k: Optional[int] = None) -> QueryResult:
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {_STRATEGIES}")

        # --- Retrieve ---
        local_result = global_result = {"chunks": [], "entities": [], "communities": []}
        if strategy in ("local", "hybrid"):
            local_result = self.local.search(question, top_k=top_k)
        if strategy in ("global", "hybrid"):
            global_result = self.global_.search(question, top_k=top_k)

        merged = _merge(local_result, global_result)

        # --- Generate ---
        answer, citations, insufficient = self._generate(question, merged)
        return QueryResult(
            answer=answer,
            strategy=strategy,
            chunks=merged["chunks"][: settings.rerank_top_k],
            entities=merged["entities"],
            relationships=merged.get("relationships", []),
            communities=merged["communities"],
            citations=citations,
            insufficient_information=insufficient,
        )

    # ------------------------------------------------------------------
    def _generate(self, question: str, merged: dict) -> tuple[str, list[str], bool]:
        # Build the structured context per spec §7.
        community_ctx = _community_context(merged.get("communities", []))
        entity_ctx = _entity_relationships(merged.get("relationships", []),
                                           merged.get("entities", []))
        evidence_ctx = _evidence(merged.get("chunks", []))

        if not llm.is_configured:
            fallback = (
                "LLM not configured. Retrieved context below — set GEMINI_API_KEY or "
                "HF_TOKEN to generate an answer.\n\n" + evidence_ctx
            )
            return fallback, [], False

        try:
            answer = llm.chat(
                system=GENERATION_SYSTEM,
                user=GENERATION_USER.format(
                    community_context=community_ctx,
                    entity_relationships=entity_ctx,
                    supporting_evidence=evidence_ctx,
                    query=question,
                ),
                temperature=settings.generation_temperature,
            ).strip()
        except LLMError as exc:
            logger.error("Generation failed: %s", exc)
            return f"Generation error: {exc}", [], False

        insufficient = _INSUFFICIENT_MARKER.lower() in answer.lower()
        citations = sorted(set(re.findall(r"\[chunk_[^\]]+\]", answer)))
        return answer, citations, insufficient


# ---------------------------------------------------------------------------
# Context assembly (spec §7 format)
# ---------------------------------------------------------------------------
def _community_context(communities: list) -> str:
    if not communities:
        return "(no relevant communities)"
    lines = []
    for c in communities:
        summary = c.get("summary") or f"Community {c.get('community_id')}"
        members = ", ".join(c.get("entities", [])[:5])
        lines.append(f"- {summary}" + (f" (entities: {members})" if members else ""))
    return "\n".join(lines)


def _entity_relationships(rels: list, entities: list) -> str:
    if not rels:
        # Fall back to just listing entities.
        names = ", ".join(e.get("name", "") for e in entities if e.get("name"))
        return f"(entities mentioned: {names})" if names else "(no entities)"
    lines = []
    for r in rels:
        src = r.get("source", "?")
        tgt = r.get("target", "?")
        rtype = r.get("type", "RELATED_TO")
        chunks = r.get("evidence_chunk_ids") or []
        cite = ", ".join(chunks[:3]) if chunks else ""
        line = f"- {src} ({rtype}) {tgt}"
        if cite:
            line += f" [Source: {cite}]"
        lines.append(line)
    return "\n".join(lines)


def _evidence(chunks: list) -> str:
    if not chunks:
        return "(no supporting chunks retrieved)"
    lines = []
    for c in chunks:
        cid = c.get("chunk_id", "?")
        text = (c.get("text", "") or "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + "..."
        lines.append(f"[{cid}] {text}")
    return "\n".join(lines)


def _merge(local_result: dict, global_result: dict) -> dict:
    """Merge local + global results: dedupe chunks, combine entities/communities."""
    seen_chunks = set()
    chunks = []
    for c in local_result.get("chunks", []) + global_result.get("chunks", []):
        cid = c.get("chunk_id")
        if cid and cid not in seen_chunks:
            seen_chunks.add(cid)
            chunks.append(c)

    # Dedupe entities by name.
    seen_ents = set()
    entities = []
    for e in local_result.get("entities", []) + global_result.get("entities", []):
        name = (e.get("name") or "").lower()
        if name and name not in seen_ents:
            seen_ents.add(name)
            entities.append(e)

    communities = global_result.get("communities", [])
    relationships = local_result.get("relationships", global_result.get("relationships", []))
    return {
        "chunks": chunks,
        "entities": entities,
        "communities": communities,
        "relationships": relationships,
    }
