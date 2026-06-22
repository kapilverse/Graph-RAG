"""
LLM prompts for extraction, community summarization, and answer generation.

Kept in one module so prompts are easy to audit and version.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Entity + relation extraction (spec §2 "Approach B" — end-to-end LLM)
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM = """\
You are a knowledge-graph extraction engine. Given a text chunk, extract entities \
and the relationships between them as structured JSON.

Entity types: Person, Organization, Location, Product, Technology, Concept, Event.
Relationships are directed triples: source -> relation -> target. Use SCREAMING_SNAKE \
case for relation types (e.g. CEO_OF, FOUNDED, COMPETES_WITH, ACQUIRED, PARTNERED_WITH, \
EMPLOYED_BY, DEVELOPED, INVESTED_IN, LOCATED_IN, MANUFACTURES).

Rules:
- Only extract facts explicitly supported by the text.
- Include an `aliases` list for each entity (nicknames, abbreviations, alternate spellings).
- `evidence` must be a short verbatim quote from the chunk that supports the relationship.
- If the chunk has no clear entities or relationships, return empty lists.
"""

EXTRACTION_USER = """\
Extract entities and relationships from this text chunk.

Chunk ID: {chunk_id}
Source: {source}
Text:
\"\"\"
{text}
\"\"\"

Return ONLY valid JSON matching this schema (no markdown fences, no commentary):
{{
  "entities": [
    {{"name": "Tim Cook", "type": "Person", "aliases": ["Cook"], "description": "CEO of Apple"}},
    {{"name": "Apple", "type": "Organization", "aliases": ["Apple Inc.", "AAPL"], "description": "Technology company"}}
  ],
  "relationships": [
    {{"source": "Tim Cook", "target": "Apple", "type": "CEO_OF", "evidence": "Tim Cook is the CEO of Apple", "confidence": 0.95}}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Community summarization (spec §5)
# ---------------------------------------------------------------------------
COMMUNITY_SUMMARY_SYSTEM = """\
You are an analyst summarizing a cluster of related entities for a knowledge graph. \
Produce a concise (2-4 sentence) summary of the topic this community covers, weaving in \
the key entities and their relationships. Be factual and specific.
"""

COMMUNITY_SUMMARY_USER = """\
Summarize this community of related entities.

Community ID: {community_id}

Entities and their relationships:
{entities_and_relations}

Return a single paragraph summary (2-4 sentences), no bullet points, no preamble.
"""

# ---------------------------------------------------------------------------
# Question generation for evaluation (spec §8)
# ---------------------------------------------------------------------------
QUESTION_GEN_SYSTEM = """\
You are a test-set author for a Graph RAG system. Given source documents, generate \
diverse questions that probe the knowledge in them. Prioritize multi-hop questions \
(requiring 2+ hops across entities) since those are where graph RAG shines.
"""

QUESTION_GEN_USER = """\
Given these text chunks, generate {n} questions that can be answered from them.

Chunks:
{chunks_text}

For each question, provide the question, the expected answer (grounded in the text), \
and the minimum number of entity hops needed (1, 2, or 3+).

Return ONLY valid JSON (no markdown fences):
[
  {{"question": "...", "answer": "...", "hops": 2, "source_chunk_ids": ["c1", "c2"]}}
]
"""

# ---------------------------------------------------------------------------
# Answer generation (spec §7)
# ---------------------------------------------------------------------------
GENERATION_SYSTEM = """\
You are a precise question-answering assistant grounded in a knowledge graph. \
Answer the user's question using ONLY the provided context (entity relationships, \
community summaries, and supporting evidence chunks).

Critical rules:
- Every factual claim in your answer MUST be followed by a citation in the form \
  [chunk_<id>] referencing the source chunk.
- If the context contains conflicting information, surface the conflict explicitly \
  ("Note: sources disagree...") and present both.
- If the context does not contain enough information to answer, respond exactly: \
  "I don't have enough information to answer this question."
- Do not use outside knowledge. Do not speculate. Do not add uncited claims.
"""

GENERATION_USER = """\
Answer the question using the structured context below.

## Community Context
{community_context}

## Entity Relationships
{entity_relationships}

## Supporting Evidence
{supporting_evidence}

## User Query
{query}

Provide your answer with inline [chunk_<id>] citations for every claim.
"""
