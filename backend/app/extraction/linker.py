"""
Entity linking / deduplication (spec §3 Stage 2).

"Apple Inc.", "Apple", "AAPL" should map to one canonical Entity node. We resolve
candidates using a two-pass strategy:

1. **String match**: lowercase, strip suffixes (Inc., Corp., Ltd.), compare.
2. **Embedding similarity**: if string match fails, compare entity name embeddings;
   above a cosine threshold, treat as the same entity.

A canonical id is assigned: <type>:<sanitized_canonical_name>. This id is what gets
written to the graph and stored in chunk.entity_ids.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.llm.schemas import ExtractionResult
from app.llm.schemas import ExtractedEntity

logger = logging.getLogger(__name__)

# Suffixes stripped during canonicalization.
_LEGAL_SUFFIXES = re.compile(
    r"\b(inc|corp|corporation|ltd|limited|llc|co|company|plc|gmbh|sa|ag)\b\.?",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SIMILARITY_THRESHOLD = 0.88  # cosine threshold for embedding-based merge


@dataclass
class _CanonicalEntity:
    canonical_id: str
    canonical_name: str
    type: str
    aliases: set = field(default_factory=set)
    description: str = ""
    embedding: Optional[List[float]] = None


class EntityLinker:
    """Resolves raw extracted entity mentions to canonical entities."""

    def __init__(self, similarity_threshold: float = _SIMILARITY_THRESHOLD) -> None:
        self.threshold = similarity_threshold
        self._registry: Dict[str, _CanonicalEntity] = {}  # canonical_id -> entity
        self._name_index: Dict[str, str] = {}  # normalized name -> canonical_id
        self._embedder = None  # lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def link(self, result: ExtractionResult) -> ExtractionResult:
        """Rewrite an ExtractionResult so entity names are canonical."""
        # Map raw name -> canonical name for relationship rewriting.
        name_remap: Dict[str, str] = {}
        for ent in result.entities:
            canonical = self._resolve(ent.name, ent.type, ent.aliases, ent.description)
            name_remap[ent.name] = canonical.canonical_name
            for alias in ent.aliases:
                name_remap[alias] = canonical.canonical_name
            ent.name = canonical.canonical_name
            ent.aliases = sorted(canonical.aliases)
            ent.type = canonical.type

        # Rewrite relationship endpoints to canonical names.
        for rel in result.relationships:
            rel.source = name_remap.get(rel.source, rel.source)
            rel.target = name_remap.get(rel.target, rel.target)
        return result

    def canonical_id(self, name: str, entity_type: str) -> str:
        """Return the canonical id for an already-resolved entity name."""
        normalized = _normalize(name)
        for cid, ent in self._registry.items():
            if ent.canonical_name.lower() == name.lower() or _normalize(ent.canonical_name) == normalized:
                if ent.type.lower() == entity_type.lower():
                    return cid
        # Not yet seen — assign on the fly.
        return _make_canonical_id(name, entity_type)

    def all_entities(self) -> List[_CanonicalEntity]:
        return list(self._registry.values())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve(
        self,
        name: str,
        entity_type: str,
        aliases: List[str],
        description: str,
    ) -> _CanonicalEntity:
        normalized = _normalize(name)
        # 1) exact normalized-string hit (across aliases too)
        candidates = [name] + list(aliases)
        for cand in candidates:
            nc = _normalize(cand)
            cid = self._name_index.get(nc)
            if cid and self._registry[cid].type.lower() == entity_type.lower():
                ent = self._registry[cid]
                ent.aliases.update(a for a in aliases if _normalize(a) != nc)
                if description and not ent.description:
                    ent.description = description
                return ent

        # 2) embedding similarity (only among same-type entities)
        if len(self._registry) > 0:
            cand_emb = self._embed([name])
            best: Optional[tuple[float, _CanonicalEntity]] = None
            for ent in self._registry.values():
                if ent.type.lower() != entity_type.lower():
                    continue
                if ent.embedding is None:
                    continue
                sim = _cosine(cand_emb[0], ent.embedding)
                if best is None or sim > best[0]:
                    best = (sim, ent)
            if best and best[0] >= self.threshold:
                ent = best[1]
                ent.aliases.add(name)
                for a in aliases:
                    ent.aliases.add(a)
                self._name_index[normalized] = ent.canonical_id
                return ent

        # 3) new canonical entity
        canonical_name = _humanize(name)
        cid = _make_canonical_id(canonical_name, entity_type)
        ent = _CanonicalEntity(
            canonical_id=cid,
            canonical_name=canonical_name,
            type=entity_type,
            aliases=set(aliases) | {name} - {canonical_name},
            description=description,
        )
        ent.embedding = self._embed([canonical_name])[0]
        self._registry[cid] = ent
        self._name_index[normalized] = cid
        for a in aliases:
            self._name_index[_normalize(a)] = cid
        return ent

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self._embedder is None:
            try:
                from app.vector.embedder import get_embedder
                self._embedder = get_embedder()
            except Exception as exc:  # noqa: BLE001 — embeddings optional for linking
                logger.debug("Embedder unavailable, using string-only linking: %s", exc)
                self._embedder = _NullEmbedder()
        return self._embedder.encode(texts)


class _NullEmbedder:
    """Fallback when the real embedder can't load — returns orthogonal vectors."""
    def encode(self, texts):
        import itertools
        return [[float(i == j) for j in range(len(texts))] for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(name: str) -> str:
    name = name.strip().lower()
    name = _LEGAL_SUFFIXES.sub("", name)
    name = _NON_ALNUM.sub(" ", name).strip()
    return name


def _humanize(name: str) -> str:
    """Pick a clean canonical name (proper-cased, suffix-free)."""
    cleaned = _LEGAL_SUFFIXES.sub("", name).strip()
    # Tidy whitespace left by suffix removal.
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")
    return cleaned or name


def _make_canonical_id(name: str, entity_type: str) -> str:
    slug = _NON_ALNUM.sub("_", name.lower()).strip("_")
    return f"{entity_type.lower()}:{slug}"


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
