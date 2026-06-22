"""
Neo4j graph store (spec §3 Stage 3 + Stage 6).

Encapsulates all Cypher. Uses MERGE for idempotent upserts so re-ingestion is safe.
Holds the schema in schema.cypher and applies it on connect.

Node labels:  Document, Chunk, Entity (Person|Org|Tech|Concept|Location|...), Community
Edges:       MENTIONS, RELATED_TO, SIMILAR_TO, PART_OF, BELONGS_TO
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.ingestion.chunker import Chunk
from app.llm.schemas import ExtractionResult

logger = logging.getLogger(__name__)

_SCHEMA_FILE = Path(__file__).parent / "schema.cypher"


class Neo4jStore:
    """Thin wrapper around the neo4j driver with graph-RAG-specific helpers."""

    def __init__(self) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        """Apply constraints/indexes from schema.cypher."""
        statements = _split_cypher(_SCHEMA_FILE.read_text(encoding="utf-8"))
        with self._driver.session(database=settings.neo4j_database) as session:
            for stmt in statements:
                session.run(stmt)
        logger.info("Neo4j schema ensured (%d statements)", len(statements))

    def clear_all(self) -> None:
        """Wipe all Graph RAG data (used by re-ingest)."""
        with self._driver.session(database=settings.neo4j_database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all nodes/relationships from Neo4j")

    def ping(self) -> bool:
        try:
            with self._driver.session(database=settings.neo4j_database) as session:
                session.run("RETURN 1")
            return True
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Writes (ingestion)
    # ------------------------------------------------------------------
    def add_document(self, doc_id: str, source_doc: str, num_chunks: int) -> None:
        with self._driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.source_doc = $source_doc, d.num_chunks = $num_chunks
                """,
                doc_id=doc_id, source_doc=source_doc, num_chunks=num_chunks,
            )

    def add_chunk(self, chunk: Chunk) -> None:
        with self._driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.source_doc = $source_doc,
                    c.section_header = $section_header,
                    c.page_num = $page_num,
                    c.chunk_index = $chunk_index
                MERGE (c)-[:PART_OF]->(d)
                """,
                doc_id=chunk.source_doc,
                chunk_id=chunk.id,
                text=chunk.text,
                source_doc=chunk.source_doc,
                section_header=chunk.section_header,
                page_num=chunk.page_num,
                chunk_index=chunk.chunk_index,
            )

    def apply_extraction(self, chunk: Chunk, extraction: ExtractionResult) -> None:
        """Write entities + relationships extracted from a chunk, wired to the chunk."""
        with self._driver.session(database=settings.neo4j_database) as session:
            # 1. Entities + MENTIONS edges.
            for ent in extraction.entities:
                ent_id = f"{ent.type.lower()}:{_slug(ent.name)}"
                session.run(
                    """
                    MERGE (e:Entity {id: $eid})
                    SET e.name = $name,
                        e.type = $type,
                        e.aliases = $aliases,
                        e.description = coalesce(e.description, $desc)
                    WITH e
                    MATCH (c:Chunk {id: $cid})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    eid=ent_id,
                    name=ent.name,
                    type=ent.type,
                    aliases=ent.aliases,
                    desc=ent.description,
                    cid=chunk.id,
                )

            # 2. RELATED_TO edges (dedup evidence_chunk_ids on merge).
            for rel in extraction.relationships:
                src_id = self._find_entity_id(session, rel.source)
                tgt_id = self._find_entity_id(session, rel.target)
                if not src_id or not tgt_id:
                    logger.debug("Skipping rel %s (endpoint not found)", rel.type)
                    continue
                session.run(
                    """
                    MATCH (src:Entity {id: $src}), (tgt:Entity {id: $tgt})
                    MERGE (src)-[r:RELATED_TO {type: $rtype, target: $tgt}]->(tgt)
                    SET r.relation_type = $rtype,
                        r.confidence = CASE WHEN r.confidence IS NULL THEN $conf
                                            ELSE (r.confidence + $conf) / 2.0 END,
                        r.evidence_chunk_ids =
                            CASE WHEN $cid IN coalesce(r.evidence_chunk_ids, [])
                                 THEN r.evidence_chunk_ids
                                 ELSE coalesce(r.evidence_chunk_ids, []) + $cid END
                    """,
                    src=src_id, tgt=tgt_id, rtype=rel.type,
                    conf=rel.confidence, cid=chunk.id,
                )

    def assign_community(self, entity_id: str, community_id: str, community_summary: str = "") -> None:
        with self._driver.session(database=settings.neo4j_database) as session:
            session.run(
                """
                MERGE (cm:Community {id: $cid})
                SET cm.summary = coalesce(cm.summary, $summary)
                WITH cm
                MATCH (e:Entity {id: $eid})
                MERGE (e)-[:BELONGS_TO]->(cm)
                """,
                cid=community_id, summary=community_summary, eid=entity_id,
            )

    # ------------------------------------------------------------------
    # Reads (retrieval)
    # ------------------------------------------------------------------
    def get_neighbors(
        self, entity_name: str, depth: int = 2, limit: int = 20
    ) -> Dict[str, Any]:
        """Local-search traversal: entity -> RELATED_TO neighbors (N hops) -> evidence chunks.

        Mirrors the spec §3 Stage 6 Strategy A Cypher.
        """
        with self._driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})
                CALL {
                    WITH e
                    MATCH path = (e)-[:RELATED_TO*1..$depth]-(neighbor:Entity)
                    UNWIND nodes(path) as n
                    RETURN DISTINCT n
                    LIMIT $limit
                }
                OPTIONAL MATCH (neighbor)-[r:RELATED_TO]-(other:Entity)
                WITH collect(DISTINCT neighbor) + e AS allEnts,
                     collect(DISTINCT [neighbor, r, other]) AS rels
                UNWIND allEnts as ent
                OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(ent)
                RETURN collect(DISTINCT ent) AS entities,
                       rels AS relationships,
                       collect(DISTINCT chunk.id) AS chunk_ids
                """,
                name=entity_name,
                depth=settings.local_traversal_depth,
                limit=settings.local_neighbor_limit,
            )
            record = result.single()
            if not record:
                return {"entities": [], "relationships": [], "chunk_ids": []}
            return {
                "entities": [_node_to_dict(n) for n in record["entities"] if n is not None],
                "relationships": [_rel_to_dict(t) for t in record["relationships"] if t is not None],
                "chunk_ids": [c for c in record["chunk_ids"] if c],
            }

    def explore_neighborhood(self, entity_name: str, limit: int = 20) -> Dict[str, Any]:
        """Return a subgraph (nodes + links) around an entity for the D3 frontend."""
        with self._driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                MATCH (center:Entity {name: $name})
                OPTIONAL MATCH (center)-[r:RELATED_TO]-(neighbor:Entity)
                WITH center, collect(DISTINCT neighbor)[0..$limit] AS neighbors,
                     collect(DISTINCT r)[0..50] AS rels
                RETURN center, neighbors, rels
                """,
                name=entity_name, limit=limit,
            )
            record = result.single()
            if not record:
                return {"nodes": [], "links": []}
            nodes = [_node_to_dict(record["center"])] if record["center"] else []
            for n in (record["neighbors"] or []):
                if n is not None:
                    nodes.append(_node_to_dict(n))
            links = []
            for r in (record["rels"] or []):
                if r is None:
                    continue
                links.append({
                    "source": r.start_node["name"],
                    "target": r.end_node["name"],
                    "type": r["relation_type"] or r.type,
                })
            return {"nodes": nodes, "links": links}

    def get_entity_communities(self, community_ids: List[str]) -> List[Dict[str, Any]]:
        with self._driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                UNWIND $cids AS cid
                MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {id: cid})
                RETURN c.id AS community_id,
                       c.summary AS summary,
                       collect({name: e.name, type: e.type, description: e.description}) AS entities
                """,
                cids=community_ids,
            )
            return [
                {
                    "community_id": r["community_id"],
                    "summary": r["summary"] or "",
                    "entities": [e for e in r["entities"]],
                }
                for r in result
            ]

    def stats(self) -> Dict[str, int]:
        with self._driver.session(database=settings.neo4j_database) as session:
            counts = {}
            for label in ("Document", "Chunk", "Entity", "Community"):
                rec = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
                counts[label.lower()] = rec["c"] if rec else 0
            rec = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS c").single()
            counts["relationships"] = rec["c"] if rec else 0
            return counts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _find_entity_id(self, session, name: str) -> Optional[str]:
        rec = session.run(
            "MATCH (e:Entity {name: $name}) RETURN e.id AS id LIMIT 1", name=name
        ).single()
        return rec["id"] if rec else None


# ---------------------------------------------------------------------------
# Cypher / node helpers
# ---------------------------------------------------------------------------
def _split_cypher(text: str) -> List[str]:
    """Split a .cypher file into statements on ';', stripping comments."""
    statements = []
    for chunk in text.split(";"):
        lines = [ln for ln in chunk.splitlines() if not ln.strip().startswith("//")]
        stmt = "\n".join(lines).strip()
        if stmt:
            statements.append(stmt)
    return statements


def _node_to_dict(node) -> Dict[str, Any]:
    return dict(node)


def _rel_to_dict(triple) -> Dict[str, Any]:
    src, rel, tgt = triple
    return {
        "source": src["name"],
        "target": tgt["name"],
        "type": rel["relation_type"] or rel.type,
        "confidence": rel.get("confidence"),
        "evidence_chunk_ids": rel.get("evidence_chunk_ids", []),
    }


def _slug(name: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "unknown"
