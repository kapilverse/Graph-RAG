"""
Community detection (spec §3 Stage 5 — "Microsoft GraphRAG's Secret Sauce").

Run the Leiden algorithm over the entity RELATED_TO graph to find clusters of tightly
connected entities (topics). Each community gets an LLM-written summary and is stored:
- in Neo4j as :Community nodes with :BELONGS_TO edges
- in Qdrant's community_summaries collection for global search

Runnable:
    python -m app.community.detection
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import igraph as ig
from leidenalg import RBConfigurationVertexPartition, find_partition

from app.config import settings
from app.llm.client import LLMError, llm
from app.llm.prompts import COMMUNITY_SUMMARY_SYSTEM, COMMUNITY_SUMMARY_USER

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detects communities via Leiden and writes summaries to Neo4j + Qdrant."""

    def __init__(self, neo4j_store, qdrant_store, embedder) -> None:
        self.neo4j = neo4j_store
        self.qdrant = qdrant_store
        self.embedder = embedder

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def build_communities(self, resolution: float = 1.0, min_size: int = 2) -> Dict[str, int]:
        """Run Leiden, assign entities to communities, summarize, and store.

        Returns counts: {communities, entities_assigned, summaries_written}.
        """
        logger.info("Loading entity graph from Neo4j...")
        nodes, edges = self._load_graph()
        if len(nodes) < min_size:
            logger.warning("Too few entities (%d) for community detection", len(nodes))
            return {"communities": 0, "entities_assigned": 0, "summaries_written": 0}

        graph = self._build_igraph(nodes, edges)
        logger.info("Built graph: %d entities, %d edges", graph.vcount(), graph.ecount())

        # Leiden.
        partition = find_partition(
            graph,
            RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=42,
        )
        logger.info("Leiden found %d communities", len(set(partition.membership)))

        # Group entity ids by community.
        communities: Dict[int, List[str]] = {}
        for idx, comm in enumerate(partition.membership):
            communities.setdefault(comm, []).append(graph.vs[idx]["name"])

        assigned = written = 0
        for comm_id, member_entity_ids in communities.items():
            if len(member_entity_ids) < min_size:
                continue
            # Resolve canonical names + relationships for the summary.
            entities_and_rels = self._describe_community(member_entity_ids)
            summary = self._summarize(comm_id, entities_and_rels)

            # Store summary embedding in Qdrant for global search.
            emb = self.embedder.encode([summary])[0]
            entity_names = [e["name"] for e in entities_and_rels["entities"]]
            cid = f"community_{comm_id}"
            self.qdrant.upsert_community(cid, summary, emb, entity_names)

            # Assign BELONGS_TO in Neo4j.
            for eid in member_entity_ids:
                self.neo4j.assign_community(eid, cid, summary)
            assigned += len(member_entity_ids)
            written += 1

        logger.info("Communities built: %d (covering %d entities)", written, assigned)
        return {"communities": written, "entities_assigned": assigned, "summaries_written": written}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_graph(self):
        """Pull entities + RELATED_TO edges from Neo4j into name-keyed structures."""
        with self.neo4j._driver.session(database=settings.neo4j_database) as session:
            ents = session.run("MATCH (e:Entity) RETURN e.id AS id, e.name AS name")
            nodes = [(r["id"], r["name"]) for r in ents]
            rels = session.run(
                """
                MATCH (a:Entity)-[:RELATED_TO]->(b:Entity)
                RETURN a.id AS src, b.id AS tgt
                """
            )
            edges = [(r["src"], r["tgt"]) for r in rels]
        return nodes, edges

    def _build_igraph(self, nodes, edges):
        ids = [n[0] for n in nodes]
        name_by_id = {n[0]: n[1] for n in nodes}
        g = ig.Graph(directed=False)
        g.add_vertices(len(ids))
        g.vs["name"] = ids  # store id as the canonical key
        g.vs["label"] = [name_by_id[i] for i in ids]
        # Map edges to indices, dropping unknown endpoints.
        idx = {eid: i for i, eid in enumerate(ids)}
        edge_idx = [(idx[s], idx[t]) for s, t in edges if s in idx and t in idx and s != t]
        if edge_idx:
            g.add_edges(edge_idx)
        g.simplify(combine_edges="sum")  # collapse multi-edges
        return g

    def _describe_community(self, entity_ids: List[str]) -> dict:
        with self.neo4j._driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                """
                UNWIND $eids AS eid
                MATCH (e:Entity {id: eid})
                OPTIONAL MATCH (e)-[r:RELATED_TO]-(other:Entity)
                WHERE other.id IN $eids
                RETURN collect(DISTINCT {
                    name: e.name, type: e.type, description: e.description
                }) AS entities,
                       collect(DISTINCT {
                    source: e.name, type: coalesce(r.relation_type, r.type), target: other.name
                }) AS relationships
                """,
                eids=entity_ids,
            )
            rec = result.single()
            return {
                "entities": rec["entities"] if rec else [],
                "relationships": rec["relationships"] if rec else [],
            }

    def _summarize(self, comm_id: int, entities_and_rels: dict) -> str:
        if not llm.is_configured:
            # Fallback summary if no LLM available.
            names = ", ".join(e["name"] for e in entities_and_rels["entities"][:10])
            return f"Community {comm_id} covers: {names}."
        entities_text = "\n".join(
            f"- {e['name']} ({e.get('type', 'Entity')}): {e.get('description', '')}"
            for e in entities_and_rels["entities"]
        )
        rels_text = "\n".join(
            f"- {r['source']} --{r.get('type', 'RELATED_TO')}--> {r['target']}"
            for r in entities_and_rels["relationships"]
            if r.get("source") and r.get("target")
        )
        try:
            return llm.chat(
                system=COMMUNITY_SUMMARY_SYSTEM,
                user=COMMUNITY_SUMMARY_USER.format(
                    community_id=comm_id,
                    entities_and_relations=f"Entities:\n{entities_text}\n\nRelationships:\n{rels_text}",
                ),
            ).strip()
        except LLMError as exc:
            logger.warning("Community summary failed: %s", exc)
            names = ", ".join(e["name"] for e in entities_and_rels["entities"][:10])
            return f"Community {comm_id} covers: {names}."


# ---------------------------------------------------------------------------
def _main() -> int:
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    from app.graph.neo4j_store import Neo4jStore
    from app.vector.embedder import get_embedder
    from app.vector.qdrant_store import QdrantStore

    detector = CommunityDetector(Neo4jStore(), QdrantStore(), get_embedder())
    print(detector.build_communities())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
