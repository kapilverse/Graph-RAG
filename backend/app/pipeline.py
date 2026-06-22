"""
GraphRAGPipeline — the orchestrator (spec §8 skeleton, fleshed out).

Wires together: parse → chunk → embed → store vectors → extract entities/relations →
build graph. Exposes query() once retrieval is plugged in (Stage 8).

Runnable as a module:
    python -m app.pipeline ingest data/sample
    python -m app.pipeline stats
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from app.config import settings
from app.extraction.extractor import LLMExtractor
from app.extraction.linker import EntityLinker
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.parser import parse_file
from app.llm.client import LLMError, llm
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """End-to-end Graph RAG orchestrator. Holds long-lived clients."""

    def __init__(self, *, connect: bool = True) -> None:
        self.linker = EntityLinker()
        self.extractor = LLMExtractor(linker=self.linker)
        self._embedder = None
        self._qdrant = None
        self._neo4j = None
        self._retriever = None
        if connect:
            self._init_clients()

    # ------------------------------------------------------------------
    # Lazy clients (so unit tests can construct without infra)
    # ------------------------------------------------------------------
    def _init_clients(self) -> None:
        try:
            from app.vector.embedder import get_embedder
            from app.vector.qdrant_store import QdrantStore
            from app.graph.neo4j_store import Neo4jStore

            self._embedder = get_embedder()
            self._qdrant = QdrantStore()
            self._neo4j = Neo4jStore()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Infra clients not ready (services may be down): %s", exc)

    @property
    def qdrant(self):
        if self._qdrant is None:
            from app.vector.qdrant_store import QdrantStore
            self._qdrant = QdrantStore()
        return self._qdrant

    @property
    def neo4j(self):
        if self._neo4j is None:
            from app.graph.neo4j_store import Neo4jStore
            self._neo4j = Neo4jStore()
        return self._neo4j

    @property
    def embedder(self):
        if self._embedder is None:
            from app.vector.embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = HybridRetriever(
                embedder=self.embedder,
                qdrant=self.qdrant,
                neo4j=self.neo4j,
            )
        return self._retriever

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, path: str | Path, *, reset: bool = False) -> dict:
        """Ingest a single file end-to-end: parse → chunk → embed → extract → graph."""
        path = Path(path)
        logger.info("=== Ingesting %s ===", path.name)

        # 1. Parse + chunk
        doc = parse_file(path)
        chunks = chunk_document(doc)
        if not chunks:
            logger.warning("No chunks produced from %s", path.name)
            return {"chunks": 0, "entities": 0, "relationships": 0}

        # 2. Prepare stores
        if reset:
            self._reset_stores()
        self.qdrant.init_collections(vector_dim=self.embedder.dim)
        self.neo4j.ensure_schema()

        # 3. Embed chunks
        embeddings = self.embedder.encode([c.text for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # 4. Store chunks in vector DB + graph
        self.qdrant.upsert_chunks(chunks)
        self.neo4j.add_document(doc_id=path.name, source_doc=path.name, num_chunks=len(chunks))
        for chunk in chunks:
            self.neo4j.add_chunk(chunk)

        # 5. Extract entities + relationships per chunk, write to graph
        if not llm.is_configured:
            logger.warning(
                "LLM not configured — skipping extraction. Set GEMINI_API_KEY/HF_TOKEN."
            )
            return {"chunks": len(chunks), "entities": 0, "relationships": 0}

        total_entities = total_rels = 0
        for chunk in chunks:
            try:
                extraction = self.extractor.extract(chunk)
            except LLMError as exc:
                logger.error("Extraction failed for %s: %s", chunk.id, exc)
                continue
            # Track entity ids on the chunk for metadata filtering later.
            chunk.entity_ids = [
                f"{e.type.lower()}:{_slug(e.name)}" for e in extraction.entities
            ]
            self.neo4j.apply_extraction(chunk, extraction)
            total_entities += len(extraction.entities)
            total_rels += len(extraction.relationships)

        # Re-upsert chunks now that entity_ids are populated (updates payload).
        self.qdrant.upsert_chunks(chunks)

        logger.info(
            "Ingested %s: %d chunks, %d entities, %d relationships",
            path.name, len(chunks), total_entities, total_rels,
        )
        return {"chunks": len(chunks), "entities": total_entities, "relationships": total_rels}

    def ingest_dir(self, dir_path: str | Path, *, reset: bool = False) -> List[dict]:
        """Ingest all supported files in a directory."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Not a directory: {dir_path}")
        if reset:
            self._reset_stores()
        results = []
        files = sorted(
            p for p in dir_path.iterdir()
            if p.suffix.lower() in (".pdf", ".html", ".htm", ".txt", ".md")
        )
        logger.info("Found %d files to ingest in %s", len(files), dir_path)
        for f in files:
            try:
                results.append(self.ingest(f))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to ingest %s: %s", f.name, exc)
                results.append({"file": f.name, "error": str(exc)})
        return results

    # ------------------------------------------------------------------
    # Query (delegates to retrieval, Stage 8)
    # ------------------------------------------------------------------
    def query(self, question: str, strategy: str = "hybrid", top_k: Optional[int] = None):
        """Answer a question using the configured retrieval strategy."""
        return self.retriever.query(question, strategy=strategy, top_k=top_k)

    # ------------------------------------------------------------------
    # Stats / admin
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        out: dict = {}
        try:
            out["neo4j"] = self.neo4j.stats()
        except Exception as exc:  # noqa: BLE001
            out["neo4j"] = {"error": str(exc)}
        out["llm_provider"] = llm.primary_provider_name if llm.is_configured else "none"
        return out

    def _reset_stores(self) -> None:
        self.qdrant.reset_collections()
        self.neo4j.clear_all()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unknown"


def _main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if len(sys.argv) < 2:
        print("Usage: python -m app.pipeline {ingest|ingest-dir|stats} [path]")
        return 2

    cmd = sys.argv[1]
    pipeline = GraphRAGPipeline(connect=True)

    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: ingest <file> [--reset]")
            return 2
        reset = "--reset" in sys.argv
        result = pipeline.ingest(sys.argv[2], reset=reset)
        print(result)
    elif cmd == "ingest-dir":
        if len(sys.argv) < 3:
            print("Usage: ingest-dir <dir> [--reset]")
            return 2
        reset = "--reset" in sys.argv
        for r in pipeline.ingest_dir(sys.argv[2], reset=reset):
            print(r)
    elif cmd == "stats":
        print(pipeline.stats())
    else:
        print(f"Unknown command: {cmd}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
