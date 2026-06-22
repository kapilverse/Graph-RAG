"""
Entity & relation extraction (spec §3 Stage 2).

Primary: End-to-End LLM extraction ("Approach B") — one LLM call per chunk returns
structured JSON of entities + relationships. Higher quality, simpler, the spec's
recommended default for an MVP.

The extractor returns ExtractionResult objects; entity linking/dedup happens in
linker.py, and graph writes happen in graph/neo4j_store.py.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.extraction.linker import EntityLinker
from app.ingestion.chunker import Chunk
from app.llm.client import LLMError, llm
from app.llm.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER
from app.llm.schemas import ExtractionResult

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Extract entities + relationships from chunks via structured LLM output."""

    def __init__(self, linker: Optional[EntityLinker] = None) -> None:
        self.linker = linker or EntityLinker()

    def extract(self, chunk: Chunk) -> ExtractionResult:
        """Extract entities + relationships from a single chunk."""
        if not llm.is_configured:
            raise LLMError(
                "No LLM provider configured. Set GEMINI_API_KEY or HF_TOKEN in .env."
            )

        prompt = EXTRACTION_USER.format(
            chunk_id=chunk.id,
            source=chunk.source_doc,
            text=chunk.text,
        )
        try:
            data = llm.extract_json(
                system=EXTRACTION_SYSTEM,
                user=prompt,
                temperature=0.0,
            )
            result = ExtractionResult.model_validate(data)
            # Normalize: strip, uppercase types, link entities to canonical ids.
            result = self.linker.link(result)
            logger.info(
                "Extracted from %s: %d entities, %d relationships",
                chunk.id, len(result.entities), len(result.relationships),
            )
            return result
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — surface as LLMError with context
            logger.warning("Extraction failed for %s: %s", chunk.id, exc)
            return ExtractionResult()

    def extract_batch(self, chunks: List[Chunk]) -> List[ExtractionResult]:
        """Extract from a list of chunks, tolerating per-chunk failures."""
        results: List[ExtractionResult] = []
        for chunk in chunks:
            try:
                results.append(self.extract(chunk))
            except LLMError as exc:
                logger.error("Extraction error on %s (skipping): %s", chunk.id, exc)
                results.append(ExtractionResult())
        return results
