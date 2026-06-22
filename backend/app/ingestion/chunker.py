"""
Semantic chunker.

Per spec §3 Stage 1: "Semantic chunking (not fixed 512 tokens)... Overlap: 10-15%
between chunks so entities at boundaries aren't lost."

Strategy: split into sentences, then greedily group sentences into chunks that stay
under the token budget, carrying an overlap of N tokens from the previous chunk.
Each chunk carries provenance metadata (source_doc, section_header, page_num, index).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import settings
from app.ingestion.parser import ParsedDocument, Section

logger = logging.getLogger(__name__)

# Lightweight sentence splitter — avoids pulling in nltk/spacy at chunk time.
# Handles common abbreviations conservatively.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    """A text chunk with full provenance — the core unit flowing through the pipeline."""

    id: str
    text: str
    source_doc: str
    section_header: str = ""
    page_num: Optional[int] = None
    chunk_index: int = 0
    embedding: Optional[List[float]] = None
    entity_ids: List[str] = field(default_factory=list)
    community_ids: List[str] = field(default_factory=list)

    @property
    def document_id(self) -> str:
        return self.source_doc


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving content."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _approx_token_count(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Cheap, no tokenizer dependency."""
    return max(1, len(text) // 4)


def chunk_document(
    doc: ParsedDocument,
    chunk_size_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
) -> List[Chunk]:
    """Chunk a parsed document into overlapping, sentence-aware pieces."""
    chunk_size = chunk_size_tokens or settings.chunk_size_tokens
    overlap = overlap_tokens or settings.chunk_overlap_tokens

    chunks: List[Chunk] = []
    chunk_idx = 0
    doc_stem = _doc_stem(doc.source_doc)

    for section in doc.sections:
        sentences = _split_sentences(section.full_text)
        if not sentences:
            continue

        current_sentences: List[str] = []
        current_tokens = 0
        overlap_carry: List[str] = []

        def flush() -> None:
            nonlocal current_sentences, current_tokens, chunk_idx
            if not current_sentences:
                return
            text = " ".join(current_sentences)
            chunk_id = f"{doc_stem}_c{chunk_idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source_doc=doc.source_doc,
                    section_header=section.header,
                    page_num=section.page_num,
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1
            # Carry overlap: the trailing sentences whose token count is ~ overlap.
            overlap_carry.clear()
            carry_tokens = 0
            for sent in reversed(current_sentences):
                st = _approx_token_count(sent)
                if carry_tokens + st > overlap:
                    break
                overlap_carry.insert(0, sent)
                carry_tokens += st
            current_sentences = list(overlap_carry)
            current_tokens = carry_tokens

        for sent in sentences:
            sent_tokens = _approx_token_count(sent)
            # If a single sentence exceeds the budget, emit it as its own chunk.
            if sent_tokens > chunk_size and not current_sentences:
                current_sentences = [sent]
                current_tokens = sent_tokens
                flush()
                continue

            if current_tokens + sent_tokens > chunk_size:
                flush()
            current_sentences.append(sent)
            current_tokens += sent_tokens

        flush()

    logger.info("Chunked %s into %d chunks (size=%d, overlap=%d tokens)",
                doc.source_doc, len(chunks), chunk_size, overlap)
    return chunks


def chunk_file(path: str) -> List[Chunk]:
    """Convenience: parse + chunk a single file."""
    from app.ingestion.parser import parse_file
    return chunk_document(parse_file(path))


def _doc_stem(source_doc: str) -> str:
    """Stable stem for chunk IDs, e.g. 'apple_history.pdf' -> 'apple_history'."""
    name = source_doc.rsplit(".", 1)[0] if "." in source_doc else source_doc
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower() or "doc"
