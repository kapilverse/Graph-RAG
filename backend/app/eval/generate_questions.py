"""
Test-set question generator (spec §8).

Uses the LLM to generate multi-hop questions from ingested documents, with known
answers grounded in the source chunks. Outputs a JSONL test set for benchmarking.

Runnable:
    python -m app.eval.generate_questions --n 50 --output testset/questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from random import sample
from typing import List

from app.config import settings
from app.llm.client import LLMError, llm
from app.llm.prompts import QUESTION_GEN_SYSTEM, QUESTION_GEN_USER
from app.vector.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = str(Path(__file__).parent / "testset" / "questions.jsonl")


def generate_test_set(
    n: int = 50,
    chunks_per_call: int = 3,
    output_path: str = DEFAULT_OUTPUT,
) -> List[dict]:
    """Generate n questions by sampling chunk groups and asking the LLM."""
    qdrant = QdrantStore()

    # Fetch a sample of stored chunks.
    collection = settings.qdrant_collection_chunks
    if not qdrant.client.collection_exists(collection):
        raise RuntimeError(f"Collection '{collection}' does not exist — ingest documents first.")

    scroll, _ = qdrant.client.scroll(
        collection_name=collection,
        limit=200,  # sample from up to 200 chunks
        with_payload=True,
        with_vectors=False,
    )
    if not scroll:
        raise RuntimeError("No chunks found — ingest documents first.")

    # Build a test set by sampling random groups of chunks.
    questions: List[dict] = []
    attempts = 0
    max_attempts = n * 3  # allow retries for bad generations

    while len(questions) < n and attempts < max_attempts:
        attempts += 1
        group = sample(scroll, min(chunks_per_call, len(scroll)))
        chunks_text = "\n---\n".join(
            f"[{p.payload.get('chunk_id', '?')}] {p.payload.get('text', '')[:600]}"
            for p in group
        )
        chunk_ids = [p.payload.get("chunk_id", "?") for p in group]

        try:
            data = llm.extract_json(
                system=QUESTION_GEN_SYSTEM,
                user=QUESTION_GEN_USER.format(n=1, chunks_text=chunks_text),
            )
            items = data if isinstance(data, list) else [data]
            for item in items[:1]:
                q = {
                    "question": str(item.get("question", "")).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "hops": int(item.get("hops", 1)),
                    "source_chunk_ids": item.get("source_chunk_ids", chunk_ids),
                }
                if q["question"] and q["answer"]:
                    questions.append(q)
                    logger.info("Q%d/%d: %s", len(questions), n, q["question"][:60])
        except LLMError as exc:
            logger.warning("Generation attempt %d failed: %s", attempts, exc)
            continue

    # Write JSONL.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    logger.info("Wrote %d questions to %s", len(questions), output_path)
    return questions


def _main() -> int:
    parser = argparse.ArgumentParser(description="Generate multi-hop test questions")
    parser.add_argument("--n", type=int, default=50, help="Number of questions")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output JSONL path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    if not llm.is_configured:
        print("ERROR: LLM not configured. Set GEMINI_API_KEY or HF_TOKEN.", file=sys.stderr)
        return 1
    generate_test_set(n=args.n, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
