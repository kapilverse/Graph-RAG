"""
Evaluation harness — benchmarks Graph RAG vs vanilla (vector-only) RAG (spec §8).

Metrics:
- Multi-hop accuracy: does the answer match the ground truth?
- Faithfulness: is the answer supported by the retrieved context?
- Answer relevance: does the answer address the question?
- Retrieval recall@K: did we retrieve the right chunks?

Runnable:
    python -m app.eval.evaluate --testset app/eval/testset/questions.jsonl --report reports/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.llm.client import LLMError, llm
from app.vector.embedder import get_embedder
from app.vector.qdrant_store import QdrantStore
from app.graph.neo4j_store import Neo4jStore
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

DEFAULT_TESTSET = str(Path(__file__).parent / "testset" / "questions.jsonl")
DEFAULT_REPORT = str(Path(__file__).parent / "reports" / "eval_report.md")


def load_testset(path: str) -> List[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def evaluate_single(
    retriever: HybridRetriever,
    qdrant: QdrantStore,
    embedder,
    question: dict,
    strategy: str = "hybrid",
    top_k: int = 10,
) -> Dict[str, Any]:
    """Evaluate one question against Graph RAG."""
    q_text = question["question"]
    ground_truth = question.get("answer", "")
    source_ids = set(question.get("source_chunk_ids", []))

    # --- Graph RAG answer ---
    start = time.time()
    try:
        result = retriever.query(q_text, strategy=strategy, top_k=top_k)
        grag_answer = result.answer
        grag_chunks = {c["chunk_id"] for c in result.chunks}
        grag_latency = time.time() - start
    except Exception as exc:  # noqa: BLE001
        grag_answer = f"ERROR: {exc}"
        grag_chunks = set()
        grag_latency = time.time() - start

    # --- Vanilla RAG answer (same chunks, no graph) ---
    try:
        query_vec = embedder.encode([q_text])[0]
        vanilla_hits = qdrant.search_chunks(query_vec, top_k=top_k)
        vanilla_chunk_ids = {h["chunk_id"] for h in vanilla_hits}
    except Exception:  # noqa: BLE001
        vanilla_chunk_ids = set()

    # --- Metrics ---
    retrieval_recall = len(source_ids & grag_chunks) / max(len(source_ids), 1)
    vanilla_recall = len(source_ids & vanilla_chunk_ids) / max(len(source_ids), 1)

    faithfulness = _judge_faithfulness(grag_answer, result.chunks if 'result' in dir() else []) if llm.is_configured else None
    relevance = _judge_relevance(q_text, grag_answer) if llm.is_configured else None

    return {
        "question": q_text,
        "ground_truth": ground_truth,
        "hops": question.get("hops", 1),
        "graph_rag": {
            "answer": grag_answer,
            "chunks_retrieved": grag_chunks,
            "latency_s": round(grag_latency, 3),
        },
        "vanilla_rag": {
            "chunks_retrieved": vanilla_chunk_ids,
        },
        "metrics": {
            "retrieval_recall@k": round(retrieval_recall, 3),
            "vanilla_recall@k": round(vanilla_recall, 3),
            "faithfulness": faithfulness,
            "relevance": relevance,
        },
    }


def run_evaluation(
    testset_path: str = DEFAULT_TESTSET,
    report_path: str = DEFAULT_REPORT,
    strategy: str = "hybrid",
) -> Dict[str, Any]:
    """Run the full evaluation and write a markdown report."""
    questions = load_testset(testset_path)
    if not questions:
        raise FileNotFoundError(f"No questions in {testset_path}")

    embedder = get_embedder()
    qdrant = QdrantStore()
    neo4j = Neo4jStore()
    retriever = HybridRetriever(embedder, qdrant, neo4j)

    results = []
    for i, q in enumerate(questions):
        logger.info("Evaluating %d/%d: %s", i + 1, len(questions), q["question"][:50])
        results.append(evaluate_single(retriever, qdrant, embedder, q, strategy=strategy))

    # Aggregate.
    avg = lambda key: sum(r["metrics"][key] for r in results if r["metrics"][key] is not None) / max(len([r for r in results if r["metrics"][key] is not None]), 1)
    avg_latency = sum(r["graph_rag"]["latency_s"] for r in results) / len(results)

    summary = {
        "total_questions": len(questions),
        "avg_retrieval_recall": round(avg("retrieval_recall@k"), 3),
        "avg_vanilla_recall": round(avg("vanilla_recall@k"), 3),
        "avg_faithfulness": avg("faithfulness") if any(r["metrics"]["faithfulness"] is not None for r in results) else "N/A",
        "avg_relevance": avg("relevance") if any(r["metrics"]["relevance"] is not None for r in results) else "N/A",
        "avg_latency_s": round(avg_latency, 3),
        "multi_hop_breakdown": _multi_hop_breakdown(results),
    }

    # Write markdown report.
    _write_report(report_path, results, summary)
    logger.info("Report written to %s", report_path)
    return summary


# ---------------------------------------------------------------------------
# LLM-as-judge helpers
# ---------------------------------------------------------------------------
def _judge_faithfulness(answer: str, chunks: list) -> Optional[float]:
    """Ask the LLM if the answer is supported by the chunks (1-5 scale → 0-1)."""
    if not chunks:
        return None
    evidence = "\n".join(f"[{c.get('chunk_id')}] {c.get('text', '')[:300]}" for c in chunks[:5])
    prompt = f"""On a scale of 1-5, is this answer faithful to (supported by) the evidence?

Answer: {answer[:500]}

Evidence:
{evidence}

Return ONLY a JSON number 1-5."""
    try:
        import ast
        resp = llm.chat(system="You are a faithful judge. Return only a number.", user=prompt)
        score = ast.literal_eval(resp.strip())
        return max(0.0, min(1.0, (score - 1) / 4))
    except Exception:  # noqa: BLE001
        return None


def _judge_relevance(question: str, answer: str) -> Optional[float]:
    """Ask the LLM if the answer addresses the question (1-5 scale → 0-1)."""
    prompt = f"""On a scale of 1-5, how relevant is this answer to the question?

Question: {question}
Answer: {answer[:500]}

Return ONLY a JSON number 1-5."""
    try:
        import ast
        resp = llm.chat(system="You are a relevance judge. Return only a number.", user=prompt)
        score = ast.literal_eval(resp.strip())
        return max(0.0, min(1.0, (score - 1) / 4))
    except Exception:  # noqa: BLE001
        return None


def _multi_hop_breakdown(results: List[dict]) -> dict:
    by_hop: dict[int, list] = {}
    for r in results:
        h = r.get("hops", 1)
        by_hop.setdefault(h, []).append(r)
    breakdown = {}
    for h in sorted(by_hop):
        group = by_hop[h]
        gr = sum(r["metrics"]["retrieval_recall@k"] for r in group) / len(group)
        vr = sum(r["metrics"]["vanilla_recall@k"] for r in group) / len(group)
        breakdown[str(h)] = {"count": len(group), "graph_rag_recall": round(gr, 3), "vanilla_recall": round(vr, 3)}
    return breakdown


def _write_report(path: str, results: List[dict], summary: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Graph RAG Evaluation Report\n\n")
        f.write(f"**Total questions:** {summary['total_questions']}\n\n")
        f.write("## Aggregate Metrics\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Avg Retrieval Recall@K (Graph RAG) | {summary['avg_retrieval_recall']} |\n")
        f.write(f"| Avg Retrieval Recall@K (Vanilla RAG) | {summary['avg_vanilla_recall']} |\n")
        f.write(f"| Avg Faithfulness | {summary['avg_faithfulness']} |\n")
        f.write(f"| Avg Relevance | {summary['avg_relevance']} |\n")
        f.write(f"| Avg Latency (s) | {summary['avg_latency_s']} |\n\n")

        f.write("## Multi-Hop Breakdown\n\n")
        f.write("| Hops | Count | Graph RAG Recall | Vanilla Recall |\n|---|---|---|---|\n")
        for h, d in summary.get("multi_hop_breakdown", {}).items():
            f.write(f"| {h} | {d['count']} | {d['graph_rag_recall']} | {d['vanilla_recall']} |\n")

        f.write("\n## Per-Question Results\n\n")
        for i, r in enumerate(results):
            f.write(f"### Q{i+1}: {r['question']}\n\n")
            f.write(f"- **Hops:** {r['hops']}\n")
            f.write(f"- **Graph RAG recall:** {r['metrics']['retrieval_recall@k']}\n")
            f.write(f"- **Vanilla recall:** {r['metrics']['vanilla_recall@k']}\n")
            f.write(f"- **Latency:** {r['graph_rag']['latency_s']}s\n\n")
            f.write(f"**Answer:** {r['graph_rag']['answer'][:200]}...\n\n")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Graph RAG vs Vanilla RAG")
    parser.add_argument("--testset", type=str, default=DEFAULT_TESTSET)
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT)
    parser.add_argument("--strategy", type=str, default="hybrid", choices=["local", "global", "hybrid"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    summary = run_evaluation(testset_path=args.testset, report_path=args.report, strategy=args.strategy)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
