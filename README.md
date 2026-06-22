# Graph RAG

A portfolio-grade **Graph Retrieval-Augmented Generation** system that converts documents
into a knowledge graph of entities and relationships, then answers multi-hop questions by
traversing that graph — instead of doing flat vector search over text chunks.

> **The differentiator:** Vanilla RAG can't answer *"Which companies founded by ex-Apple
> employees compete with Apple?"* because no single chunk contains all three facts. Graph RAG
> traverses `Apple --employed--> Person --founded--> Company --competes_with--> Apple` and pulls
> the supporting chunks for each hop.

---

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Documents  │───▶│  Ingestion   │───▶│ Text Chunks  │──▶ Qdrant (vectors)
│ (PDF/HTML/TXT)│  │ Parse + Split│    │              │
└─────────────┘    └──────────────┘    └──────────────┘
                         │
                         ▼
                 ┌──────────────────┐
                 │ Entity + Relation│──▶ Neo4j (knowledge graph)
                 │   Extraction     │
                 └──────────────────┘
                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
  Local Search    Global Search     Hybrid Search
  (entity hops)  (communities)      (both, merged)
        │                │                 │
        └────────────────┼─────────────────┘
                         ▼
                ┌─────────────────┐
                │ Context Assembly│──▶ citations + provenance
                │   + LLM Answer  │
                └─────────────────┘
```

**Pipeline stages** (spec §3):
1. **Ingestion** — rich parsing (PDF/HTML/TXT) + semantic chunking with overlap.
2. **Entity & relation extraction** — end-to-end LLM (JSON output) with entity linking/dedup.
3. **Graph data model** — Neo4j nodes (`Document`, `Chunk`, `Entity`, `Community`) + edges
   (`MENTIONS`, `RELATED_TO`, `SIMILAR_TO`, `PART_OF`, `BELONGS_TO`).
4. **Vector store** — Qdrant for chunk + community-summary embeddings.
5. **Community detection** — Leiden algorithm + LLM-written community summaries.
6. **Retrieval engine** — local (entity-centric), global (community-centric), hybrid.
7. **Context assembly & generation** — structured prompt with enforced citations.

## Tech stack

| Layer | Technology |
|---|---|
| Parsing | pymupdf, beautifulsoup4 |
| Extraction | OpenAI-compatible LLM (Gemini default, HuggingFace fallback) |
| Embeddings | sentence-transformers `BAAI/bge-large-en-v1.5` |
| Re-ranker | `BAAI/bge-reranker-v2-m3` cross-encoder |
| Graph DB | Neo4j 5.x |
| Vector DB | Qdrant 1.x |
| Community detection | igraph + leidenalg |
| Backend | FastAPI |
| Frontend | React + Vite + TypeScript, d3-force |
| Orchestration | docker-compose |

## Quick start

### 1. Configure
```bash
cp .env.example .env
# Edit .env — set at least one of: GEMINI_API_KEY or HF_TOKEN
```

### 2. Start infrastructure (Neo4j + Qdrant)
```bash
docker compose up -d neo4j qdrant
```

### 3. Install Python deps + run the pipeline
```bash
pip install -r requirements.txt
python -m app.pipeline ingest data/sample     # ingest + extract + build graph
python -m app.pipeline build-communities      # Leiden + summaries
```

### 4. Run the backend + frontend
```bash
uvicorn app.main:app --reload                 # API at http://localhost:8000
cd frontend && npm install && npm run dev     # UI at http://localhost:5173
```

Or run the whole stack in Docker:
```bash
docker compose up --build
```

## Evaluation (spec §8)

```bash
python -m app.eval.generate_questions --n 50    # build test set
python -m app.eval.evaluate --report            # benchmark vs vanilla RAG
```

Metrics: Recall@K, entity recall, faithfulness, answer relevance, multi-hop accuracy.

## Project structure

```
Graph_RAG/
├── backend/app/
│   ├── config.py            # settings (pydantic-settings)
│   ├── main.py              # FastAPI app
│   ├── llm/                 # pluggable LLM client, prompts, schemas
│   ├── ingestion/           # parser + chunker
│   ├── extraction/          # entity/relation extractor + entity linking
│   ├── graph/               # Neo4j store + schema.cypher
│   ├── vector/              # embedder + Qdrant store
│   ├── community/           # Leiden detection + LLM summaries
│   ├── retrieval/           # local / global / hybrid + reranker
│   ├── pipeline.py          # GraphRAGPipeline orchestrator
│   ├── api/                 # FastAPI routes + schemas
│   └── eval/                # question generation + benchmarking
├── frontend/                # React + D3 graph explorer
├── data/sample/             # bundled demo docs
├── docker-compose.yml
└── requirements.txt
```

## LLM providers

The system is OpenAI-compatible and tries providers in order:
1. **Gemini 2.0 Flash** (default) — free tier, native JSON mode, reliable.
2. **Hugging Face Inference** (fallback) — open models, free.

Switch by editing `.env` — no code changes needed. Also works with OpenAI, Ollama, vLLM, Groq.
