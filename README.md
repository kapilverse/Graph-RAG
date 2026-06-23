# Graph RAG

A portfolio-grade **Graph Retrieval-Augmented Generation** system that converts documents
into a knowledge graph of entities and relationships, then answers multi-hop questions by
traversing that graph — instead of doing flat vector search over text chunks.

> **The differentiator:** Vanilla RAG can't answer *"Which companies founded by ex-Apple
> employees compete with Apple?"* because no single chunk contains all three facts. Graph RAG
> traverses `Apple --employed--> Person --founded--> Company --competes_with--> Apple` and pulls
> the supporting chunks for each hop.

📖 **New to this project?** Read [**EXPLAINED.md**](EXPLAINED.md) for a complete walkthrough
of every component, how the pipeline works, how to configure it, and how to extend it.

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

### Pipeline Stages

| Stage | What It Does | Key File |
|---|---|---|
| 1. Ingestion | Parse PDF/HTML/TXT → semantic chunks with overlap | `ingestion/parser.py`, `ingestion/chunker.py` |
| 2. Extraction | LLM extracts entities + relationships from each chunk | `extraction/extractor.py` |
| 3. Entity Linking | Deduplicate "Apple" / "Apple Inc." / "AAPL" → one node | `extraction/linker.py` |
| 4. Graph | Write entities/relationships to Neo4j knowledge graph | `graph/neo4j_store.py` |
| 5. Vector Store | Embed chunks, store in Qdrant for semantic search | `vector/embedder.py`, `vector/qdrant_store.py` |
| 6. Communities | Leiden algorithm clusters entities into topics + LLM summaries | `community/detection.py` |
| 7. Retrieval | Local (entity hops) / Global (communities) / Hybrid | `retrieval/local.py`, `retrieval/global_search.py`, `retrieval/hybrid.py` |
| 8. Generation | Structured prompt with citations → LLM generates answer | `retrieval/hybrid.py`, `llm/prompts.py` |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Parsing | `pymupdf`, `beautifulsoup4` | Preserves layout, handles tables |
| LLM Extraction | OpenAI-compatible SDK (Gemini 2.0 Flash default, HF fallback) | Free tier, reliable, provider-agnostic |
| Embeddings | `sentence-transformers` `BAAI/bge-large-en-v1.5` | Best open-source retrieval model, runs locally |
| Re-ranker | `BAAI/bge-reranker-v2-m3` cross-encoder | Accurate final ranking |
| Graph DB | **Neo4j 5.x** | Cypher is learnable, visualization built-in |
| Vector DB | **Qdrant 1.x** | Fast, metadata filtering, free tier |
| Community Detection | `igraph` + `leidenalg` | Fast, proven clustering |
| Backend | **FastAPI** | Async, auto-docs, industry standard |
| Frontend | **React + Vite + TypeScript**, `d3-force` | Interactive graph visualization |
| Orchestration | **docker-compose** | One command to start everything |

---

## Quick Start

### Prerequisites
- **Python 3.10+**
- **Docker** (for Neo4j + Qdrant)
- **Node.js 18+** (for the frontend)
- **At least one LLM API key** (Gemini free tier recommended)

### 1. Configure

```bash
cp .env.example .env
# Edit .env — set at least one of:
#   GEMINI_API_KEY=your_key_here    (free at https://aistudio.google.com)
#   HF_TOKEN=your_token_here       (free at https://huggingface.co/settings/tokens)
```

### 2. Start infrastructure (Neo4j + Qdrant)

```bash
docker compose up -d neo4j qdrant
```

Verify: Neo4j browser at http://localhost:7474, Qdrant dashboard at http://localhost:6333/dashboard.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> ⚡ First run downloads ML models (~1.3 GB). This is a one-time cost.

### 4. Ingest documents + build the knowledge graph

```bash
# Ingest the bundled sample documents
cd backend
python -m app.pipeline ingest-dir ../data/sample --reset

# Build communities (topic clusters with LLM summaries)
python -m app.community.detection
```

### 5. Start the backend API

```bash
cd backend
uvicorn app.main:app --reload
```

API at http://localhost:8000 — Swagger docs at http://localhost:8000/docs

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

UI at http://localhost:5173

### All-in-one (Docker)

```bash
docker compose up --build
```

Opens: API :8000, Frontend :5173, Neo4j :7474, Qdrant :6333.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/ingest` | Ingest a file or directory into the pipeline |
| `POST` | `/api/query` | Ask a question (supports `strategy: local\|global\|hybrid`) |
| `POST` | `/api/graph/explore` | Fetch a subgraph around an entity (for D3 visualization) |
| `GET` | `/api/communities` | List detected communities |
| `GET` | `/api/documents` | List ingested documents |
| `GET` | `/api/stats` | Graph + vector store statistics |
| `GET` | `/api/health` | Service health check (Neo4j, Qdrant, LLM status) |

---

## Evaluation

```bash
cd backend

# Generate 50 multi-hop test questions from your documents
python -m app.eval.generate_questions --n 50 --output app/eval/testset/questions.jsonl

# Benchmark Graph RAG vs Vanilla RAG
python -m app.eval.evaluate --testset app/eval/testset/questions.jsonl --report app/eval/reports/
```

Outputs a markdown report with: retrieval recall, vanilla RAG baseline, faithfulness,
answer relevance, and multi-hop breakdown.

---

## Project Structure

```
Graph_RAG/
├── EXPLAINED.md                 ← Full walkthrough of every component
├── README.md                    ← ← YOU ARE HERE
├── docker-compose.yml           ← Neo4j + Qdrant + Backend + Frontend
├── requirements.txt             ← Python dependencies
├── .env.example                 ← Configuration template
│
├── data/
│   ├── sample/                  ← Bundled demo docs (Apple, Samsung, Google)
│   └── uploads/                 ← Put your own documents here
│
├── backend/
│   ├── Dockerfile
│   ├── app/
│   │   ├── config.py            ← Central settings (reads .env)
│   │   ├── main.py              ← FastAPI app
│   │   ├── pipeline.py          ← End-to-end orchestrator
│   │   ├── llm/                 ← LLM client, prompts, schemas
│   │   ├── ingestion/           ← Document parser + chunker
│   │   ├── extraction/          ← Entity/relation extractor + entity linker
│   │   ├── graph/               ← Neo4j store + schema
│   │   ├── vector/              ← Embedder + Qdrant store
│   │   ├── community/           ← Leiden community detection
│   │   ├── retrieval/           ← Local / global / hybrid search + reranker
│   │   ├── api/                 ← FastAPI routes + schemas
│   │   └── eval/                ← Test set generation + benchmarking
│   └── tests/
│
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── vite.config.ts
    └── src/
        ├── App.tsx              ← Main layout
        ├── api.ts               ← HTTP client
        └── components/
            ├── Chat.tsx          ← Question/answer interface
            ├── GraphExplorer.tsx ← D3 force-directed graph
            └── CitationPanel.tsx ← Source chunks + provenance
```

---

## LLM Providers

The system is **OpenAI-compatible** and tries providers in order with transparent
fallback. Switch by editing `.env` — **no code changes needed.**

| Priority | Provider | Cost | How to enable |
|---|---|---|---|
| 1 (default) | **Gemini 2.0 Flash** | Free tier (1,500 req/day) | `GEMINI_API_KEY=...` in `.env` |
| 2 (fallback) | **Hugging Face Inference** | Free | `HF_TOKEN=...` in `.env` |
| Also works | OpenAI, Ollama, vLLM, Groq, Together, LM Studio | Varies | Set `BASE_URL` + `API_KEY` + `MODEL` |

---

## Sample Questions to Try

After ingesting `data/sample/`, try these multi-hop questions:

1. *"Which companies founded by ex-Apple employees compete with Apple?"*
   → Requires: Apple → Tony Fadell → Nest Labs → Google → competes with Apple

2. *"Who supplies Apple's chips and what do they compete with?"*
   → Requires: Apple → TSMC (partner) + Samsung (competitor in foundry)

3. *"What is the connection between Microsoft, OpenAI, and Apple?"*
   → Requires: Microsoft → invested in OpenAI → competes with Apple in AI

4. *"How did Apple's privacy changes affect Meta?"*
   → Requires: Apple → App Tracking Transparency → impacted Meta ad revenue

---

## License

This is a portfolio/educational project.
