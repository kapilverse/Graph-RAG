"""
Graph RAG — FastAPI application entrypoint.

Run:  uvicorn app.main:app --reload
API:  http://localhost:8000/docs   (Swagger UI)
"""
from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Graph RAG",
    description="Portfolio-grade Graph Retrieval-Augmented Generation system. "
                "Converts documents into a knowledge graph of entities and relationships, "
                "then answers multi-hop questions by traversing that graph.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"name": "Graph RAG API", "docs": "/docs", "health": "/api/health"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
