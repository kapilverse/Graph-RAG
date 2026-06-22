"""
Central configuration for Graph RAG.

All settings are read from environment variables / a .env file and exposed as a
singleton `settings`. Nothing else in the codebase should touch os.environ
directly — import `settings` instead.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM providers (tried in order; first configured wins)
    # ------------------------------------------------------------------
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_model: str = "gemini-2.0-flash"

    hf_token: str = ""
    hf_base_url: str = "https://router.huggingface.co/v1"
    hf_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    extraction_temperature: float = 0.0
    generation_temperature: float = 0.2
    llm_timeout_seconds: int = 60

    # ------------------------------------------------------------------
    # Embeddings & reranker
    # ------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    force_cpu: bool = True

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag123"
    neo4j_database: str = "neo4j"

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_chunks: str = "graphrag_chunks"
    qdrant_collection_communities: str = "graphrag_communities"

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    local_traversal_depth: int = 2
    local_neighbor_limit: int = 20
    rerank_top_k: int = 10
    community_top_k: int = 5

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir: str = "data"
    models_cache_dir: str = "models_cache"

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    @property
    def llm_providers(self) -> List["LLMProvider"]:
        """Ordered list of configured LLM providers (primary first)."""
        providers: List[LLMProvider] = []
        if self.gemini_api_key:
            providers.append(
                LLMProvider(
                    name="gemini",
                    base_url=self.gemini_base_url,
                    api_key=self.gemini_api_key,
                    model=self.gemini_model,
                )
            )
        if self.hf_token:
            providers.append(
                LLMProvider(
                    name="huggingface",
                    base_url=self.hf_base_url,
                    api_key=self.hf_token,
                    model=self.hf_model,
                )
            )
        return providers


class LLMProvider:
    """A single OpenAI-compatible LLM endpoint."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model = model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()


# Module-level singleton — import this everywhere.
settings = get_settings()
