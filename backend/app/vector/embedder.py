"""
Embedding model wrapper.

Uses sentence-transformers `BAAI/bge-large-en-v1.5` locally (spec §4) — better than
OpenAI for retrieval, and free/offline. The model is downloaded on first use into
the configured cache dir.

Also exposes a singleton reranker (bge-reranker-v2-m3 cross-encoder) for retrieval.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded singletons (loading ~1.3GB of weights is expensive).
_embedder = None
_reranker = None


def _ensure_cache_dir() -> None:
    os.makedirs(settings.models_cache_dir, exist_ok=True)


class Embedder:
    """Sentence-transformers embedder (BGE-large-en-v1.5)."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None

    def _load(self):
        if self._model is None:
            _ensure_cache_dir()
            from sentence_transformers import SentenceTransformer

            kwargs = {"cache_folder": settings.models_cache_dir}
            if settings.force_cpu:
                kwargs["device"] = "cpu"
            logger.info("Loading embedding model %s (this downloads on first run)...", self.model_name)
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def encode(self, texts: List[str] | str) -> List[List[float]]:
        """Encode one or more texts into normalized vectors."""
        single = isinstance(texts, str)
        model = self._load()
        vecs = model.encode(
            [texts] if single else texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out = vecs.tolist()
        return out[0] if single else out  # type: ignore[return-value]


class Reranker:
    """Cross-encoder reranker (BGE-reranker-v2-m3)."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self._model = None

    def _load(self):
        if self._model is None:
            _ensure_cache_dir()
            from sentence_transformers import CrossEncoder

            kwargs = {"cache_folder": settings.models_cache_dir}
            if settings.force_cpu:
                kwargs["device"] = "cpu"
            logger.info("Loading reranker model %s ...", self.model_name)
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def rank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[tuple[int, float]]:
        """Score query-doc pairs. Returns list of (doc_index, score) sorted desc."""
        if not documents:
            return []
        model = self._load()
        pairs = [(query, doc) for doc in documents]
        scores = model.predict(pairs).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
