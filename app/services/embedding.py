"""Embedding utilities for dense and sparse representations."""
from __future__ import annotations

import hashlib
from typing import Iterable, List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as err:  # pragma: no cover - offline environments
                logger.warning("failed to load embedding model, falling back", exc_info=err)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        text_list = list(texts)
        if self.model:
            embeddings = self.model.encode(text_list, normalize_embeddings=True)
            return embeddings.tolist()
        logger.warning("using hash-based embedding fallback")
        return [self._fallback_embedding(text) for text in text_list]

    def tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in text.split() if token.strip()]

    def _fallback_embedding(self, text: str, dim: int = 768) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = (digest * ((dim // len(digest)) + 1))[:dim]
        return [int(b) / 255.0 for b in repeated]
