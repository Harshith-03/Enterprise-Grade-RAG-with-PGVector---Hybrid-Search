"""Hybrid retrieval that blends dense and sparse signals via RRF."""
from __future__ import annotations

from typing import Dict, List

from rank_bm25 import BM25Okapi

from ..utils.rrf import reciprocal_rank_fusion
from .embedding import EmbeddingService
from .pgvector_store import PGVectorStore


class HybridRetriever:
    def __init__(
        self,
        store: PGVectorStore,
        embedder: EmbeddingService,
        bm25_k1: float,
        bm25_b: float,
        rrf_k: int,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 8) -> List[Dict]:
        dense_hits = self._dense_retrieval(query, top_k * 2)
        sparse_hits = self._sparse_retrieval(query, top_k * 4)
        fused = reciprocal_rank_fusion(
            {
                "dense": [(hit["chunk_id"], hit["score"]) for hit in dense_hits],
                "sparse": [(hit["chunk_id"], hit["score"]) for hit in sparse_hits],
            },
            k=self.rrf_k,
        )
        fused_map = {chunk_id: score for chunk_id, score, _ in fused}
        enriched: Dict[str, Dict] = {hit["chunk_id"]: hit for hit in dense_hits + sparse_hits}
        ranked = sorted(
            (
                {
                    **enriched[chunk_id],
                    "score": fused_map[chunk_id],
                }
                for chunk_id in fused_map
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked[:top_k]

    def _dense_retrieval(self, query: str, top_k: int) -> List[Dict]:
        query_vec = self.embedder.embed([query])[0]
        return self.store.dense_search(query_vec, top_k)

    def _sparse_retrieval(self, query: str, top_k: int) -> List[Dict]:
        corpus = self.store.fetch_sparse_corpus()
        if not corpus:
            return []
        tokenized_docs = [doc["tokens"] for doc in corpus]
        bm25 = BM25Okapi(tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
        query_tokens = self.embedder.tokenize(query)
        scores = bm25.get_scores(query_tokens)
        items = [
            {
                "chunk_id": doc["chunk_id"],
                "document_id": doc["document_id"],
                "content": doc["content"],
                "level": doc["metadata"].get("level", "paragraph"),
                "metadata": doc["metadata"],
                "score": score,
            }
            for doc, score in zip(corpus, scores)
        ]
        items.sort(key=lambda item: item["score"], reverse=True)
        return items[:top_k]
