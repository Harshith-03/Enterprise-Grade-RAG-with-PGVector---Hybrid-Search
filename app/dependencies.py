"""Dependency wiring for FastAPI routes."""
from functools import lru_cache

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .core.config import Settings, get_settings
from .services.chunking import HierarchicalChunker
from .services.embedding import EmbeddingService
from .services.hybrid_retriever import HybridRetriever
from .services.ingestion import IngestionService
from .services.orchestration import QueryOrchestrator
from .services.pgvector_store import PGVectorStore
from .services.table_extractor import TableExtractionService


@lru_cache
def _engine_factory() -> sessionmaker:
    settings = get_settings()
    engine = create_engine(settings.postgres_url(), pool_pre_ping=True)
    return sessionmaker(bind=engine)


def get_store(settings: Settings = Depends(get_settings)) -> PGVectorStore:
    session_factory = _engine_factory()
    return PGVectorStore(session_factory=session_factory, embedding_dim=settings.embedding_dim)


def get_embedding_service(settings: Settings = Depends(get_settings)) -> EmbeddingService:
    return EmbeddingService(model_name=settings.embedding_model_name)


def get_chunker(settings: Settings = Depends(get_settings)) -> HierarchicalChunker:
    return HierarchicalChunker(max_depth=settings.max_hierarchical_depth)


def get_table_service() -> TableExtractionService:
    return TableExtractionService()


def get_ingestion_service(
    store: PGVectorStore = Depends(get_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    chunker: HierarchicalChunker = Depends(get_chunker),
    table_service: TableExtractionService = Depends(get_table_service),
) -> IngestionService:
    return IngestionService(
        store=store,
        embedder=embedding_service,
        chunker=chunker,
        table_service=table_service,
    )


def get_hybrid_retriever(
    store: PGVectorStore = Depends(get_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    settings: Settings = Depends(get_settings),
) -> HybridRetriever:
    return HybridRetriever(
        store=store,
        embedder=embedding_service,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        rrf_k=settings.rrf_k,
    )


def get_query_orchestrator(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    settings: Settings = Depends(get_settings),
) -> QueryOrchestrator:
    return QueryOrchestrator(retriever=retriever, settings=settings)
