"""Persistence layer for hierarchical chunks stored in PostgreSQL + pgvector."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterable, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Column, DateTime, MetaData, String, Table, Text, select
from sqlalchemy.dialects.postgresql import ARRAY, insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class PGVectorStore:
    """Wraps SQLAlchemy metadata and CRUD helpers."""

    def __init__(self, session_factory: sessionmaker, embedding_dim: int) -> None:
        self.session_factory = session_factory
        self.embedding_dim = embedding_dim
        self.metadata = MetaData()
        self.chunks = Table(
            "document_chunks",
            self.metadata,
            Column("id", String, primary_key=True),
            Column("document_id", String, nullable=False, index=True),
            Column("parent_id", String, nullable=True),
            Column("level", String, nullable=False),
            Column("content", Text, nullable=False),
            Column("metadata", JSON, nullable=False, default={}),
            Column("embedding", Vector(embedding_dim)),
            Column("sparse_tokens", ARRAY(String)),
            Column("created_at", DateTime, default=datetime.utcnow, nullable=False),
        )
        self._ensure_schema()

    @property
    def engine(self) -> Engine:
        return self.session_factory.kw["bind"]

    def _ensure_schema(self) -> None:
        self.metadata.create_all(self.engine, checkfirst=True)

    @contextmanager
    def session_scope(self) -> Session:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_chunks(self, payloads: Iterable[Dict]) -> int:
        rows = list(payloads)
        if not rows:
            return 0
        with self.session_scope() as session:
            for row in rows:
                stmt = (
                    insert(self.chunks)
                    .values(
                        id=row["chunk_id"],
                        document_id=row["document_id"],
                        parent_id=row["parent_id"],
                        level=row["level"],
                        content=row["content"],
                        metadata=row["metadata"],
                        embedding=row.get("embedding"),
                        sparse_tokens=row.get("sparse_tokens"),
                    )
                    .on_conflict_do_update(
                        index_elements=[self.chunks.c.id],
                        set_={
                            "content": row["content"],
                            "metadata": row["metadata"],
                            "embedding": row.get("embedding"),
                            "sparse_tokens": row.get("sparse_tokens"),
                        },
                    )
                )
                session.execute(stmt)
        logger.info("upserted chunks", extra={"count": len(rows)})
        return len(rows)

    def dense_search(self, query_vector: List[float], limit: int) -> List[Dict]:
        stmt = (
            select(
                self.chunks.c.id,
                self.chunks.c.document_id,
                self.chunks.c.content,
                self.chunks.c.level,
                self.chunks.c.metadata,
                self.chunks.c.embedding.cosine_distance(query_vector).label("distance"),
            )
            .where(self.chunks.c.embedding.is_not(None))
            .order_by("distance")
            .limit(limit)
        )
        with self.session_scope() as session:
            rows = session.execute(stmt).all()
        return [
            {
                "chunk_id": row.id,
                "document_id": row.document_id,
                "content": row.content,
                "level": row.level,
                "metadata": row.metadata,
                "score": 1 - row.distance,
            }
            for row in rows
        ]

    def fetch_sparse_corpus(self, limit: int = 5000) -> List[Dict]:
        stmt = select(
            self.chunks.c.id,
            self.chunks.c.document_id,
            self.chunks.c.content,
            self.chunks.c.sparse_tokens,
            self.chunks.c.metadata,
        ).limit(limit)
        with self.session_scope() as session:
            rows = session.execute(stmt).all()
        return [
            {
                "chunk_id": row.id,
                "document_id": row.document_id,
                "content": row.content,
                "tokens": row.sparse_tokens or [],
                "metadata": row.metadata,
            }
            for row in rows
        ]
