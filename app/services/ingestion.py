"""Document ingestion orchestrator."""
from __future__ import annotations

import time
import uuid
from typing import Dict, List

from ..core.logging_config import get_logger
from .chunking import Chunk, HierarchicalChunker
from .embedding import EmbeddingService
from .pgvector_store import PGVectorStore
from .table_extractor import TableExtractionService

logger = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        store: PGVectorStore,
        embedder: EmbeddingService,
        chunker: HierarchicalChunker,
        table_service: TableExtractionService,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunker = chunker
        self.table_service = table_service

    def ingest(self, document_bytes: bytes, filename: str, metadata: Dict[str, str]) -> Dict[str, int]:
        start = time.perf_counter()
        document_id = metadata.get("document_id") or str(uuid.uuid4())

        text, tables = self.table_service.parse(document_bytes=document_bytes, filename=filename)
        base_metadata = {**metadata, "document_id": document_id}
        chunks = self.chunker.chunk_document(document_id=document_id, text=text, metadata=base_metadata)

        table_chunks: List[Chunk] = [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                parent_id=None,
                level="table",
                content=table["csv"],
                metadata={
                    **base_metadata,
                    "table_id": table.get("table_id"),
                    "table_title": table.get("title"),
                    "source_file": filename,
                },
            )
            for table in tables
            if table.get("csv")
        ]
        chunks.extend(table_chunks)

        embeddings = self.embedder.embed(chunk.content for chunk in chunks)
        payloads = []
        for chunk, vector in zip(chunks, embeddings):
            chunk_metadata = {**chunk.metadata, "level": chunk.level}
            payloads.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "parent_id": chunk.parent_id,
                    "level": chunk.level,
                    "content": chunk.content,
                    "metadata": chunk_metadata,
                    "embedding": vector,
                    "sparse_tokens": self.embedder.tokenize(chunk.content),
                }
            )
        stored = self.store.upsert_chunks(payloads)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "ingested document",
            extra={"document_id": document_id, "chunks": stored, "elapsed_ms": elapsed_ms},
        )
        return {
            "chunks": stored,
            "tables": len(tables),
            "elapsed_ms": elapsed_ms,
            "document_id": document_id,
        }
