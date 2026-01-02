"""Hierarchical chunking strategies for regulatory documents."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from ..core.logging_config import get_logger

logger = get_logger(__name__)

default_split_regex = re.compile(r"\n{2,}")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    parent_id: Optional[str]
    level: str
    content: str
    metadata: Dict[str, str]


class HierarchicalChunker:
    """Splits content across title, section, and paragraph granularity."""

    def __init__(self, max_depth: int = 3) -> None:
        self.max_depth = max_depth

    def chunk_document(self, document_id: str, text: str, metadata: Dict[str, str]) -> List[Chunk]:
        if not text.strip():
            return []

        title_chunk = self._make_chunk(document_id, None, "title", metadata.get("title", text[:80]), metadata)
        sections = self._split_sections(text)

        chunks: List[Chunk] = [title_chunk]
        for section_title, section_body in sections:
            section_chunk = self._make_chunk(document_id, title_chunk.chunk_id, "section", section_title, metadata)
            chunks.append(section_chunk)

            paragraph_chunks = self._split_paragraphs(
                document_id=document_id,
                parent_id=section_chunk.chunk_id,
                body=section_body,
                metadata=metadata,
            )
            chunks.extend(paragraph_chunks)

        logger.info("chunked document", extra={"document_id": document_id, "chunks": len(chunks)})
        return chunks

    def _make_chunk(
        self,
        document_id: str,
        parent_id: Optional[str],
        level: str,
        content: str,
        metadata: Dict[str, str],
    ) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            parent_id=parent_id,
            level=level,
            content=content.strip(),
            metadata=metadata,
        )

    def _split_sections(self, text: str) -> List[tuple[str, str]]:
        segments = default_split_regex.split(text)
        sections: List[tuple[str, str]] = []
        for segment in segments:
            heading, body = self._extract_heading(segment)
            sections.append((heading, body))
        return sections

    def _split_paragraphs(
        self,
        document_id: str,
        parent_id: str,
        body: str,
        metadata: Dict[str, str],
    ) -> List[Chunk]:
        paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                parent_id=parent_id,
                level="paragraph",
                content=paragraph,
                metadata=metadata,
            )
            for paragraph in paragraphs
        ]

    def _extract_heading(self, segment: str) -> tuple[str, str]:
        lines = [line.strip() for line in segment.split("\n") if line.strip()]
        if not lines:
            return ("Untitled Section", "")
        heading = lines[0]
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""
        return (heading, body)

    @staticmethod
    def tokenize(content: Iterable[str]) -> List[List[str]]:
        return [[token.lower() for token in re.findall(r"\w+", text)] for text in content]
