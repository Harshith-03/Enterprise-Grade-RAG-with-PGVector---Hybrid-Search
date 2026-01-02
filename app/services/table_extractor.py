"""Extract financial tables using Docling when available."""
from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

try:  # Docling is optional at runtime
    from docling.document_converter import DocumentConverter
except Exception:  # pragma: no cover - Docling may fail to import without native deps
    DocumentConverter = None  # type: ignore[assignment]

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class TableExtractionService:
    """Tries to extract tables as CSV strings plus rich metadata."""

    def __init__(self) -> None:
        self.converter = DocumentConverter() if DocumentConverter else None

    def parse(self, document_bytes: bytes, filename: str) -> Tuple[str, List[Dict[str, str]]]:
        """Return normalized text plus table metadata for downstream chunking."""

        doc = self._convert(document_bytes, filename)
        if not doc:
            logger.warning("docling unavailable, falling back to utf-8 decode")
            return document_bytes.decode("utf-8", errors="ignore"), []

        text = self._extract_text(doc, document_bytes)
        tables = self._extract_tables(doc, filename)
        return text, tables

    def _convert(self, document_bytes: bytes, filename: str):
        if not self.converter:
            return None
        buffer = io.BytesIO(document_bytes)
        buffer.name = filename
        try:
            return self.converter.convert(buffer)
        except Exception as err:  # pragma: no cover
            logger.warning("docling conversion failed", extra={"file": filename, "error": str(err)})
            return None

    def _extract_tables(self, doc, filename: str) -> List[Dict[str, str]]:
        tables: List[Dict[str, str]] = []
        for table in getattr(doc, "tables", []):
            csv_payload = ""
            try:
                csv_payload = table.to_pandas().to_csv(index=False)
            except Exception as err:  # pragma: no cover - pandas conversions can fail per table
                logger.warning(
                    "failed to serialize table",
                    extra={"table_id": getattr(table, "id", ""), "error": str(err)},
                )
            tables.append(
                {
                    "table_id": getattr(table, "id", ""),
                    "title": getattr(table, "title", ""),
                    "csv": csv_payload,
                }
            )
        logger.info("extracted tables", extra={"count": len(tables), "file": filename})
        return tables

    def _extract_text(self, doc, document_bytes: bytes) -> str:
        rich_text = getattr(doc, "document_text", None) or getattr(doc, "plaintext", None)
        if rich_text:
            return rich_text
        return document_bytes.decode("utf-8", errors="ignore")
