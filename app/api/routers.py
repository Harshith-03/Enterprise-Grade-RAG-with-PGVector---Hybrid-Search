"""FastAPI routers for ingestion, query, evaluation, and health."""
from __future__ import annotations

import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..dependencies import (
    get_ingestion_service,
    get_query_orchestrator,
    get_settings,
)
from ..models.schemas import (
    EvaluationResponse,
    EvaluationSample,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from ..services.evaluation import EvaluationService
from ..services.ingestion import IngestionService
from ..services.orchestration import QueryOrchestrator

router = APIRouter(prefix="/api/v1")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    metadata_json: str = Form("{}"),
    file: UploadFile = File(...),
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    try:
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):  # pragma: no cover - defensive guard
            raise ValueError
    except ValueError as err:  # pragma: no cover - FastAPI handles input validation
        raise HTTPException(status_code=400, detail="metadata must be a JSON object") from err
    payload = await file.read()
    result = ingestion.ingest(document_bytes=payload, filename=file.filename, metadata=metadata)
    return IngestResponse(
        document_id=result["document_id"],
        chunks_indexed=result["chunks"],
        tables_indexed=result["tables"],
        elapsed_ms=result["elapsed_ms"],
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    orchestrator: QueryOrchestrator = Depends(get_query_orchestrator),
) -> QueryResponse:
    result = await run_in_threadpool(lambda: orchestrator.answer(body.question, body.top_k, body.audit_trail))
    references = [
        RetrievedChunk(
            chunk_id=hit["chunk_id"],
            document_id=hit["document_id"],
            content=hit["content"],
            level=hit["level"],
            metadata=hit["metadata"],
            score=hit["score"],
        )
        for hit in result.references
    ]
    return QueryResponse(
        answer=result.answer,
        references=references,
        grounded=result.grounded,
        latency_ms=result.latency_ms,
    )


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    samples: List[EvaluationSample],
    service: EvaluationService = Depends(lambda: EvaluationService(get_settings().rag_eval_top_k)),
) -> EvaluationResponse:
    return service.run(samples)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0", timestamp=datetime.utcnow())
