"""Pydantic data-transfer objects."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Metadata(BaseModel):
    document_id: str
    source_uri: Optional[str] = None
    filing_type: Optional[str] = None
    fiscal_period: Optional[str] = None
    region: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    tables_indexed: int
    elapsed_ms: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 8
    audit_trail: bool = True


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float
    level: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    answer: str
    references: List[RetrievedChunk]
    grounded: bool
    latency_ms: int


class EvaluationSample(BaseModel):
    question: str
    answer: str
    ground_truth: str
    citations: List[str] = Field(default_factory=list)


class EvaluationMetrics(BaseModel):
    faithfulness: float
    answer_relevancy: float
    groundedness: float
    recall_at_k: float


class EvaluationResponse(BaseModel):
    samples_evaluated: int
    metrics: EvaluationMetrics
    generated_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
