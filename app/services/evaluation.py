"""Evaluation helpers built on top of RAGAS."""
from __future__ import annotations

from datetime import datetime
from typing import List

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics.collections import answer_relevancy, faithfulness
except Exception:  # pragma: no cover
    Dataset = None  # type: ignore
    evaluate = None  # type: ignore
    faithfulness = None  # type: ignore
    answer_relevancy = None  # type: ignore

from ..core.logging_config import get_logger
from ..models.schemas import EvaluationMetrics, EvaluationResponse, EvaluationSample

logger = get_logger(__name__)


class EvaluationService:
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def run(self, samples: List[EvaluationSample]) -> EvaluationResponse:
        if not samples:
            raise ValueError("No evaluation samples provided")
        if not Dataset or not evaluate:
            raise RuntimeError("ragas and datasets must be installed to evaluate")

        dataset = Dataset.from_dict(
            {
                "question": [s.question for s in samples],
                "answer": [s.answer for s in samples],
                "contexts": [s.citations for s in samples],
                "ground_truth": [s.ground_truth for s in samples],
            }
        )
        results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        metrics = EvaluationMetrics(
            faithfulness=float(results["faithfulness"]),
            answer_relevancy=float(results["answer_relevancy"]),
            groundedness=self._groundedness(samples),
            recall_at_k=self._recall_at_k(samples, self.top_k),
        )
        logger.info("evaluation completed", extra=metrics.model_dump())
        return EvaluationResponse(
            samples_evaluated=len(samples),
            metrics=metrics,
            generated_at=datetime.utcnow(),
        )

    def _groundedness(self, samples: List[EvaluationSample]) -> float:
        grounded = 0
        for sample in samples:
            if any(sample.ground_truth.lower() in citation.lower() for citation in sample.citations):
                grounded += 1
        return grounded / len(samples)

    def _recall_at_k(self, samples: List[EvaluationSample], k: int) -> float:
        hits = 0
        for sample in samples:
            window = sample.citations[:k]
            if any(sample.ground_truth.lower() in citation.lower() for citation in window):
                hits += 1
        return hits / len(samples)
