"""LangChain-backed orchestration that grounds answers in retrieved evidence."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:  # Provider-specific client is optional
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore

from ..core.config import Settings
from ..core.logging_config import get_logger
from .hybrid_retriever import HybridRetriever

logger = get_logger(__name__)


@dataclass
class QueryResult:
    answer: str
    references: List[Dict]
    grounded: bool
    latency_ms: int


class QueryOrchestrator:
    def __init__(self, retriever: HybridRetriever, settings: Settings) -> None:
        self.retriever = retriever
        self.settings = settings
        self.llm = self._build_llm()
        self.chain = self._build_chain(self.llm) if self.llm else None

    def answer(self, question: str, top_k: int, audit_trail: bool) -> QueryResult:
        start = time.perf_counter()
        hits = self.retriever.retrieve(question, top_k)
        answer = self._generate_answer(question, hits)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        references = hits if audit_trail else []
        return QueryResult(
            answer=answer,
            references=references,
            grounded=bool(hits),
            latency_ms=elapsed_ms,
        )

    def _build_llm(self):
        if self.settings.llm_provider == "openai" and self.settings.openai_api_key and ChatOpenAI:
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                openai_api_key=self.settings.openai_api_key,
            )
        logger.warning("no LLM configured, falling back to extractive answers")
        return None

    def _build_chain(self, llm) -> Optional:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a compliance-focused RAG assistant. Answer only with provided context and cite chunk_ids.",
                ),
                (
                    "human",
                    "Question: {question}\nContext:\n{context}\nFormat answers with citations like [chunk_id].",
                ),
            ]
        )
        return prompt | llm | StrOutputParser()

    def _generate_answer(self, question: str, hits: List[Dict]) -> str:
        if not hits:
            return "No relevant answer found."
        context_block = self._format_context(hits)
        if self.chain:
            try:
                return self.chain.invoke({"question": question, "context": context_block})
            except Exception as err:  # pragma: no cover
                logger.warning("LLM invocation failed, using fallback", exc_info=err)
        return self._fallback_answer(hits, question)

    def _format_context(self, hits: List[Dict]) -> str:
        formatted = []
        for idx, hit in enumerate(hits, start=1):
            formatted.append(
                f"[{hit['chunk_id']}] doc={hit['document_id']} level={hit['level']} score={hit['score']:.4f}\n{hit['content']}"
            )
        return "\n\n".join(formatted)

    def _fallback_answer(self, hits: List[Dict], question: str) -> str:
        best = hits[0]
        return (
            "Answer synthesized without LLM due to configuration limits.\n"
            f"Question: {question}\n"
            f"Best match ({best['chunk_id']}): {best['content']}"
        )
