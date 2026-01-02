"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI

from .api.routers import router
from .core.logging_config import configure_logging

configure_logging()
app = FastAPI(title="Enterprise RAG", version="0.1.0")
app.include_router(router)
