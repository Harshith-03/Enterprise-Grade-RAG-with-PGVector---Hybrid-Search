"""Centralized runtime configuration using pydantic settings."""
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level app configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Enterprise RAG")
    app_env: str = Field(default="local", alias="APP_ENV")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="ragdb", alias="POSTGRES_DB")
    postgres_user: str = Field(default="rag_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="rag_pass", alias="POSTGRES_PASSWORD")
    postgres_dsn: Optional[str] = Field(default=None, alias="POSTGRES_DSN")
    pgvector_schema: str = Field(default="public", alias="PGVECTOR_SCHEMA")

    embedding_dim: int = Field(default=768, alias="EMBEDDING_DIM")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2", alias="EMBEDDING_MODEL_NAME"
    )
    max_hierarchical_depth: int = Field(default=3, alias="MAX_HIERARCHICAL_DEPTH")

    bm25_k1: float = Field(default=1.5, alias="BM25_K1")
    bm25_b: float = Field(default=0.75, alias="BM25_B")
    rrf_k: int = Field(default=60, alias="RRF_K")

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, alias="HUGGINGFACE_API_KEY")

    rag_evaluation_sample_size: int = Field(default=50, alias="RAG_EVALUATION_SAMPLE_SIZE")
    rag_eval_top_k: int = Field(default=8, alias="RAG_EVAL_TOP_K")

    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    tracing_tags: List[str] = Field(default_factory=list)

    def postgres_url(self) -> str:
        """Build a SQLAlchemy DSN if one is not explicitly provided."""

        if self.postgres_dsn:
            return self.postgres_dsn
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def telemetry_context(self) -> Dict[str, Any]:
        """Small helper for structured logs and traces."""

        return {
            "app_env": self.app_env,
            "feature_flags": sorted(k for k, v in self.feature_flags.items() if v),
        }


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
