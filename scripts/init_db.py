"""Bootstrap PostgreSQL schema and enable pgvector."""
from __future__ import annotations

import argparse

from sqlalchemy import text
from sqlalchemy.engine import create_engine

from app.core.config import Settings


def init_db(settings: Settings) -> None:
    engine = create_engine(settings.postgres_url())
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize pgvector schema")
    parser.add_argument("--dsn", type=str, default=None, help="Override database DSN")
    args = parser.parse_args()

    settings = Settings()
    if args.dsn:
        settings.postgres_dsn = args.dsn  # type: ignore[attr-defined]
    init_db(settings)
    print("pgvector extension ensured")
