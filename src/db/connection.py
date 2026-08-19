"""Shared Postgres connection helpers for GraphIQ.

All of src/db, src/models, and src/api read connection settings from the
same .env file so every component talks to the same database.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def get_database_url() -> str:
    user = os.environ.get("POSTGRES_USER", "graphiq")
    password = os.environ.get("POSTGRES_PASSWORD", "graphiq_dev_password")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "graphiq")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(get_database_url(), pool_pre_ping=True)
    return _engine
