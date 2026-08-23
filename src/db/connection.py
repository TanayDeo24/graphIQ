"""Shared Postgres connection helpers for GraphIQ.

All of src/db, src/models, and src/api read connection settings from the
same .env file so every component talks to the same database.

DATABASE_URL (Neon's connection string in production, set by Render) takes
priority over the discrete POSTGRES_* vars when present -- the standard
12-factor pattern, and the only way this app can reach its database at all
once deployed (there's no local Postgres on Render's servers). Locally,
simply not setting DATABASE_URL keeps using the POSTGRES_* vars against a
local Postgres instance as before; setting it (e.g. to point a local run
at Neon for verification, as this project's own build process did) is an
explicit, deliberate choice, not an accident -- be careful running any
data-mutating script (load_to_postgres, model-fitting evaluate scripts)
while it's set, since those would then target the live Neon database.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url

load_dotenv()


def get_database_url() -> str:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
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


def get_agent_readonly_database_url() -> str:
    """Connection string for the explainability agent's SQL-generation
    pipeline (src/agent/sql_agent.py) -- a dedicated, separate Postgres
    role (graphiq_agent_readonly) with SELECT granted only on validated
    result tables and zero access to raw source tables. Never used for
    anything except executing LLM-generated, validator-approved SQL. See
    src/db/agent_readonly.py for role creation and
    sql/agent_readonly_grants.sql for the actual grants.

    Reuses whichever host/port/db/sslmode the primary connection resolves
    to (Neon in production, local Postgres in dev) via
    get_database_url(), swapping in only the read-only role's own
    credentials -- so this correctly follows DATABASE_URL to Neon without
    a second, separately-maintained connection string."""
    user = os.environ.get("AGENT_READONLY_DB_USER", "graphiq_agent_readonly")
    password = os.environ.get("AGENT_READONLY_DB_PASSWORD", "")
    base = make_url(get_database_url())
    return str(base.set(username=user, password=password))


_agent_readonly_engine: Engine | None = None


def get_agent_readonly_engine() -> Engine:
    global _agent_readonly_engine
    if _agent_readonly_engine is None:
        if not os.environ.get("AGENT_READONLY_DB_PASSWORD"):
            raise RuntimeError(
                "AGENT_READONLY_DB_PASSWORD is not set -- run `python -m src.db.agent_readonly` "
                "to create the read-only role and set this in your .env (see .env.example)."
            )
        # statement_timeout set here too, in addition to the role-level
        # default from sql/agent_readonly_grants.sql -- belt and suspenders
        # against a runaway LLM-generated query, since this connection
        # only ever runs untrusted (validator-checked) SQL.
        _agent_readonly_engine = create_engine(
            get_agent_readonly_database_url(),
            pool_pre_ping=True,
            connect_args={"options": "-c statement_timeout=5000"},
        )
    return _agent_readonly_engine
