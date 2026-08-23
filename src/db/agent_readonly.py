"""One-time (idempotent) setup for the explainability agent's read-only
Postgres role. Run as: python -m src.db.agent_readonly

Creates graphiq_agent_readonly (if it doesn't already exist) using the
app's normal read-write connection (which owns the database and can grant
privileges), sets its password from AGENT_READONLY_DB_PASSWORD, and
applies sql/agent_readonly_grants.sql to grant SELECT on the validated
result tables only. Safe to re-run any time the result-table set changes
-- the grants file REVOKEs-then-GRANTs, so a table dropped from the
allowlist there loses its grant on the next run rather than accumulating
stale access.
"""

import os
import secrets

from dotenv import load_dotenv
from sqlalchemy import text

from src.agent.sql_schema import ALLOWED_TABLES
from src.db.connection import get_engine

load_dotenv()

ROLE_NAME = "graphiq_agent_readonly"
GRANTS_SQL_PATH = "sql/agent_readonly_grants.sql"


def _generate_password() -> str:
    return secrets.token_urlsafe(24)


def create_agent_readonly_role() -> str:
    """Returns the password in effect after this call (existing one from
    .env if already set there, otherwise a freshly generated one -- which
    the caller is responsible for saving to .env, since this function has
    no way to write to that file safely on the caller's behalf)."""
    password = os.environ.get("AGENT_READONLY_DB_PASSWORD") or _generate_password()
    engine = get_engine()

    with engine.begin() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :role"), {"role": ROLE_NAME}
        ).scalar()
        if exists:
            conn.execute(text(f"ALTER ROLE {ROLE_NAME} WITH LOGIN PASSWORD :pw"), {"pw": password})
        else:
            conn.execute(text(f"CREATE ROLE {ROLE_NAME} WITH LOGIN PASSWORD :pw"), {"pw": password})

    # The database name differs between local Postgres ("graphiq") and
    # Neon ("neondb" by default) -- resolved from the active connection
    # itself rather than hardcoded in the .sql file (which broke on Neon
    # the first time this was applied there, found live).
    db_name = engine.url.database
    with engine.begin() as conn:
        conn.execute(text(f'GRANT CONNECT ON DATABASE "{db_name}" TO {ROLE_NAME}'))

    with open(GRANTS_SQL_PATH) as f:
        grants_ddl = f.read()
    table_list = ", ".join(ALLOWED_TABLES.keys())
    grants_ddl += f"\nGRANT SELECT ON {table_list} TO {ROLE_NAME};\n"

    with engine.begin() as conn:
        conn.execute(text(grants_ddl))

    return password


if __name__ == "__main__":
    pw = create_agent_readonly_role()
    already_had_it = bool(os.environ.get("AGENT_READONLY_DB_PASSWORD"))
    print(f"Role {ROLE_NAME!r} created/updated, grants applied from {GRANTS_SQL_PATH}.")
    if already_had_it:
        print("Password unchanged (AGENT_READONLY_DB_PASSWORD was already set in .env).")
    else:
        print(f"Generated password -- add this to your .env:\nAGENT_READONLY_DB_PASSWORD={pw}")
