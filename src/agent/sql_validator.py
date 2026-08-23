"""Validates and sanitizes LLM-generated SQL before it's ever executed
against the read-only database connection. This is the second, independent
layer of defense on top of the graphiq_agent_readonly database role's own
table-level grants (see src/db/agent_readonly.py) -- even if this
validator had a bug, the database role itself has no access to anything
outside the allowlist. Both layers are real, not decorative (see the
adversarial eval in src/agent/eval/ for proof).

Rejects anything that:
  (a) is not a single SELECT statement (no INSERT/UPDATE/DELETE/DROP/
      ALTER/CREATE/etc, no UNION -- kept simple and strict rather than
      trying to enumerate which compound forms are "safe")
  (b) references any table outside ALLOWED_TABLES
  (c) is more than one statement (semicolon-chained)
Auto-appends LIMIT 500 if the query has none.

On any validation failure, the caller should not retry with a looser
check -- return the honest refusal instead (see orchestrator.py).
"""

import sqlglot
from sqlglot import exp

from src.agent.sql_schema import ALLOWED_TABLES

DEFAULT_ROW_LIMIT = 500


class SQLValidationError(Exception):
    pass


def validate_and_prepare_sql(sql_text: str) -> tuple:
    """Returns (validated_sql, referenced_tables) -- the validated
    (LIMIT-ensured) SQL string ready to execute, and the set of real
    table names it references (used by the API layer to build an
    accurate "live query: X table" source label, not guessed after the
    fact). Raises SQLValidationError with a specific reason on any
    rejection."""
    sql_text = (sql_text or "").strip()
    if not sql_text:
        raise SQLValidationError("Empty SQL.")

    try:
        statements = sqlglot.parse(sql_text, read="postgres")
    except Exception as e:
        raise SQLValidationError(f"SQL failed to parse: {e}") from e

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SQLValidationError(
            f"Expected exactly one SQL statement, found {len(statements)} (semicolon-chained input is rejected)."
        )

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise SQLValidationError(
            f"Only single SELECT statements are allowed (got {type(stmt).__name__})."
        )

    # CTE names (WITH t AS (...) SELECT ... FROM t) are picked up by
    # find_all(exp.Table) as if they were real table references, since a
    # CTE reference and a table reference look identical at that point in
    # the tree -- excluded here so a legitimate CTE query isn't rejected
    # for "referencing" its own alias. Table-name comparison is
    # lowercased since Postgres folds unquoted identifiers to lowercase,
    # so a differently-cased but otherwise-legitimate table name (e.g.
    # generated as `Attrition_Risk_Scores`) isn't incorrectly rejected.
    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    allowed_lower = {t.lower() for t in ALLOWED_TABLES}
    referenced_tables = {t.name.lower() for t in stmt.find_all(exp.Table)} - cte_names
    disallowed = referenced_tables - allowed_lower
    if disallowed:
        raise SQLValidationError(
            f"Query references table(s) outside the allowed result-table list: {sorted(disallowed)}."
        )

    if stmt.args.get("limit") is None:
        stmt = stmt.limit(DEFAULT_ROW_LIMIT)

    return stmt.sql(dialect="postgres"), referenced_tables
