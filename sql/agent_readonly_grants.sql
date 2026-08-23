-- Read-only Postgres role for the explainability agent's text-to-SQL
-- pipeline (src/agent/sql_agent.py). SELECT-only, on the validated result
-- tables enumerated below -- explicitly NOT on the raw source tables
-- (employees, comp_history, performance_reviews, benefits_enrollment,
-- expense_transactions). `departments` is included: it's a pure
-- id -> name lookup with no employee-level or otherwise sensitive data,
-- and several result tables (e.g. spend_dollar_treemap) already join
-- against it in the app's own existing routes.
--
-- Role creation (with password) is handled by
-- src/db/agent_readonly.py, not this file, so the password never has to
-- be embedded in a checked-in SQL file. This file only grants privileges
-- to a role that's assumed to already exist -- run
-- `python -m src.db.agent_readonly` to create the role AND apply these
-- grants in one step.

-- GRANT CONNECT ON DATABASE is deliberately NOT here: the database name
-- differs between local Postgres ("graphiq") and Neon ("neondb" by
-- default), and hardcoding either would break the other -- found live
-- when this file was first applied against Neon. src/db/agent_readonly.py
-- issues that one grant separately, against whichever database name the
-- active connection actually resolves to.
GRANT USAGE ON SCHEMA public TO graphiq_agent_readonly;

-- Revoke first so re-applying this after the result-table set changes
-- never leaves a stale grant on a table that's since been dropped from
-- the allowlist. The actual per-table GRANT SELECT statement is NOT
-- here -- it's generated at apply time by src/db/agent_readonly.py from
-- src/agent/sql_schema.py's ALLOWED_TABLES, the single source of truth
-- for this allowlist, so this file can never silently drift out of sync
-- with it. This file only holds the boilerplate that doesn't depend on
-- the table list.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM graphiq_agent_readonly;

ALTER ROLE graphiq_agent_readonly SET statement_timeout = '5s';
