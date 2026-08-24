"""Dual LLM backend switch. LLM_BACKEND ("local" or "cloudflare") picks
the entire backend at import time -- no other agent code (sql_agent.py,
doc_retrieval.py, orchestrator.py) ever branches on which one is active;
they only ever call the three functions re-exported here.

  local:      src/agent/local_backend.py      -- MLX on Apple Silicon, dev
  cloudflare: src/agent/cloudflare_backend.py  -- Workers AI REST API, prod

Both implement the identical interface:
  generate(prompt, system_instruction=None, max_tokens=800) -> str
  generate_sql(question, schema_context) -> str
  embed_texts(texts) -> list[list[float]]
"""

import os

from dotenv import load_dotenv

load_dotenv()

# No default. A missing LLM_BACKEND on a deploy target must fail loudly
# and immediately, with a clear message pointing at the actual cause --
# not silently fall back to "local" and crash three layers down inside
# local_backend.py with a confusing `ModuleNotFoundError: No module named
# 'mlx'` (mlx isn't even installed on non-macOS deploy targets in the
# first place -- see requirements-local.txt). Found live: this exact
# silent-default behavior is the most plausible reason a deploy target
# with LLM_BACKEND unset in its real environment (as opposed to what
# render.yaml *says* it should be -- render.yaml's envVars only
# auto-apply to a service actually deployed as a Blueprint, not to an
# existing service that was created manually through the dashboard
# before render.yaml existed in this repo) would hit the mlx crash
# despite this module's own conditional-import logic being correct.
BACKEND = os.environ.get("LLM_BACKEND")

if BACKEND == "local":
    from src.agent import local_backend as _impl
elif BACKEND == "cloudflare":
    from src.agent import cloudflare_backend as _impl
elif BACKEND is None:
    raise RuntimeError(
        "LLM_BACKEND is not set in the environment. This must be set explicitly to "
        "'local' or 'cloudflare' -- there is no default, on purpose, so a missing "
        "value fails immediately and clearly here rather than silently picking the "
        "wrong backend and crashing later with an unrelated-looking import error. "
        "Set LLM_BACKEND in your .env (local dev) or in the actual deploy "
        "environment's env vars (Render dashboard -> Environment tab, not just "
        "render.yaml -- render.yaml's envVars only auto-apply to a service deployed "
        "as a Blueprint, not one created manually through the dashboard)."
    )
else:
    raise RuntimeError(f"Unknown LLM_BACKEND={BACKEND!r} -- expected 'local' or 'cloudflare'.")


def generate(prompt, system_instruction: str = None, max_tokens: int = 800) -> str:
    return _impl.generate(prompt, system_instruction=system_instruction, max_tokens=max_tokens)


def generate_sql(question: str, schema_context: str) -> str:
    return _impl.generate_sql(question, schema_context)


def embed_texts(texts: list) -> list:
    return _impl.embed_texts(texts)
