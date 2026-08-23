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

BACKEND = os.environ.get("LLM_BACKEND", "local")

if BACKEND == "local":
    from src.agent import local_backend as _impl
elif BACKEND == "cloudflare":
    from src.agent import cloudflare_backend as _impl
else:
    raise RuntimeError(f"Unknown LLM_BACKEND={BACKEND!r} -- expected 'local' or 'cloudflare'.")


def generate(prompt, system_instruction: str = None, max_tokens: int = 800) -> str:
    return _impl.generate(prompt, system_instruction=system_instruction, max_tokens=max_tokens)


def generate_sql(question: str, schema_context: str) -> str:
    return _impl.generate_sql(question, schema_context)


def embed_texts(texts: list) -> list:
    return _impl.embed_texts(texts)
