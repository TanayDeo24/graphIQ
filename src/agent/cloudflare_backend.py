"""Cloudflare Workers AI backend (LLM_BACKEND=cloudflare) -- the
production backend, called via Cloudflare's REST API.

Model substitution, found and verified live against this project's actual
Cloudflare account before being wired up (not assumed from documentation):
`@cf/defog/sqlcoder-7b-2`, the model originally specified for SQL
generation, returns HTTP 410 Gone -- "deprecated on 2026-05-30" -- current
and confirmed at build time. `@cf/qwen/qwen2.5-coder-32b-instruct` was
selected as the closest current live equivalent (a current, actively
maintained code/SQL-specialized model) and verified with a real prompt
that it returns correct, executable SQL before being adopted. The general
model (`@cf/meta/llama-3.2-3b-instruct`) and the embedding model
(`@cf/baai/bge-base-en-v1.5`, 768-dim, chosen from Cloudflare's live model
catalog since `@cf/baai/bge-*` in the spec was a family, not one exact
id) were both verified live too.

Implements the same interface as local_backend.py:
  generate(prompt, system_instruction=None, max_tokens=800) -> str
  generate_sql(question, schema_context) -> str
  embed_texts(texts) -> list[list[float]]
so orchestrator.py, sql_agent.py, and doc_retrieval.py never branch on
which backend is active (src/agent/llm_backend.py picks one at import time
based on LLM_BACKEND).
"""

import os

import requests

ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")

GENERAL_MODEL = os.environ.get("CLOUDFLARE_GENERAL_MODEL", "@cf/meta/llama-3.2-3b-instruct")
SQLCODER_MODEL = os.environ.get("CLOUDFLARE_SQLCODER_MODEL", "@cf/qwen/qwen2.5-coder-32b-instruct")
EMBEDDING_MODEL = os.environ.get("CLOUDFLARE_EMBEDDING_MODEL", "@cf/baai/bge-base-en-v1.5")

_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
_TIMEOUT_SECONDS = 60


def _run(model: str, payload: dict) -> dict:
    if not ACCOUNT_ID or not API_TOKEN:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set -- required when LLM_BACKEND=cloudflare."
        )
    url = _BASE_URL.format(account_id=ACCOUNT_ID, model=model)
    resp = requests.post(
        url, headers={"Authorization": f"Bearer {API_TOKEN}"}, json=payload, timeout=_TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success", True):
        raise RuntimeError(f"Cloudflare Workers AI error for {model}: {data.get('errors')}")
    return data["result"]


def _response_text(result: dict) -> str:
    """Cloudflare's `result.response` field is auto-parsed into a dict
    when the model's raw output looks like JSON (found live: asking for a
    JSON object back made `result["response"]` a real dict, not a string,
    breaking every caller that expects text) -- `result.choices[0].message.content`
    is always the model's literal raw string output regardless of its
    shape, so that's used here instead of the seemingly-more-convenient
    `response` field."""
    choices = result.get("choices") or []
    if choices:
        content = choices[0].get("message", {}).get("content")
        if content:
            return content.strip()
    # Fallback for older/simpler Workers AI response shapes without `choices`.
    response = result.get("response")
    if isinstance(response, str):
        return response.strip()
    return ""


def generate(prompt, system_instruction: str = None, max_tokens: int = 800) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    if isinstance(prompt, list):
        messages.extend(prompt)
    else:
        messages.append({"role": "user", "content": prompt})
    # temperature=0: routing/synthesis need consistent, parseable structured
    # output, not creative variance -- found live to reduce (not eliminate)
    # a smaller free-tier model's tendency to run on past its budget or
    # produce inconsistent JSON shape across otherwise-identical calls.
    result = _run(GENERAL_MODEL, {"messages": messages, "max_tokens": max_tokens, "temperature": 0})
    return _response_text(result)


SQL_SYSTEM_INSTRUCTION = (
    "You write PostgreSQL SELECT statements only. Respond with raw SQL text -- no markdown code fences, "
    "no explanation, no commentary, just the query."
)


def generate_sql(question: str, schema_context: str) -> str:
    prompt = f"Database schema:\n{schema_context}\n\nQuestion: {question}\n\nSQL query:"
    result = _run(
        SQLCODER_MODEL,
        {"messages": [{"role": "system", "content": SQL_SYSTEM_INSTRUCTION}, {"role": "user", "content": prompt}],
         "max_tokens": 500},
    )
    return _response_text(result)


def embed_texts(texts: list) -> list:
    result = _run(EMBEDDING_MODEL, {"text": texts})
    return result["data"]
