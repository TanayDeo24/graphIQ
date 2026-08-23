"""Local MLX backend (LLM_BACKEND=local) -- Apple Silicon only, for
development on an M3 Mac.

Two large models are not kept resident simultaneously, per the build
spec's 16GB memory budget: SQLCoder-7B-2 (MLX, quantized to 4-bit locally
via `mlx_lm.convert` -- mlx-community only hosts it at full precision, no
pre-quantized 4-bit version exists there) is loaded only when a SQL-
generation call is actually needed and unloaded immediately after; a small
3B instruct model (Qwen2.5-3B-Instruct-4bit, from mlx-community) is kept
resident the rest of the time for routing, synthesis, and general-concept
answers. A single-slot "currently loaded" cache means consecutive calls of
the same kind (e.g. two synthesis calls in a row) don't reload anything.

SQLCoder-7B-2 is a completion-style fine-tune, not an instruction/chat
model -- it does NOT use a chat template. generate_sql() builds Defog's
own documented prompt format directly
(https://huggingface.co/defog/sqlcoder-7b-2: "### Task ... ### Database
Schema ... ### Answer ... [SQL]") rather than routing through
tokenizer.apply_chat_template, which would produce a malformed prompt for
this specific model. Defog's own docs recommend do_sample=False,
num_beams=4; mlx_lm's generate() supports greedy decoding (used here) but
not beam search, a known, disclosed simplification.

Implements the same interface as cloudflare_backend.py:
  generate(prompt, system_instruction=None, max_tokens=800) -> str
  generate_sql(question, schema_context) -> str
  embed_texts(texts) -> list[list[float]]
"""

import gc
import os

import mlx.core as mx
from mlx_lm import generate as _mlx_generate
from mlx_lm import load as _mlx_load
from mlx_lm.sample_utils import make_sampler

SQLCODER_PATH = os.environ.get("LOCAL_SQLCODER_PATH", "models/sqlcoder-7b-2-4bit")
GENERAL_MODEL_PATH = os.environ.get("LOCAL_GENERAL_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
EMBEDDING_MODEL_NAME = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

_GREEDY_SAMPLER = make_sampler(temp=0.0)

_state = {"which": None, "model": None, "tokenizer": None}
_embed_model = None


def _ensure_loaded(which: str):
    if _state["which"] == which and _state["model"] is not None:
        return _state["model"], _state["tokenizer"]
    if _state["model"] is not None:
        _state["model"] = None
        _state["tokenizer"] = None
        gc.collect()
        mx.clear_cache()
    path = SQLCODER_PATH if which == "sql" else GENERAL_MODEL_PATH
    model, tokenizer = _mlx_load(path)
    _state.update(which=which, model=model, tokenizer=tokenizer)
    return model, tokenizer


def generate(prompt, system_instruction: str = None, max_tokens: int = 800) -> str:
    model, tokenizer = _ensure_loaded("general")
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    if isinstance(prompt, list):
        messages.extend(prompt)
    else:
        messages.append({"role": "user", "content": prompt})
    chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return _mlx_generate(
        model, tokenizer, prompt=chat_prompt, max_tokens=max_tokens, sampler=_GREEDY_SAMPLER, verbose=False
    ).strip()


def generate_sql(question: str, schema_context: str) -> str:
    model, tokenizer = _ensure_loaded("sql")
    # Defog's own documented prompt format for SQLCoder-7B-2 -- see
    # module docstring. Not a chat template.
    prompt = (
        f"### Task\nGenerate a SQL query to answer [QUESTION]{question}[/QUESTION]\n\n"
        f"### Database Schema\nThe query will run on a database with the following schema:\n{schema_context}\n\n"
        f"### Answer\nGiven the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]\n[SQL]"
    )
    result = _mlx_generate(
        model, tokenizer, prompt=prompt, max_tokens=500, sampler=_GREEDY_SAMPLER, verbose=False
    )
    return result.strip()


def embed_texts(texts: list) -> list:
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        # MPS (Apple Silicon GPU via PyTorch) if available, else CPU --
        # this is a small model over a small (~24-chunk) corpus either way.
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _embed_model.encode(texts, normalize_embeddings=True).tolist()
