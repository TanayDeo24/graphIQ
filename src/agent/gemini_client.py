"""Gemini Flash integration via the `google-genai` SDK (the actively
maintained package -- `google-generativeai` is deprecated upstream and was
deliberately not used here), using native function calling for tool
selection only.

Gemini's role in this pipeline is deliberately narrow: given a question,
decide which of the tools in tools.TOOL_REGISTRY (if any) answer it, and
with what arguments. It never writes the final response text -- that's
always a fixed template (src/agent/templates.py) filled with values read
directly from the tool's own return dict in code
(src/agent/orchestrator.py), never transcribed by the model. This split
exists specifically so a model transcription error can't introduce an
ungrounded number; see orchestrator.py's module docstring for the full
design rationale. Automatic function *execution* is explicitly disabled
(AutomaticFunctionCallingConfig(disable=True)) so the orchestrator executes
and logs every call itself (principle 4), rather than the SDK doing it
invisibly.

A second, separate, tightly-scoped call (generate_wrapper_sentence) may
optionally produce one short, number-free lead-in sentence around the
rendered template -- "minimal conversational wrapping" per principle 2.
Its output still passes through the same groundedness check as everything
else before being shown to a user.
"""

import os
import threading
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.agent.tools import TOOL_REGISTRY

load_dotenv()

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Live-tested against this project's actual key: gemini-2.5-flash's free
# tier turned out to cap at just 20 requests/DAY (not per-minute) for this
# project's key -- nowhere near enough for the eval suite's ~90 live
# calls, discovered by actually hitting the quota mid-build, not assumed
# in advance. gemini-3.5-flash-lite (still Flash-family, per the build
# spec's "Gemini Flash integration") has a materially higher free-tier
# quota and was verified to still support native function calling
# correctly against this project's own tool set before switching to it.
# Override via GEMINI_MODEL / GEMINI_MIN_INTERVAL_SECONDS if your key's
# quota differs.
_MIN_CALL_INTERVAL_SECONDS = float(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", "2"))
_rate_limit_lock = threading.Lock()
_last_call_at = [0.0]


def _throttle():
    with _rate_limit_lock:
        wait = _last_call_at[0] + _MIN_CALL_INTERVAL_SECONDS - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call_at[0] = time.time()


def _is_retryable(exc: BaseException) -> bool:
    """Gemini's hosted models return transient 503 ServerErrors under load
    reasonably often, and a free-tier key's quota (even after picking the
    higher-quota model above) can still be brushed up against under real
    call volume -- both are retried with backoff. Not retried: any other
    4xx (bad request, auth, etc.), which waiting won't fix."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"
    return False


_retry_on_transient_error = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=5, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)

SYSTEM_INSTRUCTION = (
    "You are a tool-selection component inside GraphIQ, a workforce-analytics demo. Your only job "
    "is to decide which function(s) from the provided tools best answer the user's question, and "
    "with what arguments -- you do not write explanatory prose yourself, a separate fixed-template "
    "system does that from your function call results. "
    "If a question needs both an employee's risk score and why they're at risk, call both "
    "get_attrition_risk_score and get_attrition_shap. "
    "If a question doesn't clearly match any tool, or asks something outside what these tools cover, "
    "call list_available_topics instead of guessing. "
    "Always call at least one tool -- never answer directly from your own knowledge."
)

WRAPPER_SYSTEM_INSTRUCTION = (
    "Write exactly one short, generic, content-free lead-in sentence to introduce the answer that "
    "follows -- something like 'Here's what the data shows:' or 'Looking at that:'. It must NOT include "
    "any numbers, percentages, or statistics, and it must NOT make any factual claim, judgment, or "
    "characterization about the answer (e.g. never say something 'performs well', 'is high', 'is "
    "reliable', etc.) -- you haven't seen the answer, only the question, so you have no basis to "
    "characterize it. Respond with only the generic lead-in sentence, nothing else."
)


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env (see .env.example) -- the "
            "explainability agent cannot run without it."
        )
    return genai.Client(api_key=api_key)


def _history_to_contents(conversation_history: list) -> list:
    """conversation_history: [{"role": "user"|"model", "text": "..."}] ->
    genai Content list. Only plain prior turns are replayed (not tool-call
    internals), keeping every turn's tool-call/groundedness trail
    independent and re-checkable per principle 4."""
    contents = []
    for turn in conversation_history:
        role = "user" if turn.get("role") == "user" else "model"
        text = turn.get("text", "")
        if text:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return contents


@_retry_on_transient_error
def _generate_content(client, contents, config):
    _throttle()
    return client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)


def request_tool_calls(question: str, page_context: str, conversation_history: list = None) -> list:
    """Sends the question to Gemini with function calling enabled and
    returns the list of function calls it chose, as
    [{"name": ..., "args": {...}}, ...]. Does NOT execute them -- that's
    the orchestrator's job, so every call and its raw result can be logged
    before anything is sent back to the user. Returns an empty list if
    Gemini responded with plain text instead of a function call; the
    orchestrator treats that as "no tool call made" and routes to the
    out-of-scope refusal template, per principle 1 (no tool call means no
    grounded numbers are possible)."""
    client = _client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=list(TOOL_REGISTRY.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = _history_to_contents(conversation_history or [])
    prompt = f"[User is currently viewing: {page_context}]\n{question}" if page_context else question
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

    response = _generate_content(client, contents, config)
    calls = response.function_calls or []
    return [{"name": c.name, "args": dict(c.args or {})} for c in calls]


def generate_wrapper_sentence(question: str) -> str:
    """Best-effort, optional one-sentence lead-in with no numbers allowed
    by instruction (still re-verified by groundedness.check_groundedness()
    downstream, not trusted on instruction alone). Returns "" on any
    failure -- the caller falls back to the bare template text, which is
    always valid on its own."""
    try:
        client = _client()
        config = types.GenerateContentConfig(
            system_instruction=WRAPPER_SYSTEM_INSTRUCTION,
            max_output_tokens=100,
            # gemini-2.5-flash spends part of max_output_tokens on internal
            # "thinking" tokens by default, which was silently truncating
            # this one-sentence, no-reasoning-needed output to nothing
            # useful (found via a live smoke test: a 60-token budget with
            # thinking enabled produced a single truncated word). Not
            # needed for a fixed, tightly-scoped instruction like this one.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        response = _generate_content(client, question, config)
        text = (response.text or "").strip()
        return text
    except Exception:
        return ""
