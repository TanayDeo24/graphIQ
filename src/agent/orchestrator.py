"""The agent loop, rebuilt around three distinct mechanisms rather than a
single rigid path (see README's Explainability agent section for the
full design writeup). All model calls go through
src/agent/llm_backend.py, which switches between a local MLX backend and
Cloudflare Workers AI via LLM_BACKEND -- this module never talks to either
directly or branches on which is active.

1. Routing: one model call classifies the question against three
   categories -- "concept" (general knowledge, free generation, no
   grounding needed), "sql" (a specific project number/table question,
   needs a live query), "rationale" (a project-specific "why" question,
   needs retrieval from this project's own methodology docs). A question
   can need more than one category at once -- routing returns a list, not
   a single choice.
2. Gathering: sql_agent.generate_and_execute_sql() and/or
   doc_retrieval.retrieve_relevant_chunks() run for whichever categories
   were routed.
3. Synthesis: one model call writes the final answer as a JSON object
   with a SEPARATE string per category that applies (concept_answer /
   data_answer / rationale_answer) rather than one blended blob of prose.
   This separation is deliberate, not incidental: it's what lets the
   groundedness check below apply precisely to the data_answer (checked
   against real SQL rows) without false-positiving on an ordinary number
   that legitimately appears in a free-form concept explanation (e.g.
   "concordance ranges from 0 to 1") -- exactly the failure mode a single
   whole-response regex scan would risk on a combined question like
   "explain what GBM risk score is and summarize that table."
4. Groundedness: only ever applied to data_answer, and only when a SQL
   query actually executed this turn -- reusing groundedness.py's
   check_groundedness() unmodified, now scoped correctly (principle 3).
   If it fails, the whole turn is replaced with an honest refusal, not a
   partially-patched answer.

The parts are concatenated into one final response string for display,
but the model is explicitly instructed (and this module never itself
inserts) any tool name, SQL text, or other internal-state debug text into
that prose -- sourcing is returned as separate structured fields
(sql_executed, doc_chunks_used, mechanisms_used) for the API/UI layer,
never inline in the answer text (principle 4).
"""

import json
import re
import time
import uuid

from src.agent import doc_retrieval, llm_backend, sql_agent
from src.agent.groundedness import check_groundedness

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> str:
    """Smaller/local models (unlike Gemini's native JSON response mode,
    used by the first version of this agent) are less reliable about
    emitting ONLY a JSON object despite being told to -- markdown code
    fences and a stray sentence before/after are both routine. Strips a
    ```json ... ``` fence if present, then falls back to grabbing the
    first {...} block via regex, so a normal json.loads() downstream has
    a fighting chance instead of failing on decoration around otherwise-
    valid JSON."""
    t = (raw or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    match = _JSON_OBJECT_RE.search(t)
    return match.group(0) if match else t


def _parse_json_lenient(raw: str) -> dict:
    """json.loads(..., strict=False) allows literal control characters
    (unescaped newlines, tabs) inside string values instead of raising --
    found live and necessary: synthesis embeds long, multi-paragraph
    rationale_answer text (drawn from README passages) inside a JSON
    string, and a smaller model reproducing that text is prone to leaving
    raw newlines in place of the \\n escape strict JSON requires. Standard-
    strict parsing failed on exactly this ("Expecting ',' delimiter")
    before this was added; the non-strict mode is a documented stdlib
    option for exactly this situation, not a hand-rolled workaround."""
    return json.loads(_extract_json(raw), strict=False)


VALID_CATEGORIES = {"concept", "sql", "rationale"}

ROUTING_SYSTEM_INSTRUCTION = """You are the routing step of GraphIQ's explainability agent, a workforce-analytics demo.

Given the user's question (and optionally which dashboard page they're viewing, and recent conversation),
decide which of these three mechanisms are needed to answer it. A question can need more than one -- do
not force a single choice when the question genuinely needs two or three combined.

- "concept": the question asks about a general concept, term, or definition (e.g. "what is a SHAP value",
  "what does concordance index mean", "what is CUSUM") that can be answered from general knowledge, with
  no project-specific data or project-specific rationale needed.
- "sql": the question asks for a specific number, employee record, table summary, comparison, or any other
  concrete project data point -- needs a live query against this project's own result tables.
- "rationale": the question asks WHY something in this project is true, or asks for this project's own
  methodology reasoning (e.g. "why do you trust the partial correlation over the bivariate one", "why did
  the CUSUM ranking change", "why is the GBM within-tenure-band concordance the honest headline number") --
  needs retrieval from this project's own methodology write-ups, not general knowledge.

Example: "explain what GBM risk score is and summarize that table" needs BOTH "concept" (what GBM risk
score is) AND "sql" (summarizing the actual table) -- return both, not just one.

Specific routing rules -- these override your general judgment when a question matches one of these shapes:

- MITIGATION/ACTION questions ("how do we fix/reduce/mitigate this", "what should we do about employee X's
  risk", "how do I lower this") -- ALWAYS route "sql", never "rationale" alone. For attrition, the SQL must
  query the attrition_sensitivity table (the counterfactual sensitivity data) for the relevant
  employee/segment. For spend, it must query spend_robustness or spend_eval_metrics (real detector
  performance data), never a made-up recommendation. Do not answer a mitigation question from general
  knowledge or from a retrieved passage alone -- it needs the real per-employee/per-detector numbers.
- TRUST/ACCURACY questions ("can I trust this", "how accurate is it", "is this model reliable") -- route
  "sql" (query attrition_model_metrics for metric_name = 'within_tenure_band_concordance', NEVER
  'concordance_index' alone -- the within-tenure-band figure is this project's own honest headline number,
  the raw concordance is known to be inflated by a structural confound) AND "rationale" (to explain why that
  specific metric, not the raw one, is the trustworthy one to cite).
- GENERALIZATION questions that ask to extend a single-entity or correlational finding into a population-
  wide or causal claim ("would raises reduce attrition fleet-wide", "is employee X leaving BECAUSE OF Y") --
  route "rationale" (the correlational-not-causal framing lives in the docs) and do NOT route "concept" for
  these -- they need the project's own explicit scope disclaimer, not a general explanation.
- RANKING/AGGREGATE questions ("who are my highest-risk employees", "which department needs attention") --
  route "sql". The query needs ORDER BY + LIMIT (for a ranked list) or GROUP BY (for a per-department/
  per-segment aggregate) -- these are structurally different from a single WHERE employee_id = X lookup.
- COMBINED/TRIAGE questions ("what should I look at today", "give me today's priorities") -- route "sql"
  (pulling from both attrition_risk_scores/cross_component_quadrant AND spend_anomaly_scores/
  spend_dollar_treemap as relevant) to reflect that this project's two components share one schema.
- TIME-HORIZON POPULATION questions ("how many employees will likely leave this quarter/this month") --
  route "sql", but be aware attrition_risk_scores only has gbm_predicted_survival_12m (a fixed 12-month
  horizon) -- there is no stored shorter-horizon (e.g. quarterly) prediction. A query can honestly answer
  a 12-month-horizon version of this question (e.g. counting employees below a survival threshold); a
  genuinely shorter-horizon version cannot be honestly answered from this schema at all -- in that case the
  SQL step will fail to find real supporting data, and the answer should say so plainly rather than
  approximate.

Respond with ONLY a JSON object, nothing else -- no markdown code fences, no explanation before or after it:
{"categories": ["concept", "sql", "rationale"]} -- include every category that applies (1 to 3 of them),
omit any that don't apply. Your entire response must be parseable by a JSON parser as-is.
"""

SYNTHESIS_SYSTEM_INSTRUCTION = """You are the answer-writing step of GraphIQ's explainability agent, a workforce-analytics demo.

Write the final answer to the user's question as a JSON object with up to three string fields --
concept_answer, data_answer, rationale_answer. The "Categories that were routed" line below tells you
EXACTLY which fields to write content for -- set every field NOT in that list to the JSON value null
immediately, with NO text at all, even if you technically know something about it. Do not write content
for an unrouted field "just in case" -- this wastes your output budget and risks truncating a field that
DOES need content. Follow these rules exactly for the field(s) you do write:

1. concept_answer: a general explanation of a term or concept, composed freely from your own knowledge.
   Never state a specific number here that's supposed to be this project's own result (general
   illustrative numbers about how a concept normally works, e.g. "concordance ranges from 0 to 1", are
   fine here since they're not a claim about this project's data).
2. data_answer: MUST be built only from the SQL RESULTS provided below. Never state a specific project
   number that doesn't appear in those SQL results. If the SQL query failed or returned no rows, say so
   plainly and naturally (e.g. "I wasn't able to retrieve that") instead of guessing a number. If any row
   has a low-confidence signal (event_count < 10, an event_count of exactly 0, or a low_confidence flag),
   you MUST hedge explicitly in this text and say why the number shouldn't be taken at face value -- never
   report it as if it were reliable.
3. rationale_answer: MUST be based only on the RETRIEVED DOC PASSAGES provided below. Do not invent a
   rationale that isn't supported by those passages. If no passages were retrieved, say plainly that you
   don't have a documented rationale for that, instead of guessing one.
   - If the question asks to generalize a single-employee or correlational finding into a population-wide
     or causal claim (e.g. "would raises reduce attrition fleet-wide", "is employee X leaving BECAUSE OF
     Y"), you MUST explicitly decline to make that leap and restate that the underlying finding is
     correlational/individual-scope only -- do not answer the generalized question as asked.
   - If the question asks whether the model can be trusted/is accurate, you MUST name the specific
     within-tenure-band concordance figure (never the raw overall concordance alone) as the honest number,
     and say why the raw figure is not the one to trust at face value.
4. If the SQL results represent a mitigation/action recommendation (from attrition_sensitivity or
   spend_robustness/spend_eval_metrics), state the real number and add: this is predicted sensitivity from
   a correlational model, not a causal guarantee that taking this action produces this exact effect.
5. Never mention tool names, SQL, "the database", "retrieval", table names, or any other internal system
   state in any of these fields -- write as a natural, plain sentence a person would say. Sourcing is shown
   separately in the UI, never inline in your text.
6. Keep each field concise -- a few sentences, not an essay, unless the question genuinely needs a
   table-like summary of several rows.

Respond with ONLY the JSON object, nothing else -- no markdown code fences, no explanation before or
after it. Your entire response must be parseable by a JSON parser as-is.
"""


def _history_to_messages(conversation_history: list) -> list:
    """conversation_history: [{"role": "user"|"model", "text": "..."}] ->
    the OpenAI-style {"role": "user"|"assistant", "content": "..."} list
    both backends' generate() accept when `prompt` is a list -- backend-
    agnostic, unlike the removed Gemini-specific Content-object builder
    this replaced."""
    messages = []
    for turn in conversation_history or []:
        role = "user" if turn.get("role") == "user" else "assistant"
        text_val = turn.get("text", "")
        if text_val:
            messages.append({"role": role, "content": text_val})
    return messages


def _route_question(question: str, page_context: str, conversation_history: list) -> list:
    messages = _history_to_messages(conversation_history)
    prompt_lines = [f"Question: {question}"]
    if page_context:
        prompt_lines.append(f"Currently viewing page: {page_context}")
    prompt_text = "\n".join(prompt_lines)
    prompt = messages + [{"role": "user", "content": prompt_text}] if messages else prompt_text

    # One retry on parse failure: found live that a smaller free-tier
    # model occasionally produces malformed/truncated JSON on a given
    # call and clean JSON on an immediate retry of the identical prompt --
    # cheap insurance against transient generation noise before falling
    # back to the safe default.
    for _ in range(2):
        try:
            raw = llm_backend.generate(prompt, system_instruction=ROUTING_SYSTEM_INSTRUCTION, max_tokens=200)
            parsed = _parse_json_lenient(raw)
            categories = [c for c in parsed.get("categories", []) if c in VALID_CATEGORIES]
            if categories:
                return categories
        except Exception:
            continue
    # Routing failed to parse or returned nothing usable -- fail toward
    # attempting everything rather than silently answering less than the
    # question needs; gathering steps that turn out irrelevant just come
    # back empty/unused, which is cheap, whereas skipping a category the
    # question actually needed risks an unhelpful or under-grounded answer.
    return ["concept", "sql", "rationale"]


def _synthesize(question: str, categories: list, sql_result: dict, doc_chunks: list) -> dict:
    parts = [f"User question: {question}", f"Categories that were routed: {categories}"]
    if "sql" in categories:
        if sql_result and sql_result["success"]:
            parts.append(f"SQL RESULTS (JSON rows):\n{json.dumps(sql_result['rows'], default=str)}")
        else:
            reason = sql_result["error"] if sql_result else "no query was executed"
            parts.append(f"SQL query did not succeed this turn ({reason}). No project data is available to answer the data part of this question.")
    if "rationale" in categories:
        if doc_chunks:
            passages = "\n\n".join(f"[{c['section_path']}]\n{c['text']}" for c in doc_chunks)
            parts.append(f"RETRIEVED DOC PASSAGES:\n{passages}")
        else:
            parts.append("No relevant doc passages were retrieved this turn. No documented rationale is available.")
    prompt = "\n\n".join(parts)

    # One retry on parse failure -- same rationale as _route_question's
    # retry: transient truncation/malformed JSON from a smaller free-tier
    # model is often not reproducible on an identical immediate retry.
    last_error = None
    for _ in range(2):
        try:
            raw = llm_backend.generate(prompt, system_instruction=SYNTHESIS_SYSTEM_INSTRUCTION, max_tokens=1400)
            parsed = _parse_json_lenient(raw)
            # Enforced in code, not just by prompting: a field is kept only if
            # its category was actually routed, regardless of what the model
            # filled in. Smaller/local models (unlike the stronger model this
            # was first built against) were found live to sometimes populate
            # concept_answer with a generic explanation even when only "sql"
            # was routed and the SQL failed -- e.g. answering "what is an
            # attrition risk score" instead of leaving the field null, masking
            # the fact that no real employee data was ever retrieved. This
            # mirrors the project's standing rule of never trusting a model's
            # self-restraint alone for something that changes what's shown.
            # A smaller local model was also found live to sometimes emit a
            # bare JSON number (e.g. `"data_answer": 2.482`) instead of a
            # string when asked to directly state a figure -- coerced to str
            # here rather than crashing downstream string-formatting code on
            # a real, meaningful (just mistyped) piece of content.
            def _as_text(value):
                if value is None or isinstance(value, str):
                    return value
                return str(value)
            return {
                "concept_answer": _as_text(parsed.get("concept_answer")) if "concept" in categories else None,
                "data_answer": _as_text(parsed.get("data_answer")) if "sql" in categories else None,
                "rationale_answer": _as_text(parsed.get("rationale_answer")) if "rationale" in categories else None,
            }
        except Exception as e:
            last_error = e
            continue
    return {"concept_answer": None, "data_answer": None, "rationale_answer": None, "_synthesis_error": str(last_error)}


NATURAL_REFUSAL_FALLBACK = (
    "I couldn't put together a reliable answer to that -- either I couldn't retrieve the right project "
    "data, or the numbers I generated didn't check out against what was actually retrieved. Rather than "
    "guess, I'd rather tell you plainly that I don't have a confident answer here."
)


def handle_question(message: str, page_context: str = None, conversation_history: list = None) -> dict:
    """Returns {"response": str, "sql_executed": str|None, "doc_chunks_used": list|None,
    "mechanisms_used": list, "log": {...}}."""
    log = {
        "id": str(uuid.uuid4()),
        "question": message,
        "page_context": page_context,
        "started_at": time.time(),
    }

    try:
        categories = _route_question(message, page_context, conversation_history)
    except RuntimeError as e:
        # missing/misconfigured backend credentials (e.g. CLOUDFLARE_API_TOKEN) etc.
        log["response"] = str(e)
        log["elapsed_seconds"] = time.time() - log["started_at"]
        return {"response": str(e), "sql_executed": None, "doc_chunks_used": None, "mechanisms_used": [], "log": log}

    log["categories_routed"] = categories

    sql_result = sql_agent.generate_and_execute_sql(message) if "sql" in categories else None
    doc_chunks = doc_retrieval.retrieve_relevant_chunks(message) if "rationale" in categories else None

    log["sql"] = sql_result
    log["doc_chunks"] = doc_chunks

    synthesis = _synthesize(message, categories, sql_result, doc_chunks)
    log["synthesis"] = synthesis

    parts = [synthesis.get("concept_answer"), synthesis.get("data_answer"), synthesis.get("rationale_answer")]
    response_text = " ".join(p.strip() for p in parts if p and p.strip())

    grounded = True
    ungrounded_numbers = []
    data_answer = synthesis.get("data_answer")
    if data_answer and sql_result and sql_result["success"]:
        # Groundedness is scoped ONLY to the data_answer text, checked
        # against the real SQL rows from this turn -- never applied to
        # concept_answer (general knowledge, no grounding claim) or
        # rationale_answer (a doc-retrieval guarantee, not a numeric one).
        # The executed SQL text itself is included as a second source:
        # an id used only as a WHERE-clause filter (e.g. "employee_id =
        # 22", echoing the user's own question back) legitimately isn't
        # present in any result *row* if it wasn't also selected as a
        # column -- but it's still a real, verifiable part of the actual
        # executed, validator-approved query, not an invented number.
        # Found as a real false-positive "ungrounded" failure in a live
        # eval run before this was added.
        check = check_groundedness(data_answer, [sql_result["rows"], sql_result["sql"]])
        grounded = check["grounded"]
        ungrounded_numbers = check["ungrounded_numbers"]

    if not response_text.strip() or not grounded:
        response_text = NATURAL_REFUSAL_FALLBACK
        mechanisms_used = []
        sql_executed_out = None
        doc_chunks_out = None
        sources = []
    else:
        mechanisms_used = []
        sources = []
        if synthesis.get("concept_answer"):
            mechanisms_used.append("concept")
            sources.append("general knowledge, not project-specific")
        if synthesis.get("data_answer") and sql_result and sql_result["success"]:
            mechanisms_used.append("sql")
            tables = sql_result.get("tables_used") or []
            table_label = ", ".join(tables) if tables else "project data"
            sources.append(f"live query: {table_label} table{'s' if len(tables) != 1 else ''}")
        if synthesis.get("rationale_answer") and doc_chunks:
            mechanisms_used.append("rationale")
            sources.append("project methodology docs (" + "; ".join(c["section_path"] for c in doc_chunks[:2]) + ")")
        sql_executed_out = sql_result["sql"] if (sql_result and sql_result["success"]) else None
        doc_chunks_out = [{"section_path": c["section_path"], "score": c["score"]} for c in doc_chunks] if doc_chunks else None

    log["response"] = response_text
    log["grounded"] = grounded
    log["ungrounded_numbers"] = ungrounded_numbers
    log["mechanisms_used"] = mechanisms_used
    log["elapsed_seconds"] = time.time() - log["started_at"]

    return {
        "response": response_text,
        "sql_executed": sql_executed_out,
        "doc_chunks_used": doc_chunks_out,
        "mechanisms_used": mechanisms_used,
        "sources": sources,
        "log": log,
    }
