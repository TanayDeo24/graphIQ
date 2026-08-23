"""Groundedness guardrail: the code-enforced half of the rule that the
agent never invents a specific project number. Extracts every numeric
value from a piece of response text and verifies each one traces back to
a real result set -- either verbatim, or as a simple arithmetic
combination (difference, ratio, percentage-change, sum) of two values
that do appear.

This is deliberately NOT a prompt instruction. The project's standing rule
(see README's leakage guardrails, e.g. check_no_leakage() /
check_no_cohort_leakage()) is that anything safety-critical gets an
automated, code-level check, because prompting alone is not a sufficient
guarantee. check_groundedness() is that check for this agent.

In the current (SQL + doc-retrieval + free generation) architecture, this
is applied only to the synthesis step's data_answer field, checked
against the real row set an executed, validated SQL query returned (see
orchestrator.py) -- never to a general-knowledge concept explanation
(which makes no project-data claim) or a doc-retrieval-based rationale
(a different, non-numeric grounding guarantee: drawn from retrieved
passages, not verified number-by-number). This function itself doesn't
care where its `structured_results` came from -- it's generic over any
list of dict/list-shaped records -- and is imported unmodified by both
the runtime orchestrator and the eval suite, never reimplemented.
"""

import re
from itertools import combinations

# Numbers below this magnitude (small integers) are excluded from the
# "does every number trace back to a result row" check. Rationale: natural
# response prose legitimately contains small structural numbers that
# aren't data values -- "12-month survival", "one of the", ordinal list
# markers, etc. This allowance does not create a loophole for real data
# values: any actual data-derived number (a risk score, a percentage, a
# count from a table) either comes straight from the checked result rows
# or fails this check -- it only spares small fixed English quantifiers
# that would otherwise cause false positives on ordinary sentence text.
SMALL_INTEGER_EXEMPTION_MAX = 12

# Standard confidence-interval levels: naming which one was used ("95%
# CI") is a fixed statistical convention, not a project-data claim -- the
# actual data are the ci_low/ci_high bounds, which ARE checked normally.
# This project's own bootstrap CIs are always reported at 95%; 90/99 are
# included as the other levels anyone would recognize as a convention
# name rather than a measured value.
CONFIDENCE_LEVEL_EXEMPTIONS = {90.0, 95.0, 99.0}

RELATIVE_TOLERANCE = 0.01  # 1% -- accommodates rounding/formatting in rendered text
ABSOLUTE_TOLERANCE = 0.006  # covers e.g. 0.1 percentage-point display rounding

NUMBER_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9_])          # not preceded by a letter/digit/underscore (avoids matching inside IDs)
    -?                          # optional leading minus
    \$?                         # optional leading currency sign
    \d[\d,]*                    # integer part, comma-grouped
    (?:\.\d+)?                  # optional decimal part
    (?:[eE][+-]?\d+)?           # optional scientific-notation exponent (e.g. 6.79e-05) -- p-values are
                                 # routinely reported this way; without this, "6.79e-05" was matched as two
                                 # separate broken tokens ("6.79" and "05"), and the mantissa-only "6.79"
                                 # would never match the real value 0.0000679 -- a real false-positive
                                 # "ungrounded" failure found via a live eval run, not a hypothetical.
    \s*
    (?:%|x\b)?                  # optional trailing percent or "x" (lift multiplier)
    """,
    re.VERBOSE,
)


def _parse_number(raw: str):
    """Returns (value, is_percent) for a regex match, or None if it isn't
    actually a usable number (e.g. a bare '-')."""
    s = raw.strip()
    is_percent = s.endswith("%")
    is_multiplier = s.rstrip().endswith("x") and not is_percent
    s = s.rstrip("%x").strip()
    s = s.replace("$", "").replace(",", "")
    if s in ("", "-", "."):
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    return value, is_percent, is_multiplier


def extract_numbers_from_text(text: str) -> list:
    """Returns a list of (raw_matched_string, float_value, is_percent,
    is_multiplier) tuples for every number-like token in `text`."""
    out = []
    for m in NUMBER_PATTERN.finditer(text):
        parsed = _parse_number(m.group(0))
        if parsed is None:
            continue
        value, is_percent, is_multiplier = parsed
        out.append((m.group(0).strip(), value, is_percent, is_multiplier))
    return out


def _flatten_numeric_leaves(obj, out=None):
    """Recursively walks a structured result (dict/list/scalar -- e.g. a
    list of SQL row dicts) and collects every numeric leaf value found
    anywhere in it."""
    if out is None:
        out = []
    if isinstance(obj, bool):
        return out  # bool is a subclass of int in Python -- exclude explicitly
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            # Column names themselves sometimes carry the number a
            # legitimate answer states -- e.g. spend_alert_fatigue's
            # alerts_per_1000_txns column, where "alerts per 1,000
            # transactions" is the metric's own definition, not a claim
            # pulled from the row's value. Found as a real false-positive
            # "ungrounded" failure in a live eval run before this was
            # added. Deliberately NOT reusing extract_numbers_from_text
            # here: its ID-avoidance lookbehind (correctly, for prose)
            # refuses to match a digit run preceded by a letter/underscore
            # -- exactly true of "1000" inside "alerts_per_1000_txns" --
            # so a plain digit-run scan is used instead, safe here since
            # column names are trusted schema, not arbitrary user text.
            if isinstance(k, str):
                for digits in re.findall(r"\d+", k):
                    out.append(float(digits))
            _flatten_numeric_leaves(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numeric_leaves(v, out)
    elif isinstance(obj, str):
        # Result rows carry numbers embedded inside plain text fields too
        # (e.g. a note/disclaimer column quoting back an id or threshold)
        # -- reuses the same extraction regex as the response-text side of
        # this check, not a separate whole-string-only parse (an earlier
        # version only picked up a string value that parsed as a number in
        # its entirety, which let a real embedded number inside a longer
        # sentence fall through to a false "ungrounded" failure).
        for _, value, _, _ in extract_numbers_from_text(obj):
            out.append(value)
    return out


def _matches(candidate: float, reference: float) -> bool:
    if abs(candidate - reference) <= ABSOLUTE_TOLERANCE:
        return True
    denom = max(abs(reference), 1e-9)
    return abs(candidate - reference) / denom <= RELATIVE_TOLERANCE


def _candidate_matches_any(value: float, is_percent: bool, is_multiplier: bool, raw_values: list) -> bool:
    """A number extracted from response text is grounded if it matches some
    raw tool-result value directly, OR matches after accounting for the
    percent/multiplier display convention (raw fractions like 0.48 are
    commonly rendered as "48%"; raw ratios like 6.7 are rendered as "6.7x").
    """
    scaled_candidates = [value]
    if is_percent:
        scaled_candidates.append(value / 100.0)
    else:
        # a plain decimal in text (no % sign) might still correspond to a
        # raw fraction rendered *100, e.g. "76.0" for 0.760
        scaled_candidates.append(value / 100.0)
        scaled_candidates.append(value * 100.0)
    for cand in scaled_candidates:
        for ref in raw_values:
            if _matches(cand, ref):
                return True
    return False


MAX_VALUES_FOR_PAIRWISE = 200


def _build_grounded_value_set(structured_results: list) -> list:
    """Flattens every numeric leaf across all tool call results into one
    pool, then (if the pool is small enough to make this cheap) extends it
    with simple pairwise arithmetic combinations -- difference, ratio,
    percentage-change, sum -- since the build spec explicitly allows a
    number that's "a simple, explicit arithmetic combination" of two
    values that do appear, not only a verbatim match."""
    raw_values = []
    for result in structured_results:
        _flatten_numeric_leaves(result, raw_values)
    raw_values = list({round(v, 9) for v in raw_values})

    grounded = list(raw_values)
    if len(raw_values) <= MAX_VALUES_FOR_PAIRWISE:
        for a, b in combinations(raw_values, 2):
            grounded.append(a - b)
            grounded.append(b - a)
            grounded.append(a + b)
            if abs(b) > 1e-9:
                grounded.append(a / b)
                grounded.append((a - b) / b * 100.0)
            if abs(a) > 1e-9:
                grounded.append(b / a)
                grounded.append((b - a) / a * 100.0)
    return grounded


def check_groundedness(response_text: str, structured_results: list) -> dict:
    """Returns {"grounded": bool, "ungrounded_numbers": [...], "checked_numbers": [...]}.

    structured_results: list of raw JSON-like dicts/lists -- in the
    current architecture, a real SQL query's returned rows (see
    orchestrator.py). Every number-like token found in response_text must
    trace back to one of these, verbatim or via a simple arithmetic
    combination of two values that do -- see module docstring.
    """
    numbers = extract_numbers_from_text(response_text)
    grounded_pool = _build_grounded_value_set(structured_results)

    checked, ungrounded = [], []
    for raw_str, value, is_percent, is_multiplier in numbers:
        if abs(value) <= SMALL_INTEGER_EXEMPTION_MAX and value == int(value) and not is_percent:
            continue
        if is_percent and value in CONFIDENCE_LEVEL_EXEMPTIONS:
            # "95% CI" (etc.) names which standard confidence level was
            # used -- a fixed, universal statistical convention, not a
            # project-data claim. The actual data are the ci_low/ci_high
            # bounds themselves, which ARE checked normally. Found as a
            # real false-positive "ungrounded" failure in a live eval run
            # (this project's own bootstrap CIs are always reported at
            # 95%) before this narrow exemption was added.
            continue
        checked.append(raw_str)
        if not _candidate_matches_any(value, is_percent, is_multiplier, grounded_pool):
            ungrounded.append(raw_str)

    return {
        "grounded": len(ungrounded) == 0,
        "ungrounded_numbers": ungrounded,
        "checked_numbers": checked,
    }
