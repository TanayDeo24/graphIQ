"""Groundedness guardrail: the code-enforced half of principle 1 (the agent
never invents a number). Extracts every numeric value from a response's
final text and verifies each one traces back to a tool call's raw returned
JSON -- either verbatim, or as a simple arithmetic combination (difference,
ratio, percentage-change, sum) of two values that do appear.

This is deliberately NOT a prompt instruction. The project's standing rule
(see README's leakage guardrails, e.g. check_no_leakage() /
check_no_cohort_leakage()) is that anything safety-critical gets an
automated, code-level check, because prompting alone is not a sufficient
guarantee. check_groundedness() is that check for this agent, and it is
imported unmodified by both the runtime orchestrator (src/agent/orchestrator.py)
and the eval suite (src/agent/eval/) -- one implementation, two call sites,
never reimplemented.
"""

import re
from itertools import combinations

# Numbers below this magnitude (small integers) are excluded from the
# "does every number trace back to a tool result" check. Rationale: template
# text legitimately contains small structural numbers that are not data
# values -- "12-month survival", "top-quartile", ordinal list markers, etc.
# Anything a template needs to state as an actual *data-derived* number
# (thresholds like the low-confidence event-count cutoff, correlation
# thresholds, etc.) is required to come from a tool result's own returned
# fields (see tools.py, which bakes such constants into its return dicts)
# rather than being hardcoded in template prose -- so this allowance does
# not create a loophole for real data values, only for small fixed English
# quantifiers ("12-month", "one of the") that would otherwise cause false
# positives on template boilerplate that never claims to be data.
SMALL_INTEGER_EXEMPTION_MAX = 12

RELATIVE_TOLERANCE = 0.01  # 1% -- accommodates rounding/formatting in rendered text
ABSOLUTE_TOLERANCE = 0.006  # covers e.g. 0.1 percentage-point display rounding

NUMBER_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9_])          # not preceded by a letter/digit/underscore (avoids matching inside IDs)
    -?                          # optional leading minus
    \$?                         # optional leading currency sign
    \d[\d,]*                    # integer part, comma-grouped
    (?:\.\d+)?                  # optional decimal part
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
    """Recursively walks a tool result (dict/list/scalar) and collects every
    numeric leaf value found anywhere in it."""
    if out is None:
        out = []
    if isinstance(obj, bool):
        return out  # bool is a subclass of int in Python -- exclude explicitly
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numeric_leaves(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numeric_leaves(v, out)
    elif isinstance(obj, str):
        # Tool results carry numbers embedded inside prose too -- e.g. a
        # not-found tool result's own explanatory string
        # ("No risk score found for employee_id=999999.") is exactly the
        # kind of thing a refusal template's `reason` fill quotes back to
        # the user, and that embedded id must count as grounded. Reuses
        # the same extraction regex as the response-text side of this
        # check, not a separate whole-string-only parse (found via a live
        # test: whole-string parsing alone let this case fall through to
        # a false "ungrounded" failure).
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


def _build_grounded_value_set(tool_call_results: list) -> list:
    """Flattens every numeric leaf across all tool call results into one
    pool, then (if the pool is small enough to make this cheap) extends it
    with simple pairwise arithmetic combinations -- difference, ratio,
    percentage-change, sum -- since principle 1 explicitly allows "a simple,
    explicit arithmetic combination of two such values" as grounded."""
    raw_values = []
    for result in tool_call_results:
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


def check_groundedness(response_text: str, tool_call_results: list) -> dict:
    """Returns {"grounded": bool, "ungrounded_numbers": [...], "checked_numbers": [...]}.

    tool_call_results: list of raw JSON-like dicts/lists as returned by the
    tool functions actually called for this turn (see tools.py). Every
    number-like token found in response_text must trace back to one of
    these, verbatim or via a simple arithmetic combination of two values
    that do -- see module docstring.
    """
    numbers = extract_numbers_from_text(response_text)
    grounded_pool = _build_grounded_value_set(tool_call_results)

    checked, ungrounded = [], []
    for raw_str, value, is_percent, is_multiplier in numbers:
        if abs(value) <= SMALL_INTEGER_EXEMPTION_MAX and value == int(value) and not is_percent:
            continue
        checked.append(raw_str)
        if not _candidate_matches_any(value, is_percent, is_multiplier, grounded_pool):
            ungrounded.append(raw_str)

    return {
        "grounded": len(ungrounded) == 0,
        "ungrounded_numbers": ungrounded,
        "checked_numbers": checked,
    }
