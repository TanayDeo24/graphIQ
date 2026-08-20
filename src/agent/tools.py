"""One tool function per agent capability. Each is a thin wrapper around an
*existing* FastAPI endpoint (called over real HTTP, exactly as the
dashboard calls it) -- no new computation, no new database query beyond
what src/api/routes/ already exposes. The one exception is
list_available_topics(), which is a static, no-DB-query fallback menu (see
its own docstring).

Every tool's return dict is the thing passed to groundedness.check_groundedness()
as ground truth, and the thing logged verbatim per principle 4 -- so tools
deliberately include any threshold constant a template might need to state
(e.g. the low-confidence event-count cutoff), reused from the exact
constant the rest of this project already uses, rather than letting a
template hardcode a number that wouldn't trace back to a tool call.
"""

import os

import requests

from src.models.attrition.evaluate import INTERACTION_HEATMAP_MIN_N, TOP_RISK_QUANTILE

API_BASE_URL = os.environ.get("AGENT_API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 15

# Same convention this project already documents for segment_calibration
# (see README: "Any segment with event_count below 10 is flagged... as
# low-confidence") -- reused here from the one place it's an actual
# constant in code, rather than a second hardcoded "10".
LOW_CONFIDENCE_EVENT_COUNT_THRESHOLD = INTERACTION_HEATMAP_MIN_N


class ToolError(Exception):
    """Raised for a genuine transport/server failure (not a 404/empty
    result, which each tool represents as a structured {"found": False}
    dict instead so the orchestrator can hand it to a refusal template)."""


def _get(path: str, params: dict = None) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise ToolError(f"Request to {path} failed: {e}") from e
    if resp.status_code == 404:
        return {"found": False, "not_found_detail": resp.json().get("detail", "not found")}
    if resp.status_code >= 400:
        raise ToolError(f"{path} returned HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def get_attrition_risk_score(employee_id: int) -> dict:
    """Look up an employee's baseline GBM attrition risk score, tenure
    band, and department. Use this for "what is X's risk score" / "is
    employee X high risk" type questions."""
    data = _get("/api/attrition/risk-scores", params={"employee_id": employee_id, "page_size": 1})
    if not data.get("results"):
        return {"found": False, "not_found_detail": f"No risk score found for employee_id={employee_id}."}
    row = data["results"][0]
    row["found"] = True
    row["top_risk_quantile"] = TOP_RISK_QUANTILE
    return row


def get_attrition_shap(employee_id: int) -> dict:
    """Look up the SHAP feature-driver breakdown for an employee's
    attrition risk score. Only available for the top-risk decile. Use this
    for "why is X at risk" / "what's driving X's risk score" questions,
    typically alongside get_attrition_risk_score."""
    data = _get(f"/api/attrition/shap/{employee_id}")
    if isinstance(data, dict) and data.get("found") is False:
        return data
    return {"found": True, "employee_id": employee_id, "shap_rows": data}


def get_segment_calibration(segment_dimension: str, segment_value: str) -> dict:
    """Look up calibration (predicted vs. observed 12-month survival) for
    one segment, e.g. segment_dimension="department", segment_value="Sales",
    or segment_dimension="tenure_band", segment_value="0-2". Valid
    dimensions: department, tenure_band, comp_band, department_x_tenure_band.
    Includes n_at_risk/event_count and this project's low-confidence
    threshold so the caller can judge reliability."""
    data = _get("/api/attrition/calibration")
    matches = [
        row for row in data
        if row.get("segment_dimension") == segment_dimension and row.get("segment_value") == segment_value
    ]
    if not matches:
        return {
            "found": False,
            "not_found_detail": f"No calibration segment for {segment_dimension}={segment_value!r}.",
        }
    row = dict(matches[0])
    row["found"] = True
    row["low_confidence_threshold_event_count"] = LOW_CONFIDENCE_EVENT_COUNT_THRESHOLD
    row["is_low_confidence"] = row["event_count"] < LOW_CONFIDENCE_EVENT_COUNT_THRESHOLD
    row["is_zero_events"] = row["event_count"] == 0
    return row


def get_detector_comparison() -> dict:
    """Full spend-anomaly detector comparison table: PR-AUC and
    lift-over-random for every detector (Isolation Forest, autoencoder,
    CUSUM, cohort CUSUM, ensemble) x anomaly type (point_spike, slow_drift,
    coordinated_pattern, overall). Use this for "which detector is best
    at X" / "how good is detector Y" questions."""
    data = _get("/api/spend/detector-comparison")
    return {"found": True, "rows": data}


def get_spend_transaction_explanation(transaction_id: int) -> dict:
    """Sub-signal decomposition for why one specific transaction was
    flagged as anomalous. Use this for "why was transaction X flagged"
    questions."""
    data = _get(f"/api/spend/transaction/{transaction_id}/explain")
    if isinstance(data, dict) and data.get("found") is False:
        return data
    return {"found": True, "transaction_id": transaction_id, "sub_signals": data}


def get_cross_component_quadrant(employee_id: int) -> dict:
    """Look up one employee's cross-component quadrant assignment (attrition
    risk vs. spend-anomaly signal), plus the overall bivariate and
    partial-correlation figures (partial correlation, controlling for
    department + monthly income, is this project's primary reported
    figure; the bivariate figure is kept for reference only)."""
    data = _get("/api/cross-component/quadrant")
    employees = data.get("employees", [])
    match = next((e for e in employees if e.get("employee_id") == employee_id), None)
    if match is None:
        return {"found": False, "not_found_detail": f"No cross-component record for employee_id={employee_id}."}
    return {
        "found": True,
        "employee": match,
        "summary": data.get("summary"),
    }


def get_lead_time_distribution() -> dict:
    """Attrition lead-time distribution: how many months before departure
    the model's predicted risk crosses the flagging threshold, for true
    positives. Use this for "how much warning does the model give"
    questions."""
    data = _get("/api/attrition/lead-time")
    return {"found": True, **data}


def get_gains_curve() -> dict:
    """Spend-anomaly dollar-weighted gains curve: what share of flagged
    anomalous dollar volume is captured at each alert-volume percentile.
    Use this for "how much value do we capture if we review the top N% of
    alerts" questions."""
    data = _get("/api/spend/gains-curve")
    return {"found": True, "points": data}


AVAILABLE_TOPICS = [
    {
        "topic": "attrition risk scores",
        "description": "An individual employee's baseline attrition risk score and tenure band.",
        "example_question": "What is employee 42's attrition risk score?",
    },
    {
        "topic": "attrition risk drivers (SHAP)",
        "description": "Which factors are driving a specific (top-risk-decile) employee's risk score.",
        "example_question": "Why is employee 42 flagged as high risk?",
    },
    {
        "topic": "attrition model calibration",
        "description": "Whether the model's predicted survival matches observed survival for a segment "
        "(department, tenure band, comp band, or a cross of department x tenure band), including sample "
        "size and a low-confidence flag.",
        "example_question": "How well-calibrated is the model for the Sales department?",
    },
    {
        "topic": "attrition lead time",
        "description": "How many months of advance warning the model gives before an actual departure.",
        "example_question": "How much lead time does the attrition model give before someone leaves?",
    },
    {
        "topic": "spend anomaly detector comparison",
        "description": "Which spend-anomaly detector (Isolation Forest, autoencoder, CUSUM, ensemble) "
        "performs best for a given anomaly type, with PR-AUC and lift-over-random.",
        "example_question": "Which detector is best at catching slow_drift spend anomalies?",
    },
    {
        "topic": "spend transaction explanations",
        "description": "Why one specific flagged transaction was flagged, broken down by sub-signal.",
        "example_question": "Why was transaction 12345 flagged?",
    },
    {
        "topic": "spend anomaly gains curve",
        "description": "How much anomalous dollar volume is captured by reviewing the top N% of alerts.",
        "example_question": "How much anomalous spend do we catch by reviewing the top 10% of alerts?",
    },
    {
        "topic": "cross-component (attrition x spend)",
        "description": "Whether an individual employee's attrition risk and spend-anomaly signal align "
        "into a quadrant, plus the overall correlation between the two across all employees.",
        "example_question": "Is there a relationship between attrition risk and spend anomalies?",
    },
]


def list_available_topics() -> dict:
    """Returns a short structured list of what this agent can answer,
    for use when a question doesn't map cleanly to any other tool. Static
    and does not query the database -- it describes the *tools*
    available, not their live data, so it needs no DB round trip."""
    return {"found": True, "topics": AVAILABLE_TOPICS}


# name -> callable, used by the orchestrator to dispatch Gemini's chosen
# function call and by the eval suite to enumerate every tool.
TOOL_REGISTRY = {
    "get_attrition_risk_score": get_attrition_risk_score,
    "get_attrition_shap": get_attrition_shap,
    "get_segment_calibration": get_segment_calibration,
    "get_detector_comparison": get_detector_comparison,
    "get_spend_transaction_explanation": get_spend_transaction_explanation,
    "get_cross_component_quadrant": get_cross_component_quadrant,
    "get_lead_time_distribution": get_lead_time_distribution,
    "get_gains_curve": get_gains_curve,
    "list_available_topics": list_available_topics,
}
