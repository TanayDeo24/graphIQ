"""The agent loop: question -> Gemini picks tool(s) -> orchestrator executes
them (real HTTP calls to existing endpoints) and logs every raw result ->
orchestrator (code, not the model) deterministically selects which fixed
template applies and reads its fill values directly out of the tool
results -> template is rendered -> an optional one-sentence,
number-free wrapper may be added -> the whole thing is checked by
groundedness.check_groundedness() before anything is returned.

Design rationale for "orchestrator picks the template, not a second model
call": the build spec's principle 1 says grounding is enforced in code,
"not only by prompting," because prompting alone isn't a sufficient
guarantee anywhere else in this project (see check_no_leakage(),
check_no_cohort_leakage()). Letting a second free-form model call choose a
template key AND transcribe numeric fill values reopens exactly the
failure mode principle 1 exists to close: a transcription slip that
doesn't match the tool result. Gemini's function-calling choice (which
tool(s), with what arguments) already carries the intent signal a template
choice would encode -- there's a natural, unambiguous mapping from "which
tool(s) were called" to "which template applies" for every capability this
agent has, including the deterministic low-confidence override required by
principle 3. So template selection and fill extraction are both
pure-Python here, and the model's only remaining creative latitude is the
strictly-validated, optional wrapper sentence. This is a stricter reading
of principle 2 than the minimum, in service of principle 1.
"""

import re
import time
import uuid

from src.agent import gemini_client
from src.agent.groundedness import check_groundedness
from src.agent.templates import render_shap_section, render_template
from src.agent.tools import TOOL_REGISTRY

ANOMALY_TYPE_KEYWORDS = {
    "point_spike": ["point spike", "point_spike", "point-spike"],
    "slow_drift": ["slow drift", "slow_drift", "slow-drift"],
    "coordinated_pattern": ["coordinated pattern", "coordinated_pattern", "coordinated-pattern"],
}

PCT_PATTERN = re.compile(r"top\s+(\d{1,3})\s*%")


def _detect_anomaly_type(question: str) -> str:
    q = question.lower()
    for anomaly_type, keywords in ANOMALY_TYPE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return anomaly_type
    return "overall"


def _p_value_display(p_value: float) -> str:
    return "< 0.001" if p_value is not None and p_value < 0.001 else f"= {p_value:.3f}"


def _topics_list_str() -> str:
    topics = TOOL_REGISTRY["list_available_topics"]()["topics"]
    return "; ".join(t["topic"] for t in topics)


def _sources_for(tool_name: str) -> str:
    endpoint_by_tool = {
        "get_attrition_risk_score": "GET /api/attrition/risk-scores",
        "get_attrition_shap": "GET /api/attrition/shap/{employee_id}",
        "get_segment_calibration": "GET /api/attrition/calibration",
        "get_detector_comparison": "GET /api/spend/detector-comparison",
        "get_spend_transaction_explanation": "GET /api/spend/transaction/{id}/explain",
        "get_cross_component_quadrant": "GET /api/cross-component/quadrant",
        "get_lead_time_distribution": "GET /api/attrition/lead-time",
        "get_gains_curve": "GET /api/spend/gains-curve",
        "list_available_topics": "(static topic list, no endpoint)",
    }
    return endpoint_by_tool.get(tool_name, tool_name)


def _select_template_and_fills(question: str, executed: list):
    """executed: [{"tool": name, "args": {...}, "result": {...}}, ...].
    Returns (template_key, fills). See module docstring for why this is
    deterministic code, not a model call."""
    by_tool = {e["tool"]: e["result"] for e in executed}

    not_found = [e for e in executed if isinstance(e["result"], dict) and e["result"].get("found") is False]
    if not_found:
        reasons = "; ".join(e["result"].get("not_found_detail", "not found") for e in not_found)
        return "refusal_not_found", {"reason": reasons}

    if "get_segment_calibration" in by_tool:
        row = by_tool["get_segment_calibration"]
        common = {
            "segment_dimension": row["segment_dimension"],
            "segment_value": row["segment_value"],
            "n_at_risk": row["n_at_risk"],
            "event_count": row["event_count"],
            "predicted_survival": row["predicted_survival"],
            "observed_survival": row["observed_survival"],
            "calibration_error": row["calibration_error"],
            "low_confidence_threshold_event_count": row["low_confidence_threshold_event_count"],
        }
        if row["is_zero_events"]:
            return "segment_calibration_zero_events", {
                "segment_dimension": row["segment_dimension"],
                "segment_value": row["segment_value"],
                "n_at_risk": row["n_at_risk"],
                "event_count": row["event_count"],
                "observed_survival": row["observed_survival"],
            }
        if row["is_low_confidence"]:
            return "segment_calibration_low_confidence", common
        return "segment_calibration_explanation", common

    if "get_attrition_risk_score" in by_tool or "get_attrition_shap" in by_tool:
        risk_row = by_tool.get("get_attrition_risk_score")
        shap_result = by_tool.get("get_attrition_shap")
        if risk_row is None:
            # SHAP was requested without a risk-score lookup; fall back to
            # fetching it so the template's required fields are available.
            risk_row = TOOL_REGISTRY["get_attrition_risk_score"](employee_id=shap_result["employee_id"])
            executed.append({"tool": "get_attrition_risk_score", "args": {}, "result": risk_row})
        shap_rows = shap_result["shap_rows"] if shap_result and shap_result.get("found") else []
        return "attrition_risk_explanation", {
            "employee_id": risk_row["employee_id"],
            "risk_score": risk_row["gbm_risk_score"],
            "tenure_band": risk_row["tenure_band"],
            "top_quartile_phrase": "is above" if risk_row["is_top_risk_quartile"] else "is not above",
            "shap_section": render_shap_section(shap_rows),
        }

    if "get_detector_comparison" in by_tool:
        rows = by_tool["get_detector_comparison"]["rows"]
        anomaly_type = _detect_anomaly_type(question)
        pr_auc_key, lift_key = f"{anomaly_type}_pr_auc", f"{anomaly_type}_lift"
        scored = [r for r in rows if r.get(pr_auc_key) is not None]
        best = max(scored, key=lambda r: r[pr_auc_key])
        return "detector_comparison_explanation", {
            "anomaly_type": anomaly_type,
            "best_detector": best["detector"],
            "pr_auc": best[pr_auc_key],
            "lift": best[lift_key],
        }

    if "get_spend_transaction_explanation" in by_tool:
        result = by_tool["get_spend_transaction_explanation"]
        sub_signals = sorted(result["sub_signals"], key=lambda r: r["contribution"], reverse=True)
        top = sub_signals[0]
        others = ", ".join(f"{r['sub_signal']} ({r['contribution']:.3f})" for r in sub_signals[1:]) or "none"
        return "spend_transaction_explanation", {
            "transaction_id": result["transaction_id"],
            "top_sub_signal": top["sub_signal"],
            "contribution": top["contribution"],
            "other_signals_list": others,
        }

    if "get_cross_component_quadrant" in by_tool:
        result = by_tool["get_cross_component_quadrant"]
        emp, summary = result["employee"], result["summary"]
        return "cross_component_explanation", {
            "employee_id": emp["employee_id"],
            "risk_score": emp["gbm_risk_score"],
            "spend_anomaly_score": emp["spend_anomaly_score"],
            "quadrant": emp["quadrant"],
            "partial_correlation": summary["partial_spearman_correlation"],
            "partial_p_value_display": _p_value_display(summary["partial_p_value"]),
            "bivariate_correlation": summary["spearman_correlation"],
            "bivariate_p_value_display": _p_value_display(summary["p_value"]),
        }

    if "get_lead_time_distribution" in by_tool:
        summary_rows = {r["statistic"]: r for r in by_tool["get_lead_time_distribution"]["summary"]}
        mean_row, median_row = summary_rows.get("mean"), summary_rows.get("median")
        return "lead_time_explanation", {
            "mean_lead_time": mean_row["point_estimate"],
            "mean_ci_low": mean_row["ci_low"],
            "mean_ci_high": mean_row["ci_high"],
            "n_true_positives": mean_row["n_true_positives"],
            "median_lead_time": median_row["point_estimate"],
        }

    if "get_gains_curve" in by_tool:
        points = by_tool["get_gains_curve"]["points"]
        m = PCT_PATTERN.search(question.lower())
        target = int(m.group(1)) / 100.0 if m else 0.10
        closest = min(points, key=lambda p: abs(p["pct_alerts_raised"] - target))
        return "gains_curve_explanation", {
            "pct_alerts_raised": closest["pct_alerts_raised"],
            "pct_dollar_volume_captured": closest["pct_dollar_volume_captured"],
        }

    # list_available_topics, or nothing matched -- out of scope
    return "refusal_out_of_scope", {"topics_list": _topics_list_str()}


def handle_question(message: str, page_context: str = None, conversation_history: list = None) -> dict:
    """Returns {"response": str, "tool_calls_made": [...], "template_used": str,
    "sources": [...], "log": {...}}. The `log` dict is the full record
    required by principle 4 (question, tool calls, raw results, final
    response) -- the API layer persists it."""
    log = {
        "id": str(uuid.uuid4()),
        "question": message,
        "page_context": page_context,
        "started_at": time.time(),
        "tool_calls": [],
        "template_used": None,
        "response": None,
        "grounded": None,
        "groundedness_detail": None,
    }

    try:
        requested_calls = gemini_client.request_tool_calls(message, page_context, conversation_history)
    except RuntimeError as e:
        # missing GEMINI_API_KEY etc. -- not a data problem, surface plainly
        log["response"] = str(e)
        log["template_used"] = None
        log["elapsed_seconds"] = time.time() - log["started_at"]
        return {"response": str(e), "tool_calls_made": [], "template_used": None, "sources": [], "log": log}

    executed = []
    for call in requested_calls:
        fn = TOOL_REGISTRY.get(call["name"])
        if fn is None:
            continue  # model named a tool that doesn't exist; skip it, don't execute
        try:
            result = fn(**call["args"])
        except Exception as e:
            result = {"found": False, "not_found_detail": f"Tool error: {e}"}
        executed.append({"tool": call["name"], "args": call["args"], "result": result})
        log["tool_calls"].append({"tool": call["name"], "args": call["args"], "raw_result": result})

    if not executed:
        template_key, fills = "refusal_out_of_scope", {"topics_list": _topics_list_str()}
    else:
        template_key, fills = _select_template_and_fills(message, executed)

    rendered = render_template(template_key, fills)
    tool_results_raw = [e["result"] for e in executed]

    wrapper = gemini_client.generate_wrapper_sentence(message) if executed and not template_key.startswith("refusal") else ""
    candidate_text = f"{wrapper} {rendered}".strip() if wrapper else rendered

    check = check_groundedness(candidate_text, tool_results_raw)
    if not check["grounded"] and wrapper:
        # the wrapper (the only model-authored free text) is the likely
        # culprit -- drop it and re-check the bare template render, which
        # is built entirely from code-extracted tool values.
        check_no_wrapper = check_groundedness(rendered, tool_results_raw)
        if check_no_wrapper["grounded"]:
            candidate_text, check = rendered, check_no_wrapper

    if not check["grounded"]:
        final_text = render_template("refusal_groundedness_failed", {"topics_list": _topics_list_str()})
        final_check = check_groundedness(final_text, tool_results_raw)
        template_key = "refusal_groundedness_failed"
    else:
        final_text, final_check = candidate_text, check

    sources = sorted({_sources_for(e["tool"]) for e in executed}) if executed else []

    log["template_used"] = template_key
    log["response"] = final_text
    log["grounded"] = final_check["grounded"]
    log["groundedness_detail"] = final_check
    log["elapsed_seconds"] = time.time() - log["started_at"]

    return {
        "response": final_text,
        "tool_calls_made": [{"tool": e["tool"], "args": e["args"]} for e in executed],
        "template_used": template_key,
        "sources": sources,
        "log": log,
    }
