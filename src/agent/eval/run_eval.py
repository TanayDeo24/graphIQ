"""Runs all four eval categories required by the build spec and writes
results to data/generated/agent_eval/. Run as: python -m src.agent.eval.run_eval

Every category is reported honestly, including confusion breakdowns and
failure cases -- this eval is not tuned to look good, per the project's
standing rule (see README's "Reported as-is, not tuned toward a
better-looking number" language elsewhere in this project).
"""

import json
import os
import statistics
import time
from collections import Counter, defaultdict

from src.agent import gemini_client, orchestrator
from src.agent.eval.datasets import GROUNDEDNESS_QUESTIONS, REFUSAL_CASES, TOOL_SELECTION_CASES
from src.agent.groundedness import check_groundedness

OUT_DIR = "data/generated/agent_eval"


def _write_json(name: str, obj) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------------------------------------------------------------------
# 1a. Groundedness negative control -- demonstrates the guardrail actually
# catches a fabricated number, not just that real answers happen to pass.
# Uses a real tool result (no live Gemini call needed for this one) with a
# deliberately invented statistic spliced into otherwise-real response
# text, exactly the failure mode principle 1 exists to catch.
# ---------------------------------------------------------------------
def run_groundedness_negative_control() -> dict:
    from src.agent.tools import get_attrition_risk_score

    real_result = get_attrition_risk_score(employee_id=4)
    fabricated_text = (
        f"Employee 4's attrition risk score is {real_result['gbm_risk_score']:.3f}, and there is a "
        "92.7% chance they will leave within 6 months."
    )
    check = check_groundedness(fabricated_text, [real_result])
    outcome = {
        "fabricated_response_text": fabricated_text,
        "tool_result_used": real_result,
        "grounded": check["grounded"],
        "ungrounded_numbers_caught": check["ungrounded_numbers"],
        "guardrail_worked": check["grounded"] is False and "92.7%" in check["ungrounded_numbers"],
    }
    _write_json("groundedness_negative_control.json", outcome)
    return outcome


# ---------------------------------------------------------------------
# 1. Groundedness
# ---------------------------------------------------------------------
def run_groundedness_eval() -> dict:
    rows = []
    latencies = []
    for question in GROUNDEDNESS_QUESTIONS:
        t0 = time.time()
        result = orchestrator.handle_question(question, page_context=None, conversation_history=[])
        elapsed = time.time() - t0
        latencies.append(elapsed)

        raw_results = [tc["raw_result"] for tc in result["log"]["tool_calls"]]
        # Re-apply check_groundedness independently here -- same function
        # the orchestrator already used to decide whether to return this
        # response at all, per the spec's requirement that the eval suite
        # reuse it unmodified rather than reimplementing the check.
        check = check_groundedness(result["response"], raw_results)
        rows.append(
            {
                "question": question,
                "template_used": result["template_used"],
                "tool_calls_made": [tc["tool"] for tc in result["tool_calls_made"]],
                "grounded": check["grounded"],
                "ungrounded_numbers": check["ungrounded_numbers"],
                "elapsed_seconds": elapsed,
            }
        )

    pass_rate = sum(r["grounded"] for r in rows) / len(rows) if rows else 0.0
    summary = {
        "n_questions": len(rows),
        "pass_rate": pass_rate,
        "n_passed": sum(r["grounded"] for r in rows),
        "n_failed": sum(not r["grounded"] for r in rows),
    }
    _write_json("groundedness_results.json", {"summary": summary, "rows": rows})
    return {"summary": summary, "rows": rows, "latencies": latencies}


# ---------------------------------------------------------------------
# 2. Tool-selection accuracy
# ---------------------------------------------------------------------
def run_tool_selection_eval() -> dict:
    rows = []
    confusion = defaultdict(Counter)  # expected_tool -> Counter of {actual_tool_or_"none": count}

    for case in TOOL_SELECTION_CASES:
        question, expected = case["question"], case["expected_tool"]
        try:
            calls = gemini_client.request_tool_calls(question, page_context=None, conversation_history=[])
        except RuntimeError as e:
            rows.append({"question": question, "expected_tool": expected, "actual_tools": [], "correct": False, "error": str(e)})
            confusion[expected]["<error>"] += 1
            continue

        actual_tools = [c["name"] for c in calls]
        correct = expected in actual_tools
        rows.append({"question": question, "expected_tool": expected, "actual_tools": actual_tools, "correct": correct})
        if correct:
            confusion[expected][expected] += 1
        else:
            for t in (actual_tools or ["<none>"]):
                confusion[expected][t] += 1

    accuracy = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
    summary = {
        "n_questions": len(rows),
        "accuracy": accuracy,
        "n_correct": sum(r["correct"] for r in rows),
        "n_incorrect": sum(not r["correct"] for r in rows),
    }
    confusion_out = {k: dict(v) for k, v in confusion.items()}
    _write_json("tool_selection_results.json", {"summary": summary, "rows": rows, "confusion": confusion_out})
    return {"summary": summary, "rows": rows, "confusion": confusion_out}


# ---------------------------------------------------------------------
# 3. Refusal correctness
# ---------------------------------------------------------------------
# "Correct" means the agent used a template that hedges or declines rather
# than answering with unqualified confidence. This is NOT just the
# refusal_* family -- segment_calibration_zero_events and
# segment_calibration_low_confidence are equally valid "I don't have a
# confident answer" outcomes; they're separate templates from
# segment_calibration_explanation specifically *because* principle 3
# requires distinct, more hedged language for the low-confidence case,
# not a refusal_* wrapper around the same sentence. An earlier version of
# this eval only counted the refusal_* prefix and scored 66.7% (10/15);
# every one of those 5 "failures" turned out to be the agent correctly
# using segment_calibration_zero_events/_low_confidence, not a real
# failure -- an eval-harness bug, not an agent behavior problem. Reported
# in the README exactly as found, including that correction.
HEDGE_TEMPLATES = {
    "refusal_not_found",
    "refusal_out_of_scope",
    "refusal_insufficient_data",
    "refusal_groundedness_failed",
    "segment_calibration_zero_events",
    "segment_calibration_low_confidence",
}


def run_refusal_eval() -> dict:
    rows = []
    latencies = []
    for case in REFUSAL_CASES:
        question, reason = case["question"], case["reason"]
        t0 = time.time()
        result = orchestrator.handle_question(question, page_context=None, conversation_history=[])
        elapsed = time.time() - t0
        latencies.append(elapsed)

        refused = result["template_used"] in HEDGE_TEMPLATES
        rows.append(
            {
                "question": question,
                "reason_should_refuse": reason,
                "template_used": result["template_used"],
                "correctly_refused": refused,
                "response": result["response"],
            }
        )

    refusal_rate = sum(r["correctly_refused"] for r in rows) / len(rows) if rows else 0.0
    summary = {
        "n_cases": len(rows),
        "correct_refusal_rate": refusal_rate,
        "n_correct": sum(r["correctly_refused"] for r in rows),
        "n_incorrect_false_confidence": sum(not r["correctly_refused"] for r in rows),
    }
    _write_json("refusal_results.json", {"summary": summary, "rows": rows})
    return {"summary": summary, "rows": rows, "latencies": latencies}


# ---------------------------------------------------------------------
# 4. Latency (p50/p95 across the full-pipeline calls made above)
# ---------------------------------------------------------------------
def compute_latency_summary(all_latencies: list) -> dict:
    if not all_latencies:
        return {"n_samples": 0}
    sorted_lat = sorted(all_latencies)
    n = len(sorted_lat)

    def _pct(p):
        idx = min(n - 1, int(round(p * (n - 1))))
        return sorted_lat[idx]

    summary = {
        "n_samples": n,
        "p50_seconds": _pct(0.50),
        "p95_seconds": _pct(0.95),
        "min_seconds": sorted_lat[0],
        "max_seconds": sorted_lat[-1],
        "mean_seconds": statistics.mean(sorted_lat),
    }
    _write_json("latency_results.json", summary)
    return summary


def run_all():
    print("Running groundedness negative control...")
    negative_control = run_groundedness_negative_control()
    print(f"  guardrail caught the fabricated number: {negative_control['guardrail_worked']}")

    print("Running groundedness eval...")
    ground = run_groundedness_eval()
    print(f"  pass rate: {ground['summary']['pass_rate']:.1%} ({ground['summary']['n_passed']}/{ground['summary']['n_questions']})")

    print("Running tool-selection accuracy eval...")
    tool_sel = run_tool_selection_eval()
    print(f"  accuracy: {tool_sel['summary']['accuracy']:.1%} ({tool_sel['summary']['n_correct']}/{tool_sel['summary']['n_questions']})")

    print("Running refusal correctness eval...")
    refusal = run_refusal_eval()
    print(f"  correct refusal rate: {refusal['summary']['correct_refusal_rate']:.1%} "
          f"({refusal['summary']['n_correct']}/{refusal['summary']['n_cases']})")

    print("Computing latency summary...")
    all_latencies = ground["latencies"] + refusal["latencies"]
    latency = compute_latency_summary(all_latencies)
    print(f"  p50: {latency.get('p50_seconds', 0):.2f}s, p95: {latency.get('p95_seconds', 0):.2f}s "
          f"(n={latency.get('n_samples', 0)})")

    overall_summary = {
        "groundedness_negative_control": negative_control,
        "groundedness": ground["summary"],
        "tool_selection": tool_sel["summary"],
        "refusal_correctness": refusal["summary"],
        "latency": latency,
    }
    _write_json("summary.json", overall_summary)
    print(f"\nAll results written to {OUT_DIR}/")
    return overall_summary


if __name__ == "__main__":
    run_all()
