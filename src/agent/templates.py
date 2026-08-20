"""Fixed template library for the explainability agent.

Non-negotiable per the build spec: methodology/explanation language comes
from these pre-written templates, not free composition by the model. The
model (via the orchestrator) only supplies which template applies and the
live values to fill into it -- it never writes new explanatory prose about
what a metric means. Any "conversational wrapping" the model adds around a
rendered template is minimal and is still passed through
groundedness.check_groundedness() before anything reaches the frontend, so
a stray invented number in the wrapper is caught the same as anywhere else.

Each entry is a TemplateSpec: a Python .format()-style string plus the set
of placeholder names it requires. Rendering is a plain, deterministic
str.format() call in code (render_template() below) -- never delegated to
the model. This is what principle 2 means by "not only by prompting":
even if the model tried to paraphrase the methodology text, the pipeline
never runs the model's paraphrase, only substitutes values into the fixed
string here.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TemplateSpec:
    key: str
    text: str
    required_fills: frozenset


TEMPLATES = {
    "attrition_risk_explanation": TemplateSpec(
        key="attrition_risk_explanation",
        text=(
            "Employee {employee_id}'s baseline GBM attrition risk score is {risk_score:.3f} "
            "(tenure band: {tenure_band}). Risk scores are model output on an arbitrary scale, not a "
            "probability -- they're meaningful for ranking employees against each other, not as a "
            "standalone number. This employee {top_quartile_phrase} the model's top-risk quartile "
            "cutoff.{shap_section}"
        ),
        required_fills=frozenset({"employee_id", "risk_score", "tenure_band", "top_quartile_phrase", "shap_section"}),
    ),
    # Used to fill the {shap_section} placeholder above when SHAP data is available.
    "_shap_section_with_data": TemplateSpec(
        key="_shap_section_with_data",
        text=(
            " The top drivers of this score (SHAP values, largest-magnitude first): {shap_driver_list}. "
            "A positive SHAP value pushes this employee's predicted risk higher; a negative value pushes "
            "it lower. SHAP values explain this specific model's specific prediction -- they are not a "
            "causal claim about what would happen if a driver changed."
        ),
        required_fills=frozenset({"shap_driver_list"}),
    ),
    "_shap_section_unavailable": TemplateSpec(
        key="_shap_section_unavailable",
        text=(
            " SHAP driver breakdown is not available for this employee -- it's only computed for the "
            "top-risk decile, and this employee falls outside it."
        ),
        required_fills=frozenset(),
    ),
    "segment_calibration_explanation": TemplateSpec(
        key="segment_calibration_explanation",
        text=(
            "For the {segment_dimension} = {segment_value} segment (n_at_risk = {n_at_risk}, "
            "event_count = {event_count}): the model predicted {predicted_survival:.1%} 12-month survival "
            "and {observed_survival:.1%} was actually observed, a calibration error of "
            "{calibration_error:.1%} points. This segment has enough observed events "
            "(>= {low_confidence_threshold_event_count}) to read that calibration error with normal "
            "confidence."
        ),
        required_fills=frozenset(
            {
                "segment_dimension", "segment_value", "n_at_risk", "event_count", "predicted_survival",
                "observed_survival", "calibration_error", "low_confidence_threshold_event_count",
            }
        ),
    ),
    "segment_calibration_low_confidence": TemplateSpec(
        key="segment_calibration_low_confidence",
        text=(
            "I can show you the numbers for the {segment_dimension} = {segment_value} segment, but I "
            "can't stand behind them as reliable: only {event_count} observed event(s) went into this "
            "estimate (n_at_risk = {n_at_risk}), below the {low_confidence_threshold_event_count}-event "
            "bar this project uses for a trustworthy calibration read. The reported numbers are "
            "predicted {predicted_survival:.1%} vs. observed {observed_survival:.1%} 12-month survival "
            "(calibration error {calibration_error:.1%} points) -- but with this few events, that "
            "calibration error could easily look very different with a handful more or fewer departures, "
            "so treat it as a data point, not a conclusion."
        ),
        required_fills=frozenset(
            {
                "segment_dimension", "segment_value", "n_at_risk", "event_count", "predicted_survival",
                "observed_survival", "calibration_error", "low_confidence_threshold_event_count",
            }
        ),
    ),
    "segment_calibration_zero_events": TemplateSpec(
        key="segment_calibration_zero_events",
        text=(
            "The {segment_dimension} = {segment_value} segment has {event_count} observed events "
            "(n_at_risk = {n_at_risk}) at this 12-month horizon, so its {observed_survival:.0%} observed "
            "survival isn't a calibration result at all -- it's the Kaplan-Meier estimator's "
            "definitionally-trivial output when zero departures were observed, carrying no real "
            "information. I can't give you a meaningful calibration read for this segment at this "
            "horizon; a longer horizon would be needed."
        ),
        required_fills=frozenset(
            {"segment_dimension", "segment_value", "n_at_risk", "event_count", "observed_survival"}
        ),
    ),
    "detector_comparison_explanation": TemplateSpec(
        key="detector_comparison_explanation",
        text=(
            "For {anomaly_type} anomalies, {best_detector} is the best-performing standalone detector: "
            "PR-AUC {pr_auc:.3f}, which is {lift:.1f}x better than randomly guessing at this anomaly "
            "type's actual prevalence rate. PR-AUC is a ranking-quality metric, not a probability of "
            "any one transaction being anomalous."
        ),
        required_fills=frozenset({"anomaly_type", "best_detector", "pr_auc", "lift"}),
    ),
    "spend_transaction_explanation": TemplateSpec(
        key="spend_transaction_explanation",
        text=(
            "Transaction {transaction_id} was flagged primarily by its {top_sub_signal} sub-signal, "
            "which contributed {contribution:.3f} of this transaction's total anomaly explanation "
            "(other contributing sub-signals: {other_signals_list}). This decomposition explains what the "
            "ensemble model reacted to, not a certainty judgment that this specific transaction is "
            "actually improper spend."
        ),
        required_fills=frozenset({"transaction_id", "top_sub_signal", "contribution", "other_signals_list"}),
    ),
    "cross_component_explanation": TemplateSpec(
        key="cross_component_explanation",
        text=(
            "For employee {employee_id}: attrition risk score {risk_score:.3f}, spend-anomaly signal "
            "(flagged transaction count) {spend_anomaly_score}, placing them in the {quadrant} quadrant. "
            "Across all employees, the primary reported figure is the partial Spearman correlation "
            "between attrition risk and spend-anomaly score, controlling for department and monthly "
            "income: {partial_correlation:.3f} (p {partial_p_value_display}). The uncontrolled bivariate "
            "correlation is {bivariate_correlation:.3f} (p {bivariate_p_value_display}), kept for "
            "reference only. This is an observed statistical association between two independently "
            "-trained model outputs on synthetic/real-hybrid data -- it is NOT a causal claim that "
            "attrition risk causes or is caused by anomalous spend, and is not a real finding about any "
            "actual person or company."
        ),
        required_fills=frozenset(
            {
                "employee_id", "risk_score", "spend_anomaly_score", "quadrant", "partial_correlation",
                "partial_p_value_display", "bivariate_correlation", "bivariate_p_value_display",
            }
        ),
    ),
    "lead_time_explanation": TemplateSpec(
        key="lead_time_explanation",
        text=(
            "For true-positive predictions (employees the model correctly flagged as high-risk who did "
            "leave), the model gives a mean of {mean_lead_time:.1f} months of advance warning "
            "(95% CI {mean_ci_low:.1f}-{mean_ci_high:.1f}, n={n_true_positives}) and a median of "
            "{median_lead_time:.1f} months. This is measured only over confirmed true positives -- it "
            "doesn't say anything about how much warning the model gives for a false-positive or a "
            "still-employed flagged employee."
        ),
        required_fills=frozenset(
            {"mean_lead_time", "mean_ci_low", "mean_ci_high", "n_true_positives", "median_lead_time"}
        ),
    ),
    "gains_curve_explanation": TemplateSpec(
        key="gains_curve_explanation",
        text=(
            "Reviewing the top {pct_alerts_raised:.0%} of alerts (ranked by ensemble anomaly score) "
            "captures {pct_dollar_volume_captured:.1%} of the total flagged anomalous dollar volume. This "
            "is a ranking-quality measure of the ensemble score, not a claim about how much of a "
            "company's real spend fraud any particular review process would catch."
        ),
        required_fills=frozenset({"pct_alerts_raised", "pct_dollar_volume_captured"}),
    ),
    "refusal_insufficient_data": TemplateSpec(
        key="refusal_insufficient_data",
        text=(
            "I don't have a confident answer for that. {reason} Rather than guess, I'd rather tell you "
            "plainly that this isn't something I can answer reliably right now."
        ),
        required_fills=frozenset({"reason"}),
    ),
    "refusal_out_of_scope": TemplateSpec(
        key="refusal_out_of_scope",
        text=(
            "That's outside what I can answer from this project's result tables. Here's what I can help "
            "with instead: {topics_list}."
        ),
        required_fills=frozenset({"topics_list"}),
    ),
    "refusal_not_found": TemplateSpec(
        key="refusal_not_found",
        text=(
            "I couldn't find that in the data. {reason} If you meant a different employee ID, department, "
            "segment, or transaction, let me know and I'll look again."
        ),
        required_fills=frozenset({"reason"}),
    ),
    "refusal_groundedness_failed": TemplateSpec(
        key="refusal_groundedness_failed",
        text=(
            "I put together an answer to that, but it didn't pass this project's groundedness check "
            "(every number in a response has to trace back to an actual tool result), so I'm not going "
            "to show it to you. This usually means the question needs a tool this agent doesn't have "
            "yet, rather than a wrong number -- try rephrasing, or ask about {topics_list}."
        ),
        required_fills=frozenset({"topics_list"}),
    ),
}


def render_template(key: str, fills: dict) -> str:
    """Deterministic, code-only rendering -- str.format() against the fixed
    template text, never model-generated. Raises KeyError loudly if a
    required fill is missing rather than silently rendering a broken
    sentence, since a missing fill means the caller built the fills dict
    wrong and that should fail fast, not reach a user."""
    spec = TEMPLATES[key]
    missing = spec.required_fills - set(fills.keys())
    if missing:
        raise KeyError(f"render_template({key!r}) missing required fills: {sorted(missing)}")
    return spec.text.format(**fills)


def render_shap_section(shap_rows: list) -> str:
    """Fills the attrition_risk_explanation template's {shap_section}
    placeholder -- itself rendered from one of the two _shap_section_*
    fixed templates above, never free text."""
    if not shap_rows:
        return render_template("_shap_section_unavailable", {})
    driver_list = "; ".join(
        f"{row['feature']} ({row['shap_value']:+.3f})" for row in shap_rows
    )
    return render_template("_shap_section_with_data", {"shap_driver_list": driver_list})
