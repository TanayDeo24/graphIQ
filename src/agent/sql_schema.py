"""Single source of truth for which Postgres tables the text-to-SQL agent
(src/agent/sql_agent.py) may query, and their schema. Referenced by three
places that must never drift apart:

1. src/db/agent_readonly.py -- generates the actual `GRANT SELECT ON
   (...)` statement from ALLOWED_TABLES, so the database-level permission
   boundary and this allowlist are the same list, not two lists someone
   has to remember to keep in sync.
2. src/agent/sql_validator.py -- rejects any generated SQL that
   references a table not in ALLOWED_TABLES, as a second, independent
   layer on top of the database grant (defense in depth: even if the
   validator had a bug, the database role itself has no access to
   anything else).
3. src/agent/sql_agent.py -- renders this as schema context in the
   text-to-SQL prompt, so the active LLM backend only ever sees the
   tables it's actually allowed to query.

Deliberately excludes the raw source tables (employees, comp_history,
performance_reviews, benefits_enrollment, expense_transactions) -- see
README's Explainability agent section for why. `departments` is included:
a pure id -> name lookup with no employee-level or otherwise sensitive
data, already joined against by this project's own existing routes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    data_type: str


@dataclass(frozen=True)
class TableSpec:
    name: str
    description: str
    columns: tuple


ALLOWED_TABLES = {
    t.name: t
    for t in [
        TableSpec(
            "attrition_risk_scores",
            "One row per employee: the validated baseline GBM/Cox attrition risk scores, department, "
            "tenure band, and whether they're in the top-risk quartile.",
            (
                ColumnSpec("employee_id", "integer"), ColumnSpec("department", "text"),
                ColumnSpec("tenure_band", "text"), ColumnSpec("data_split", "text"),
                ColumnSpec("cox_risk_score", "double precision"), ColumnSpec("gbm_risk_score", "double precision"),
                ColumnSpec("gbm_predicted_survival_12m", "double precision"),
                ColumnSpec("is_top_risk_quartile", "boolean"),
            ),
        ),
        TableSpec(
            "attrition_shap_employee",
            "Per-employee SHAP feature-driver breakdown for the GBM risk score -- only computed for the "
            "top-risk decile.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("employee_id", "integer"), ColumnSpec("feature", "text"),
                ColumnSpec("feature_value", "double precision"), ColumnSpec("shap_value", "double precision"),
                ColumnSpec("base_value", "double precision"),
            ),
        ),
        TableSpec(
            "attrition_shap_global",
            "Global (model-wide, not per-employee) mean absolute SHAP importance per feature.",
            (ColumnSpec("id", "integer"), ColumnSpec("feature", "text"), ColumnSpec("mean_abs_shap", "double precision")),
        ),
        TableSpec(
            "attrition_calibration",
            "Predicted vs. observed 12-month survival per segment (department, tenure_band, comp_band, "
            "department_x_tenure_band), with n_at_risk/event_count -- segments with event_count < 10 are "
            "low-confidence, event_count = 0 means the calibration figure is not meaningful at all.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("segment_dimension", "text"), ColumnSpec("segment_value", "text"),
                ColumnSpec("horizon_months", "integer"), ColumnSpec("predicted_survival", "double precision"),
                ColumnSpec("observed_survival", "double precision"), ColumnSpec("calibration_error", "double precision"),
                ColumnSpec("logrank_p_value", "double precision"), ColumnSpec("n_at_risk", "integer"),
                ColumnSpec("event_count", "integer"),
            ),
        ),
        TableSpec(
            "attrition_interaction_heatmap",
            "3x3 grid of tenure_band x review_trend_bucket, cell = mean GBM risk score; low_confidence "
            "flags cells with n < 10.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("tenure_band", "text"), ColumnSpec("review_trend_bucket", "text"),
                ColumnSpec("n", "integer"), ColumnSpec("mean_gbm_risk_score", "double precision"),
                ColumnSpec("low_confidence", "boolean"),
            ),
        ),
        TableSpec(
            "attrition_lead_time",
            "Per-true-positive-employee lead time (months between risk-threshold crossing and actual departure).",
            (
                ColumnSpec("id", "integer"), ColumnSpec("employee_id", "integer"), ColumnSpec("flagged_month", "integer"),
                ColumnSpec("departure_month", "integer"), ColumnSpec("lead_time_months", "integer"),
            ),
        ),
        TableSpec(
            "attrition_lead_time_summary",
            "Bootstrapped mean/median lead time (with 95% CI) across all true positives.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("statistic", "text"), ColumnSpec("point_estimate", "double precision"),
                ColumnSpec("ci_low", "double precision"), ColumnSpec("ci_high", "double precision"),
                ColumnSpec("n_bootstrap", "integer"), ColumnSpec("n_true_positives", "integer"),
                ColumnSpec("n_excluded_no_crossing", "integer"), ColumnSpec("risk_threshold_note", "text"),
            ),
        ),
        TableSpec(
            "attrition_model_metrics",
            "Headline attrition model metrics: concordance index, within-tenure-band concordance, "
            "integrated Brier score, time-dependent AUC, per model (cox_ph / gbm_survival).",
            (
                ColumnSpec("id", "integer"), ColumnSpec("model", "text"), ColumnSpec("metric_name", "text"),
                ColumnSpec("horizon_months", "integer"), ColumnSpec("metric_value", "double precision"),
                ColumnSpec("ci_low", "double precision"), ColumnSpec("ci_high", "double precision"),
            ),
        ),
        TableSpec(
            "attrition_ph_assumption",
            "Cox proportional-hazards assumption check (Schoenfeld residual test) per feature; violated=true "
            "means the constant-hazard-ratio assumption doesn't hold for that feature.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("feature", "text"), ColumnSpec("test_statistic", "double precision"),
                ColumnSpec("p_value", "double precision"), ColumnSpec("violated", "boolean"),
            ),
        ),
        TableSpec(
            "attrition_sensitivity",
            "Counterfactual sensitivity: predicted risk-score change under a simulated +10% income change, "
            "per employee. Correlational only, not a causal effect estimate.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("employee_id", "integer"), ColumnSpec("base_risk", "double precision"),
                ColumnSpec("perturbed_risk", "double precision"), ColumnSpec("risk_change", "double precision"),
                ColumnSpec("risk_change_pct", "double precision"), ColumnSpec("disclaimer", "text"),
            ),
        ),
        TableSpec(
            "attrition_risk_migration_checkpoints",
            "ILLUSTRATIVE ONLY, not the validated risk score: a separate re-scoring of the GBM model at 6 "
            "checkpoints (months) per employee, tiered low/medium/high, used only for the dashboard's Sankey.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("employee_id", "integer"), ColumnSpec("checkpoint_month", "integer"),
                ColumnSpec("risk_score", "double precision"), ColumnSpec("tier", "text"),
            ),
        ),
        TableSpec(
            "attrition_risk_migration_sankey",
            "ILLUSTRATIVE ONLY: tier-transition counts between consecutive checkpoints, derived from "
            "attrition_risk_migration_checkpoints.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("checkpoint_from", "integer"), ColumnSpec("checkpoint_to", "integer"),
                ColumnSpec("tier_from", "text"), ColumnSpec("tier_to", "text"), ColumnSpec("employee_count", "integer"),
            ),
        ),
        TableSpec(
            "cross_component_quadrant",
            "Per-employee join of attrition risk score and spend-anomaly signal (flagged transaction "
            "count), with the resulting quadrant assignment.",
            (
                ColumnSpec("employee_id", "integer"), ColumnSpec("department", "text"), ColumnSpec("tenure_band", "text"),
                ColumnSpec("gbm_risk_score", "double precision"), ColumnSpec("is_top_risk_quartile", "boolean"),
                ColumnSpec("spend_anomaly_score", "double precision"), ColumnSpec("is_top_spend_quartile", "boolean"),
                ColumnSpec("quadrant", "text"),
            ),
        ),
        TableSpec(
            "cross_component_summary",
            "One row: the bivariate Spearman correlation and the partial Spearman correlation (controlling "
            "for department + monthly_income -- the primary reported figure) between attrition risk and "
            "spend-anomaly score, both with permutation-test p-values. Correlational, not causal.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("spearman_correlation", "double precision"),
                ColumnSpec("p_value", "double precision"), ColumnSpec("n_permutations", "integer"),
                ColumnSpec("n_employees", "integer"), ColumnSpec("method_note", "text"), ColumnSpec("disclaimer", "text"),
                ColumnSpec("partial_spearman_correlation", "double precision"), ColumnSpec("partial_p_value", "double precision"),
                ColumnSpec("partial_confounds", "text"), ColumnSpec("partial_method_note", "text"),
            ),
        ),
        TableSpec(
            "cross_component_quadrant_characteristics",
            "Department/tenure-band composition breakdown within each cross-component quadrant.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("quadrant", "text"), ColumnSpec("dimension", "text"),
                ColumnSpec("dimension_value", "text"), ColumnSpec("count", "integer"), ColumnSpec("pct_of_quadrant", "double precision"),
            ),
        ),
        TableSpec(
            "departments",
            "Lookup table: department_id -> department_name. No employee-level or sensitive data.",
            (ColumnSpec("department_id", "integer"), ColumnSpec("department_name", "text")),
        ),
        TableSpec(
            "spend_anomaly_scores",
            "Per-transaction spend anomaly scores from every detector (Isolation Forest, autoencoder, "
            "CUSUM, ensemble), the predicted flag, and (for evaluation) whether it was a synthetically "
            "injected anomaly and of which type.",
            (
                ColumnSpec("transaction_id", "bigint"), ColumnSpec("employee_id", "integer"), ColumnSpec("department_id", "integer"),
                ColumnSpec("isolation_forest_score", "double precision"), ColumnSpec("autoencoder_score", "double precision"),
                ColumnSpec("cusum_flag", "boolean"), ColumnSpec("ensemble_score", "double precision"),
                ColumnSpec("predicted_flag", "boolean"), ColumnSpec("is_injected_anomaly", "boolean"),
                ColumnSpec("anomaly_type", "text"),
            ),
        ),
        TableSpec(
            "spend_eval_metrics",
            "Precision/recall/pr_auc/lift_over_random per detector x anomaly_type (point_spike, "
            "slow_drift, coordinated_pattern, overall) -- the detector comparison table's source data.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("detector", "text"), ColumnSpec("anomaly_type", "text"),
                ColumnSpec("metric_name", "text"), ColumnSpec("metric_value", "double precision"),
            ),
        ),
        TableSpec(
            "spend_robustness",
            "Detector performance (precision/recall/pr_auc/lift_over_random) re-measured at 1%/5%/10% "
            "anomaly injection rates, ensemble only, for robustness checking.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("injection_rate", "text"), ColumnSpec("anomaly_type", "text"),
                ColumnSpec("precision", "double precision"), ColumnSpec("recall", "double precision"),
                ColumnSpec("pr_auc", "double precision"),
            ),
        ),
        TableSpec(
            "spend_gains_curve",
            "Dollar-weighted gains curve: share of flagged anomalous dollar volume captured at each "
            "alert-volume percentile (ranked by ensemble score).",
            (ColumnSpec("id", "integer"), ColumnSpec("pct_alerts_raised", "double precision"), ColumnSpec("pct_dollar_volume_captured", "double precision")),
        ),
        TableSpec(
            "spend_alert_fatigue",
            "One row: the operating threshold, alerts raised, alerts per 1000 transactions, and precision "
            "at that threshold.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("operating_threshold", "double precision"),
                ColumnSpec("alerts_raised", "integer"), ColumnSpec("total_transactions", "integer"),
                ColumnSpec("alerts_per_1000_txns", "double precision"), ColumnSpec("precision_at_threshold", "double precision"),
            ),
        ),
        TableSpec(
            "spend_transaction_explain",
            "Per-flagged-transaction sub-signal decomposition (which sub-signal drove the flag, and its "
            "contribution) -- only populated for transactions flagged at the operating threshold.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("transaction_id", "bigint"), ColumnSpec("sub_signal", "text"),
                ColumnSpec("contribution", "double precision"),
            ),
        ),
        TableSpec(
            "spend_detector_overlap",
            "At the operating threshold, count of transactions flagged by each combination of {Isolation "
            "Forest, autoencoder, CUSUM} (only_IF, only_AE, only_CUSUM, IF+AE, IF+CUSUM, AE+CUSUM, all_three).",
            (ColumnSpec("id", "integer"), ColumnSpec("combination", "text"), ColumnSpec("transaction_count", "integer")),
        ),
        TableSpec(
            "spend_dollar_treemap",
            "Anomalous dollar volume and transaction count per department x merchant_category x anomaly_type.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("department_id", "integer"), ColumnSpec("merchant_category", "text"),
                ColumnSpec("anomaly_type", "text"), ColumnSpec("dollar_volume", "double precision"),
                ColumnSpec("transaction_count", "integer"),
            ),
        ),
        TableSpec(
            "spend_cusum_series",
            "Full monthly CUSUM statistic trajectory per employee x merchant_category, with the flagged boolean.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("employee_id", "integer"), ColumnSpec("department_id", "integer"),
                ColumnSpec("merchant_category", "text"), ColumnSpec("month", "date"),
                ColumnSpec("monthly_total", "double precision"), ColumnSpec("cusum_statistic", "double precision"),
                ColumnSpec("flagged", "boolean"),
            ),
        ),
        TableSpec(
            "spend_cusum_annotated_trajectory",
            "5 curated real slow_drift cases with their full monthly CUSUM trajectory, drift window, and "
            "detection month, for the dashboard's annotated chart.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("case_label", "text"), ColumnSpec("employee_id", "integer"),
                ColumnSpec("merchant_category", "text"), ColumnSpec("month", "date"), ColumnSpec("cusum_statistic", "double precision"),
                ColumnSpec("flagged", "boolean"), ColumnSpec("onset_month", "date"), ColumnSpec("end_month", "date"),
                ColumnSpec("flagged_month", "date"), ColumnSpec("caught_during_active_window", "boolean"),
            ),
        ),
        TableSpec(
            "spend_drift_delay",
            "Per-slow_drift-case detection delay (months from drift onset to flagged month), with whether "
            "it was caught while the drift was still active.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("employee_id", "integer"), ColumnSpec("merchant_category", "text"),
                ColumnSpec("onset_month", "date"), ColumnSpec("end_month", "date"), ColumnSpec("flagged_month", "date"),
                ColumnSpec("delay_months", "integer"), ColumnSpec("caught_during_active_window", "boolean"),
            ),
        ),
        TableSpec(
            "spend_drift_delay_summary",
            "Bootstrapped mean/median CUSUM drift-detection delay (with 95% CI) across all detected cases.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("statistic", "text"), ColumnSpec("point_estimate", "double precision"),
                ColumnSpec("ci_low", "double precision"), ColumnSpec("ci_high", "double precision"),
                ColumnSpec("n_bootstrap", "integer"), ColumnSpec("n_cases", "integer"),
            ),
        ),
        TableSpec(
            "spend_drift_timing_summary",
            "One row: total/detected/undetected slow_drift case counts, and the split between caught-while"
            "-active vs. caught-after-drift-ended.",
            (
                ColumnSpec("id", "integer"), ColumnSpec("n_total_cases", "integer"), ColumnSpec("n_detected", "integer"),
                ColumnSpec("n_undetected", "integer"), ColumnSpec("n_caught_during_active", "integer"),
                ColumnSpec("n_caught_after_ended", "integer"), ColumnSpec("pct_caught_during_active", "double precision"),
                ColumnSpec("pct_caught_after_ended", "double precision"),
            ),
        ),
    ]
}


def render_schema_for_prompt() -> str:
    """Renders ALLOWED_TABLES as plain-text schema context for the
    text-to-SQL prompt -- table name, one-line description, and every
    column with its type."""
    lines = []
    for table in ALLOWED_TABLES.values():
        lines.append(f"TABLE {table.name} -- {table.description}")
        cols = ", ".join(f"{c.name} ({c.data_type})" for c in table.columns)
        lines.append(f"  columns: {cols}")
    return "\n".join(lines)
