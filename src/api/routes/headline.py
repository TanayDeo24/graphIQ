import pandas as pd
from fastapi import APIRouter

from src.db.connection import get_engine

router = APIRouter()


@router.get("/stats")
def headline_stats():
    """3-4 headline numbers for the dashboard's landing page, each pulled
    from an already-computed, already-validated result table — nothing
    computed at request time. Picked as the strongest, most defensible
    numbers this project actually validated (not the most flattering ones
    available)."""
    engine = get_engine()

    within_band = pd.read_sql(
        "SELECT metric_value FROM attrition_model_metrics "
        "WHERE model = 'gbm_survival' AND metric_name = 'within_tenure_band_concordance'",
        engine,
    )

    timing = pd.read_sql("SELECT n_detected, n_total_cases, pct_caught_during_active FROM spend_drift_timing_summary", engine)

    gains = pd.read_sql("SELECT pct_alerts_raised, pct_dollar_volume_captured FROM spend_gains_curve", engine)
    gains["diff"] = (gains["pct_alerts_raised"] - 0.10).abs()
    top10 = gains.sort_values("diff").iloc[0] if not gains.empty else None

    ensemble_lift = pd.read_sql(
        "SELECT metric_value FROM spend_eval_metrics WHERE detector = 'ensemble' AND anomaly_type = 'overall' "
        "AND metric_name = 'lift_over_random'",
        engine,
    )

    stats = []
    if not within_band.empty:
        stats.append(
            {
                "label": "GBM within-tenure-band concordance",
                "value": float(within_band["metric_value"].iloc[0]),
                "format": "decimal",
                "note": "Concordance controlling for tenure dominance — the honest headline number for the attrition model, in the range typical of published real-world attrition models.",
            }
        )
    if not timing.empty:
        row = timing.iloc[0]
        stats.append(
            {
                "label": "CUSUM slow_drift detection rate",
                "value": float(row["n_detected"]) / float(row["n_total_cases"]),
                "format": "percent",
                "note": f"{int(row['n_detected'])} of {int(row['n_total_cases'])} injected drift cases detected, after the diagnosed baseline-contamination fix and tuning-dataset-derived threshold retune.",
            }
        )
    if top10 is not None:
        stats.append(
            {
                "label": "Top 10% of alerts capture",
                "value": float(top10["pct_dollar_volume_captured"]),
                "format": "percent",
                "note": "Share of flagged anomalous dollar volume captured by the top decile of ensemble-ranked alerts.",
            }
        )
    if not ensemble_lift.empty:
        stats.append(
            {
                "label": "Ensemble lift over random (overall)",
                "value": float(ensemble_lift["metric_value"].iloc[0]),
                "format": "multiplier",
                "note": "Ensemble PR-AUC divided by the overall injected anomaly prevalence rate — how much better than random guessing the ensemble does.",
            }
        )

    return {"stats": stats}
