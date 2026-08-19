from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from src.db.connection import get_engine
from src.models.spend.cusum import H_SIGMA, K_SIGMA

router = APIRouter()


@router.get("/anomalies")
def anomalies(
    anomaly_type: Optional[str] = None,
    department_id: Optional[int] = None,
    flagged_only: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    engine = get_engine()
    where_clauses, params = [], {}
    if anomaly_type:
        where_clauses.append("anomaly_type = :anomaly_type")
        params["anomaly_type"] = anomaly_type
    if department_id is not None:
        where_clauses.append("department_id = :department_id")
        params["department_id"] = department_id
    if flagged_only:
        where_clauses.append("predicted_flag = TRUE")
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    with engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM spend_anomaly_scores {where_sql}"), params).scalar()

    params["limit"] = page_size
    params["offset"] = (page - 1) * page_size
    query = text(
        f"SELECT * FROM spend_anomaly_scores {where_sql} "
        "ORDER BY ensemble_score DESC LIMIT :limit OFFSET :offset"
    )
    df = pd.read_sql(query, engine, params=params)
    return {"total": total, "page": page, "page_size": page_size, "results": df.to_dict(orient="records")}


@router.get("/gains-curve")
def gains_curve():
    engine = get_engine()
    df = pd.read_sql("SELECT pct_alerts_raised, pct_dollar_volume_captured FROM spend_gains_curve ORDER BY pct_alerts_raised", engine)
    return df.to_dict(orient="records")


@router.get("/drift")
def drift(employee_id: Optional[int] = None, department_id: Optional[int] = None):
    engine = get_engine()
    where_clauses, params = [], {}
    if employee_id is not None:
        where_clauses.append("employee_id = :employee_id")
        params["employee_id"] = employee_id
    if department_id is not None:
        where_clauses.append("department_id = :department_id")
        params["department_id"] = department_id
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = text(f"SELECT * FROM spend_cusum_series {where_sql} ORDER BY employee_id, merchant_category, month")
    df = pd.read_sql(query, engine, params=params)

    timing_df = pd.read_sql("SELECT * FROM spend_drift_timing_summary", engine)
    timing = timing_df.to_dict(orient="records")[0] if not timing_df.empty else None

    return {
        "control_limits": {"k_sigma": K_SIGMA, "h_sigma": H_SIGMA},
        "series": df.to_dict(orient="records"),
        "detection_timing": timing,
    }


@router.get("/detector-comparison")
def detector_comparison():
    """PR-AUC for each detector (Isolation Forest / autoencoder / CUSUM /
    ensemble), broken out per anomaly type — the standalone comparison
    that shows which detector is actually good at what, honestly."""
    engine = get_engine()
    df = pd.read_sql(
        "SELECT detector, anomaly_type, metric_value FROM spend_eval_metrics WHERE metric_name = 'pr_auc'",
        engine,
    )
    pivot = df.pivot_table(index="detector", columns="anomaly_type", values="metric_value").reset_index()
    return pivot.to_dict(orient="records")


@router.get("/alert-fatigue")
def alert_fatigue():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM spend_alert_fatigue", engine)
    return df.to_dict(orient="records")


@router.get("/transaction/{transaction_id}/explain")
def explain_transaction(transaction_id: int):
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT sub_signal, contribution FROM spend_transaction_explain WHERE transaction_id = :tid ORDER BY contribution DESC"),
        engine,
        params={"tid": transaction_id},
    )
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No explanation available (only computed for transactions flagged at the operating threshold).",
        )
    return df.to_dict(orient="records")
