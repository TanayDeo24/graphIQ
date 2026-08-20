from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from src.db.connection import get_engine

router = APIRouter()

SENSITIVITY_DISCLAIMER = (
    "This is predicted risk sensitivity to a simulated compensation change, estimated from a "
    "correlational model. It is NOT a causal effect of a raise on attrition."
)


@router.get("/risk-scores")
def risk_scores(
    department: Optional[str] = None,
    tenure_band: Optional[str] = None,
    employee_id: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    engine = get_engine()
    where_clauses, params = [], {}
    if department:
        where_clauses.append("department = :department")
        params["department"] = department
    if tenure_band:
        where_clauses.append("tenure_band = :tenure_band")
        params["tenure_band"] = tenure_band
    if employee_id is not None:
        where_clauses.append("employee_id = :employee_id")
        params["employee_id"] = employee_id
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    count_query = text(f"SELECT COUNT(*) FROM attrition_risk_scores {where_sql}")
    with engine.connect() as conn:
        total = conn.execute(count_query, params).scalar()

    params["limit"] = page_size
    params["offset"] = (page - 1) * page_size
    query = text(
        f"SELECT * FROM attrition_risk_scores {where_sql} "
        "ORDER BY gbm_risk_score DESC LIMIT :limit OFFSET :offset"
    )
    df = pd.read_sql(query, engine, params=params)
    return {"total": total, "page": page, "page_size": page_size, "results": df.to_dict(orient="records")}


@router.get("/calibration")
def calibration():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM attrition_calibration ORDER BY segment_dimension, segment_value", engine)
    return df.to_dict(orient="records")


@router.get("/lead-time")
def lead_time():
    engine = get_engine()
    summary = pd.read_sql("SELECT * FROM attrition_lead_time_summary", engine)
    raw = pd.read_sql("SELECT * FROM attrition_lead_time ORDER BY lead_time_months", engine)
    return {"summary": summary.to_dict(orient="records"), "distribution": raw.to_dict(orient="records")}


@router.get("/shap/{employee_id}")
def shap_breakdown(employee_id: int):
    engine = get_engine()
    df = pd.read_sql(
        text("SELECT * FROM attrition_shap_employee WHERE employee_id = :employee_id ORDER BY shap_value DESC"),
        engine,
        params={"employee_id": employee_id},
    )
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No SHAP breakdown for this employee_id (only computed for the top-risk decile).",
        )
    return df.to_dict(orient="records")


@router.get("/sensitivity")
def sensitivity():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM attrition_sensitivity", engine)
    return {
        "disclaimer": SENSITIVITY_DISCLAIMER,
        "results": df.to_dict(orient="records"),
    }


@router.get("/interaction-heatmap")
def interaction_heatmap():
    """baseline_tenure_band x review_score_trend-bucket grid, cell = mean
    baseline GBM risk score. low_confidence cells (n < 10) should render
    visually distinct (greyed), not colored as if reliable."""
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM attrition_interaction_heatmap", engine)
    return df.to_dict(orient="records")


@router.get("/risk-migration")
def risk_migration():
    """ILLUSTRATIVE ONLY -- a rolling re-scoring of the validated GBM model
    at 6 checkpoints, distinct from every other reported metric. See
    src/analysis/risk_migration.py's module docstring."""
    engine = get_engine()
    checkpoints = pd.read_sql("SELECT * FROM attrition_risk_migration_checkpoints", engine)
    sankey = pd.read_sql("SELECT * FROM attrition_risk_migration_sankey ORDER BY checkpoint_from, checkpoint_to", engine)
    return {
        "disclaimer": (
            "Illustrative re-scoring for visualization only -- NOT the validated baseline-only model "
            "used for every other reported metric in this project. Do not treat this as an evaluation result."
        ),
        "checkpoints": checkpoints.to_dict(orient="records"),
        "sankey_links": sankey.to_dict(orient="records"),
    }
