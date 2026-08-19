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
