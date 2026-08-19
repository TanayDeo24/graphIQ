"""Shared feature engineering for the attrition survival models (Cox PH
and GBM survival), read from the shared Postgres schema. Both models in
this component consume the exact same feature frame so their evaluation
metrics are directly comparable.

Features (per build spec Section 5): department, job level, monthly
income (latest value from comp_history), tenure so far, review-score
trend (slope of last 2 reviews), number of raises received, benefits
tier.
"""

import numpy as np
import pandas as pd

from src.generation.attrition_extension import assign_temporal_split
from src.db.connection import get_engine

FEATURE_COLUMNS_NUMERIC = ["job_level", "monthly_income", "tenure_months", "review_score_trend", "num_raises"]
FEATURE_COLUMNS_CATEGORICAL = ["department", "benefits_tier"]


def _latest_income(engine) -> pd.DataFrame:
    query = """
        SELECT DISTINCT ON (employee_id) employee_id, monthly_income
        FROM comp_history
        ORDER BY employee_id, effective_month DESC
    """
    return pd.read_sql(query, engine)


def _num_raises(engine) -> pd.DataFrame:
    query = """
        SELECT employee_id, COUNT(*) AS num_raises
        FROM comp_history
        WHERE change_type = 'raise'
        GROUP BY employee_id
    """
    return pd.read_sql(query, engine)


def _review_score_trend(engine) -> pd.DataFrame:
    query = """
        SELECT employee_id, review_month, review_score
        FROM performance_reviews
        ORDER BY employee_id, review_month
    """
    reviews = pd.read_sql(query, engine)
    if reviews.empty:
        return pd.DataFrame(columns=["employee_id", "review_score_trend"])

    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return 0.0
        last_two = group.tail(2)
        return float(last_two["review_score"].iloc[1] - last_two["review_score"].iloc[0])

    trend = reviews.groupby("employee_id").apply(_slope, include_groups=False).rename("review_score_trend")
    return trend.reset_index()


def build_feature_frame(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()

    employees = pd.read_sql(
        "SELECT employee_id, department, job_level, tenure_years, duration_months, event_observed "
        "FROM employees",
        engine,
    )
    benefits = pd.read_sql("SELECT employee_id, plan_tier AS benefits_tier FROM benefits_enrollment", engine)
    income = _latest_income(engine)
    raises = _num_raises(engine)
    trend = _review_score_trend(engine)

    df = employees.merge(income, on="employee_id", how="left")
    df = df.merge(raises, on="employee_id", how="left")
    df = df.merge(trend, on="employee_id", how="left")
    df = df.merge(benefits, on="employee_id", how="left")

    df["num_raises"] = df["num_raises"].fillna(0).astype(int)
    df["review_score_trend"] = df["review_score_trend"].fillna(0.0)
    df["tenure_months"] = df["duration_months"]
    df["benefits_tier"] = df["benefits_tier"].fillna("basic")

    df["data_split"] = df["duration_months"].apply(assign_temporal_split)

    return df[
        ["employee_id", "duration_months", "event_observed", "data_split"]
        + FEATURE_COLUMNS_NUMERIC
        + FEATURE_COLUMNS_CATEGORICAL
    ]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals; leaves numeric columns untouched."""
    encoded = pd.get_dummies(df, columns=FEATURE_COLUMNS_CATEGORICAL, drop_first=True)
    return encoded


