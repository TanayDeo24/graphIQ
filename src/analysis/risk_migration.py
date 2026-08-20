"""Illustrative risk-migration Sankey data — NOT the validated attrition
model used for every other reported metric in this project.

Re-scores each employee's already-trained GBM survival model at 6
checkpoints (months 6/12/18/24/30/36) using their comp_history/
performance_reviews values *as of that checkpoint* (a genuinely rolling,
cumulative-to-date computation) rather than the fixed BASELINE_MONTHS-only
window the validated model in src/models/attrition/evaluate.py uses. That
difference is the whole point of this script — it exists only to show how
a rolling re-scoring would move employees between risk tiers over time for
a Sankey diagram, and its outputs must never be confused with, or fed
back into, any reported evaluation metric (concordance, AUC, calibration,
etc.), all of which come only from the baseline-only, validated model.

For employees whose generated comp_history/performance_reviews window
ends before a given checkpoint (real tenure_years == 0 has only a
1-month window, for example), values are carried forward from their last
available observation (last observation carried forward) rather than
treated as missing — a standard, documented choice for this kind of
rolling re-scoring, not a claim that anything new happened for them.

Tier (low/medium/high) is assigned by tercile *separately at each
checkpoint* (i.e. the tier boundaries themselves shift over time with the
population's score distribution at that checkpoint), per the build
instructions.
"""

import os

import numpy as np
import pandas as pd

from src.db.connection import get_engine
from src.generation.attrition_extension import WINDOW_START
from src.models.attrition import gbm_survival
from src.models.attrition.features import build_feature_frame

OUT_DIR = "data/generated/attrition_outputs"
CHECKPOINT_MONTHS = [6, 12, 18, 24, 30, 36]
TIERS = ["low", "medium", "high"]
FIT_CENSOR_MONTHS = 24  # same admin-censoring horizon the validated model fits against


def _fit_baseline_model(engine):
    """Refit the identical validated GBM model (same fit_view construction,
    same fixed random_state in gbm_survival.py) so this script doesn't
    depend on a persisted model object. Deterministic — this is the same
    model, not a new one."""
    full_df = build_feature_frame(engine)
    fit_view = full_df.copy()
    fit_view["duration_months"] = fit_view["duration_months"].clip(upper=FIT_CENSOR_MONTHS)
    fit_view["event_observed"] = full_df["event_observed"] & (full_df["duration_months"] <= FIT_CENSOR_MONTHS)
    gbm, gbm_features = gbm_survival.fit_gbm_model(fit_view)
    return gbm, gbm_features, full_df


def _load_raw_history(engine) -> dict:
    comp = pd.read_sql("SELECT employee_id, effective_month, monthly_income, change_type FROM comp_history", engine,
                        parse_dates=["effective_month"])
    reviews = pd.read_sql("SELECT employee_id, review_month, review_score FROM performance_reviews", engine,
                           parse_dates=["review_month"])
    return {
        "comp": {eid: g.sort_values("effective_month") for eid, g in comp.groupby("employee_id")},
        "reviews": {eid: g.sort_values("review_month") for eid, g in reviews.groupby("employee_id")},
    }


def _checkpoint_features_for_employee(employee_id, checkpoint_date, comp_by_emp, reviews_by_emp) -> dict:
    comp = comp_by_emp.get(employee_id)
    as_of_comp = comp[comp["effective_month"] <= checkpoint_date] if comp is not None else None
    if as_of_comp is None or as_of_comp.empty:
        # Before the employee's first logged comp_history row (or no rows at all) --
        # carry forward the earliest available value, or 0 raises if truly none.
        monthly_income = float(comp["monthly_income"].iloc[0]) if comp is not None and not comp.empty else np.nan
        num_raises = 0
    else:
        monthly_income = float(as_of_comp["monthly_income"].iloc[-1])
        num_raises = int((as_of_comp["change_type"] == "raise").sum())

    reviews = reviews_by_emp.get(employee_id)
    as_of_reviews = reviews[reviews["review_month"] <= checkpoint_date] if reviews is not None else None
    if as_of_reviews is None or len(as_of_reviews) < 2:
        review_score_trend = 0.0
    else:
        last_two = as_of_reviews.tail(2)
        review_score_trend = float(last_two["review_score"].iloc[1] - last_two["review_score"].iloc[0])

    return {
        "monthly_income": monthly_income,
        "num_raises": num_raises,
        "review_score_trend": review_score_trend,
    }


def build_checkpoint_scores(engine) -> pd.DataFrame:
    gbm, gbm_features, full_df = _fit_baseline_model(engine)
    history = _load_raw_history(engine)

    static_cols = full_df.set_index("employee_id")[["department", "job_level", "benefits_tier", "baseline_tenure_band"]]

    rows = []
    for checkpoint_month in CHECKPOINT_MONTHS:
        checkpoint_date = WINDOW_START + pd.DateOffset(months=checkpoint_month)
        records = []
        for employee_id in full_df["employee_id"]:
            feats = _checkpoint_features_for_employee(employee_id, checkpoint_date, history["comp"], history["reviews"])
            static = static_cols.loc[employee_id]
            records.append(
                {
                    "employee_id": employee_id,
                    "department": static["department"],
                    "job_level": static["job_level"],
                    "benefits_tier": static["benefits_tier"],
                    "baseline_tenure_band": static["baseline_tenure_band"],
                    **feats,
                }
            )
        checkpoint_df = pd.DataFrame(records)
        checkpoint_df["monthly_income"] = checkpoint_df["monthly_income"].fillna(checkpoint_df["monthly_income"].median())

        risk = gbm_survival.predict_risk(gbm, checkpoint_df, gbm_features)
        checkpoint_df["risk_score"] = risk
        checkpoint_df["checkpoint_month"] = checkpoint_month
        # Tercile tier boundaries recomputed at THIS checkpoint's own score distribution.
        checkpoint_df["tier"] = pd.qcut(risk, 3, labels=TIERS)
        rows.append(checkpoint_df[["employee_id", "checkpoint_month", "risk_score", "tier"]])

    return pd.concat(rows, ignore_index=True)


def build_sankey_links(checkpoint_scores: pd.DataFrame) -> pd.DataFrame:
    wide = checkpoint_scores.pivot(index="employee_id", columns="checkpoint_month", values="tier")
    rows = []
    for i in range(len(CHECKPOINT_MONTHS) - 1):
        m_from, m_to = CHECKPOINT_MONTHS[i], CHECKPOINT_MONTHS[i + 1]
        pair_counts = wide.groupby([m_from, m_to], observed=True).size()
        for (tier_from, tier_to), count in pair_counts.items():
            rows.append(
                {
                    "checkpoint_from": m_from,
                    "checkpoint_to": m_to,
                    "tier_from": tier_from,
                    "tier_to": tier_to,
                    "employee_count": int(count),
                }
            )
    return pd.DataFrame(rows)


def _write_table(engine, df: pd.DataFrame, table: str):
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY"))
    if not df.empty:
        df.to_sql(table, engine, if_exists="append", index=False)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = get_engine()

    checkpoint_scores = build_checkpoint_scores(engine)
    sankey_links = build_sankey_links(checkpoint_scores)

    checkpoint_scores.to_csv(os.path.join(OUT_DIR, "risk_migration_checkpoint_scores.csv"), index=False)
    sankey_links.to_csv(os.path.join(OUT_DIR, "risk_migration_sankey_links.csv"), index=False)

    _write_table(engine, checkpoint_scores, "attrition_risk_migration_checkpoints")
    _write_table(engine, sankey_links, "attrition_risk_migration_sankey")

    print(f"Checkpoint scores: {len(checkpoint_scores)} rows across {len(CHECKPOINT_MONTHS)} checkpoints")
    print(f"Sankey links: {len(sankey_links)} transitions")
    return {"checkpoint_scores": len(checkpoint_scores), "sankey_links": len(sankey_links)}


if __name__ == "__main__":
    run()
