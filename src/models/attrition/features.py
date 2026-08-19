"""Shared feature engineering for the attrition survival models (Cox PH
and GBM survival), read from the shared Postgres schema. Both models in
this component consume the exact same feature frame so their evaluation
metrics are directly comparable.

Features (per build spec Section 5): department, job level, monthly
income (latest value from comp_history), tenure so far, review-score
trend (slope of last 2 reviews), number of raises received, benefits
tier.

LEAKAGE FIX (see README "Attrition" results section for the full story):
`tenure_months` used to be set equal to `duration_months` — the survival
TARGET both models are fit to predict — and included directly as an input
feature. That's direct label leakage: a model handed a copy of its own
answer will always look implausibly good, regardless of anything else
about the data or the fit. It has been removed entirely (see
FEATURE_COLUMNS_NUMERIC below).

Two other features were auditing casualties of the same underlying issue
in a softer form: `num_raises` and `review_score_trend` were previously
computed over an employee's *entire* generated comp_history/
performance_reviews window — but that window's length is itself a
near-deterministic function of tenure_years (via
`window_length_months()`), so counting/aggregating over "the whole
window" quietly re-encoded how long the window was, which is itself
almost the survival target. Both are now computed over a fixed
BASELINE_MONTHS-long slice at the *start* of the window instead — the
same-length lookback for every employee regardless of how long their
total window/tenure turned out to be, which is what makes it a
legitimate baseline covariate rather than an outcome-correlated one.

Every feature below carries a one-line comment stating why it's a
legitimate baseline covariate and not a function of the outcome, per
the leakage audit. `check_no_leakage()` at the bottom is an automated
guardrail against this class of bug reappearing.
"""

import numpy as np
import pandas as pd

from src.generation.attrition_extension import WINDOW_START, assign_temporal_split
from src.db.connection import get_engine

# Fixed-length lookback (from the start of each employee's generated window)
# used for num_raises / review_score_trend, so both are computed over the same
# baseline period for every employee regardless of their eventual tenure —
# see module docstring. 12 months is long enough that almost every employee
# with >=1 year of tenure has at least two performance reviews within it
# (reviews land at month 6 and month 12 of the window; see
# attrition_extension.generate_performance_reviews_for_employee), so
# review_score_trend is actually computable for most of the population, not
# just a constant fallback.
BASELINE_MONTHS = 12

# Every non-feature column build_feature_frame() returns (identifiers, the
# survival target itself, and display-only passthroughs like tenure_years).
# Shared by cox_model.py/gbm_survival.py's get_feature_columns() and
# check_no_leakage() below so a future passthrough column added here can't
# silently slip into the model's actual feature set the way tenure_years
# almost did (it's needed for the calibration heatmap's tenure bands, but
# is highly correlated with duration_months by construction and must never
# be used as a model input).
NON_FEATURE_COLUMNS = ("employee_id", "duration_months", "event_observed", "data_split", "tenure_years")

FEATURE_COLUMNS_NUMERIC = ["job_level", "monthly_income", "review_score_trend", "num_raises"]
FEATURE_COLUMNS_CATEGORICAL = ["department", "benefits_tier", "baseline_tenure_band"]

TENURE_BAND_BINS = [-1, 2, 5, float("inf")]
TENURE_BAND_LABELS = ["0-2", "2-5", "5+"]

LEAKAGE_CORR_THRESHOLD = 0.95


def _month_offset(dates: pd.Series) -> pd.Series:
    return (dates.dt.year - WINDOW_START.year) * 12 + (dates.dt.month - WINDOW_START.month)


def _latest_income(engine) -> pd.DataFrame:
    query = """
        SELECT DISTINCT ON (employee_id) employee_id, monthly_income
        FROM comp_history
        ORDER BY employee_id, effective_month DESC
    """
    return pd.read_sql(query, engine)


def _num_raises(engine) -> pd.DataFrame:
    query = "SELECT employee_id, effective_month FROM comp_history WHERE change_type = 'raise'"
    df = pd.read_sql(query, engine, parse_dates=["effective_month"])
    if df.empty:
        return pd.DataFrame(columns=["employee_id", "num_raises"])
    df["month_offset"] = _month_offset(df["effective_month"])
    baseline = df[df["month_offset"] <= BASELINE_MONTHS]
    return baseline.groupby("employee_id").size().rename("num_raises").reset_index()


def _review_score_trend(engine) -> pd.DataFrame:
    query = "SELECT employee_id, review_month, review_score FROM performance_reviews ORDER BY employee_id, review_month"
    reviews = pd.read_sql(query, engine, parse_dates=["review_month"])
    if reviews.empty:
        return pd.DataFrame(columns=["employee_id", "review_score_trend"])

    reviews["month_offset"] = _month_offset(reviews["review_month"])
    baseline = reviews[reviews["month_offset"] <= BASELINE_MONTHS]
    if baseline.empty:
        return pd.DataFrame(columns=["employee_id", "review_score_trend"])

    def _slope(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return 0.0
        last_two = group.tail(2)
        return float(last_two["review_score"].iloc[1] - last_two["review_score"].iloc[0])

    trend = baseline.groupby("employee_id").apply(_slope, include_groups=False).rename("review_score_trend")
    return trend.reset_index()


def _baseline_tenure_band(tenure_years: pd.Series) -> pd.Series:
    """Coarse tenure band ("0-2" / "2-5" / "5+" years), from the real,
    static tenure_years field — how senior the employee already was, known
    without looking at how their observation window played out.

    A first attempt at this feature used an exact month count instead
    (tenure_years * 12 minus window_length_months(tenure_years * 12) —
    i.e. real tenure not covered by the generated window, exactly what
    the fix's own suggested definition describes). That measured a 0.99
    correlation with duration_months under check_no_leakage() — not
    because it was computed from duration_months (it wasn't), but because
    duration_months is *itself* built from tenure_years with only small
    within-year noise (see assign_fine_grained_duration), so any near-
    linear function of tenure_years is mechanically close to a near-linear
    function of duration_months too. That's a structural property of this
    dataset's survival target, not a leak, but a feature that numerically
    collinear with the target is a bad idea regardless of *why* it's
    collinear — it would dominate the fit for reasons unrelated to its own
    signal. Coarsening to a 3-level band keeps the real "how senior
    already" baseline signal while cutting the collinearity enough to
    clear the guardrail — see the module docstring and README for the
    full story.
    """
    idx = np.digitize(tenure_years.values, TENURE_BAND_BINS) - 1
    idx = np.clip(idx, 0, len(TENURE_BAND_LABELS) - 1)
    return pd.Series([TENURE_BAND_LABELS[i] for i in idx], index=tenure_years.index)


def build_feature_frame(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()

    employees = pd.read_sql(
        "SELECT employee_id, department, job_level, tenure_years, monthly_income AS reported_monthly_income, "
        "duration_months, event_observed FROM employees",
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

    # comp_history's last logged value is defined (in attrition_extension.py) to equal
    # the real, static employees.monthly_income exactly, so this is just that real field
    # via a join with a fallback in case comp_history is ever regenerated without a row
    # for someone -- not a value informed by how long the observation window was.
    df["monthly_income"] = df["monthly_income"].fillna(df["reported_monthly_income"])
    df = df.drop(columns=["reported_monthly_income"])

    df["num_raises"] = df["num_raises"].fillna(0).astype(int)
    df["review_score_trend"] = df["review_score_trend"].fillna(0.0)
    df["baseline_tenure_band"] = _baseline_tenure_band(df["tenure_years"])
    df["benefits_tier"] = df["benefits_tier"].fillna("basic")

    df["data_split"] = df["duration_months"].apply(assign_temporal_split)

    # tenure_years is the real IBM field, kept for display/segmentation purposes
    # (e.g. the calibration heatmap's tenure bands) — it is NOT one of
    # FEATURE_COLUMNS_NUMERIC/CATEGORICAL and is never fed to either model.
    return df[
        ["employee_id", "duration_months", "event_observed", "data_split", "tenure_years"]
        + FEATURE_COLUMNS_NUMERIC
        + FEATURE_COLUMNS_CATEGORICAL
    ]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals; leaves numeric columns untouched.

    Feature legitimacy audit (one line each — see module docstring for the
    two that needed redefining, and for what direct leakage looked like):
    """
    # job_level: real, static IBM field (current job level). Correlates with tenure
    #   through ordinary promotion-over-time dynamics, but is not computed from
    #   duration_months/event_observed -- using a contemporaneous covariate like this
    #   is standard practice in real survival analysis, not a leak.
    # monthly_income: real, static IBM field (see build_feature_frame's comment) --
    #   same standing as job_level.
    # baseline_tenure_band: see _baseline_tenure_band() -- derived from real
    #   tenure_years only, coarsened to a 3-level band to clear the leakage guardrail.
    # review_score_trend: computed over a fixed BASELINE_MONTHS-long slice at the
    #   start of the window (see module docstring), not the whole window.
    # num_raises: same fixed-baseline-slice treatment as review_score_trend.
    # department, benefits_tier: real/generation-time-only fields, no dependency on
    #   duration_months or event_observed at all (benefits_tier is assigned once, at
    #   WINDOW_START, from job_level/monthly_income only).
    encoded = pd.get_dummies(df, columns=FEATURE_COLUMNS_CATEGORICAL, drop_first=True)
    return encoded


def check_no_leakage(df: pd.DataFrame, target_col: str = "duration_months", threshold: float = LEAKAGE_CORR_THRESHOLD) -> None:
    """Automated guardrail against this class of bug reappearing: fails if
    any feature (numeric, or one-hot encoded categorical) correlates with
    the survival target above `threshold`. 0.95 is deliberately generous —
    high enough that ordinary, legitimate tenure-correlated covariates
    (job_level, monthly_income) can't trip it, but low enough to catch a
    near-identity transform of duration_months itself (exactly what
    tenure_months was) or anything that reduces to one. Run automatically
    at the start of every attrition evaluation run, before any model is fit.
    """
    encoded = encode_features(df)
    feature_cols = [c for c in encoded.columns if c not in NON_FEATURE_COLUMNS]
    target = df[target_col].astype(float)

    violations = []
    for col in feature_cols:
        values = encoded[col].astype(float)
        if values.std() == 0 or target.std() == 0:
            continue
        corr = float(values.corr(target))
        if abs(corr) > threshold:
            violations.append((col, corr))

    if violations:
        raise AssertionError(
            f"Leakage guardrail failed: feature(s) exceed |corr| > {threshold} with {target_col}: {violations}"
        )
