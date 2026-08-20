"""Shared transaction-level feature engineering for the spend anomaly
detectors (Isolation Forest, autoencoder). CUSUM operates on a separate
monthly-aggregate view (see cusum.py) since it's a drift detector, not a
point-anomaly detector.

Ground truth columns (is_injected_anomaly, anomaly_type) are loaded for
evaluation only and are explicitly excluded from FEATURE_COLUMNS — no
model in this component ever trains on them.

COHORT-LEAKAGE AUDIT (see README's "Spend" results section for the full
writeup): the anomaly injection design concentrates 70% of injected
anomalies among a designated 10% of employees
(spend_generator.HOT_EMPLOYEE_FRACTION) — a real risk that a detector
could partly learn "which employees are in the flagged cohort" instead of
genuinely detecting anomalous spend *patterns*. Two things were checked:

1. Column-level leakage: `hot_employees`/cohort membership is a local
   variable in spend_generator.py, never written to expense_transactions
   or any other persisted table/column FEATURE_COLUMNS could read from.
   Confirmed absent from every feature by construction, not just by
   correlation — there's no column for it to leak through. (The
   `_hot_employees_audit.csv` spend_generator.py now also writes is an
   audit-only artifact used solely by check_no_cohort_leakage() below; it
   is never loaded by build_feature_frame() or fed to any detector.)
2. Statistical leakage: could the FEATURE VALUES nonetheless correlate
   with cohort membership even without a literal column for it (e.g. if
   hot employees' baselines were computed differently)? Checked via
   check_no_cohort_leakage(), the same >0.95-correlation guardrail
   pattern as the attrition side's check_no_leakage().

DIAGNOSED-AND-FIXED CONTAMINATION (same bug class as CUSUM's, found by
checking exactly the mechanism the audit asked about): `deviation_from_
own_mean` and `deviation_from_dept_mean` were computed via
groupby(...).transform("mean"/"std") over each employee/department-
category's *entire* transaction history, including the anomalous months
themselves — self-referential, diluting the very deviation being
measured, for the same reason CUSUM's contaminated baseline was a bug.
Fixed the same way: robust (median/MAD) center/scale instead of mean/std.
Not filtered using is_injected_anomaly instead, deliberately — a real
detector doesn't know in advance which of an employee's own transactions
are anomalous, so using the ground-truth label to clean the baseline
would itself be a *worse* form of leakage than the bug it would fix.
"""

import numpy as np
import pandas as pd

from src.db.connection import get_engine

CATEGORIES = ["travel", "software_saas", "meals", "office_supplies", "client_entertainment", "other"]

FEATURE_COLUMNS = [
    "amount_usd",
    "deviation_from_own_mean",
    "deviation_from_dept_mean",
    "day_of_week",
    "trailing_30d_frequency",
] + [f"cat_{c}" for c in CATEGORIES]

MAD_TO_STD = 1.4826
COHORT_LEAKAGE_CORR_THRESHOLD = 0.95


ROBUST_DEVIATION_CLIP = 20.0


def _robust_deviation(df: pd.DataFrame, group_cols: list) -> pd.Series:
    """(value - group median) / (1.4826 * group MAD), clamped to a std floor
    of 1.0 — the same robust center/scale used by cusum.py's
    cusum_statistic_series, applied here to per-transaction deviation
    features instead of monthly-aggregate series. See module docstring for
    why mean/std was contaminated and MAD is the fix, not a ground-truth
    filter.

    Clipped to +/-ROBUST_DEVIATION_CLIP: a handful of low-transaction-count
    (employee/department, category) groups have a near-zero MAD, which
    turns an otherwise-modest dollar difference into an enormous z-score
    (observed as high as ~52,000 pre-clip) — a numerical artifact of a
    near-zero denominator, not a genuinely 52,000-sigma anomaly. Left
    unclipped, these dominate the autoencoder's MSE training loss and
    measurably hurt its reconstruction quality on everything else (verified:
    clipping recovers autoencoder PR-AUC that an unclipped version lost —
    see README). 20 is generous relative to the ~1st/99th-percentile spread
    of the un-clipped distribution (|v| > 10 for under 2% of transactions),
    so it only bites the true numerical-artifact tail.
    """
    group = df.groupby(group_cols)["amount_usd"]
    center = group.transform("median")
    abs_dev = (df["amount_usd"] - center).abs()
    scale = MAD_TO_STD * abs_dev.groupby([df[c] for c in group_cols]).transform("median")
    scale = scale.fillna(1.0).replace(0, 1.0)
    deviation = (df["amount_usd"] - center) / scale
    return deviation.clip(-ROBUST_DEVIATION_CLIP, ROBUST_DEVIATION_CLIP)


def load_transactions(variant: str = "5pct", engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    if variant == "5pct":
        df = pd.read_sql(
            "SELECT transaction_id, employee_id, department_id, transaction_date, merchant_category, "
            "amount_usd, is_injected_anomaly, anomaly_type FROM expense_transactions",
            engine,
        )
    else:
        df = pd.read_csv(f"data/generated/expense_transactions_{variant}.csv", parse_dates=["transaction_date"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def _trailing_30d_frequency(group: pd.DataFrame) -> pd.Series:
    """Count of transactions (incl. current) within the trailing 30 days,
    per employee. Implemented with an explicit two-pointer scan (rather
    than pandas' groupby().rolling(), whose MultiIndex result order does
    not line up positionally with the original frame) so it stays safely
    aligned when assigned back onto the original row order."""
    dates = group["transaction_date"].values.astype("datetime64[ns]")
    counts = np.zeros(len(dates), dtype=int)
    window = np.timedelta64(30, "D")
    start = 0
    for i in range(len(dates)):
        while dates[i] - dates[start] > window:
            start += 1
        counts[i] = i - start + 1
    return pd.Series(counts, index=group.index)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("transaction_date").reset_index(drop=True)

    # Robust (median/MAD) deviation, not mean/std -- see module docstring for
    # the contamination bug this fixes (same class as CUSUM's).
    df["deviation_from_own_mean"] = _robust_deviation(df, ["employee_id", "merchant_category"])
    df["deviation_from_dept_mean"] = _robust_deviation(df, ["department_id", "merchant_category"])

    df["day_of_week"] = df["transaction_date"].dt.dayofweek

    df["trailing_30d_frequency"] = df.groupby("employee_id", group_keys=False).apply(
        _trailing_30d_frequency, include_groups=False
    )

    for c in CATEGORIES:
        df[f"cat_{c}"] = (df["merchant_category"] == c).astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].astype(float).values


def check_no_cohort_leakage(
    df: pd.DataFrame,
    audit_path: str = "data/generated/_hot_employees_audit.csv",
    threshold: float = COHORT_LEAKAGE_CORR_THRESHOLD,
) -> None:
    """Automated guardrail: fails if any feature in FEATURE_COLUMNS
    correlates with designated-cohort membership above `threshold`. Mirrors
    check_no_leakage() on the attrition side. `audit_path` is written by
    spend_generator.py purely for this check — never loaded by
    build_feature_frame() or fed to any detector. See module docstring for
    the full cohort-leakage audit writeup.
    """
    cohort = pd.read_csv(audit_path)
    merged = df.merge(cohort, on="employee_id", how="left")
    if merged["is_designated_cohort_member"].isna().any():
        raise AssertionError(f"Cohort audit file at {audit_path} is missing employee_id coverage.")
    cohort_flag = merged["is_designated_cohort_member"].astype(float)

    violations = []
    for col in FEATURE_COLUMNS:
        values = merged[col].astype(float)
        if values.std() == 0 or cohort_flag.std() == 0:
            continue
        corr = float(values.corr(cohort_flag))
        if abs(corr) > threshold:
            violations.append((col, corr))

    if violations:
        raise AssertionError(
            f"Cohort-leakage guardrail failed: feature(s) exceed |corr| > {threshold} "
            f"with is_designated_cohort_member: {violations}"
        )
