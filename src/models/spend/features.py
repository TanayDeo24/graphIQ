"""Shared transaction-level feature engineering for the spend anomaly
detectors (Isolation Forest, autoencoder). CUSUM operates on a separate
monthly-aggregate view (see cusum.py) since it's a drift detector, not a
point-anomaly detector.

Ground truth columns (is_injected_anomaly, anomaly_type) are loaded for
evaluation only and are explicitly excluded from FEATURE_COLUMNS — no
model in this component ever trains on them.
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

    own_mean = df.groupby(["employee_id", "merchant_category"])["amount_usd"].transform("mean")
    own_std = df.groupby(["employee_id", "merchant_category"])["amount_usd"].transform("std").fillna(1.0)
    own_std = own_std.replace(0, 1.0)
    df["deviation_from_own_mean"] = (df["amount_usd"] - own_mean) / own_std

    dept_mean = df.groupby(["department_id", "merchant_category"])["amount_usd"].transform("mean")
    dept_std = df.groupby(["department_id", "merchant_category"])["amount_usd"].transform("std").fillna(1.0)
    dept_std = dept_std.replace(0, 1.0)
    df["deviation_from_dept_mean"] = (df["amount_usd"] - dept_mean) / dept_std

    df["day_of_week"] = df["transaction_date"].dt.dayofweek

    df["trailing_30d_frequency"] = df.groupby("employee_id", group_keys=False).apply(
        _trailing_30d_frequency, include_groups=False
    )

    for c in CATEGORIES:
        df[f"cat_{c}"] = (df["merchant_category"] == c).astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].astype(float).values
