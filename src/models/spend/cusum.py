"""Manual CUSUM drift detector for employee x category monthly spend.

No library — implemented directly per the build spec. Constants are
deliberate, Western-Electric-style tuning choices (not magic numbers):

- k = 0.5 (in standard-deviation units): the "slack" — small deviations
  smaller than half a sigma don't accumulate, so routine noise doesn't
  trigger false alarms.
- h = 5 (in standard-deviation units): the control limit. A one-sided
  CUSUM with k=0.5*sigma and h=5*sigma is a standard, well-studied
  tuning that detects a sustained ~1*sigma mean shift within a handful
  of periods while keeping the false-alarm rate low — appropriate here
  since slow_drift anomalies are specifically designed to be
  undetectable from any single transaction and only visible as a
  sustained shift over several months.

Operates on monthly-aggregated spend per (employee_id, merchant_category)
— a genuinely different granularity from the transaction-level Isolation
Forest / autoencoder detectors. That was the design intent for why this
detector should catch slow_drift better than a point-anomaly detector; the
standalone per-detector evaluation in src/models/spend/evaluate.py does not
bear that out with this tuning (k=0.5, h=5) against this generation's
slow_drift parameters — CUSUM's own slow_drift PR-AUC comes out lower than
both Isolation Forest's and the autoencoder's. See the README's "Spend"
results section for the honest comparison table and numbers.
"""

import numpy as np
import pandas as pd

K_SIGMA = 0.5
H_SIGMA = 5.0


def _monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["employee_id", "merchant_category", pd.Grouper(key="transaction_date", freq="MS")])[
            "amount_usd"
        ]
        .sum()
        .rename("monthly_total")
        .reset_index()
        .rename(columns={"transaction_date": "month"})
    )
    return monthly


def _cusum_one_series(values: np.ndarray) -> np.ndarray:
    mean, std = values.mean(), values.std()
    std = std if std > 0 else 1.0
    z = (values - mean) / std

    c_plus = np.zeros(len(values))
    running = 0.0
    for i, zi in enumerate(z):
        running = max(0.0, running + zi - K_SIGMA)
        c_plus[i] = running
    return c_plus


def compute_cusum_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a (employee_id, merchant_category, month, cusum_statistic, flagged) frame."""
    monthly = _monthly_series(df)
    monthly = monthly.sort_values(["employee_id", "merchant_category", "month"])

    stats = monthly.groupby(["employee_id", "merchant_category"])["monthly_total"].transform(
        lambda s: pd.Series(_cusum_one_series(s.values), index=s.index)
    )
    monthly["cusum_statistic"] = stats
    monthly["flagged"] = monthly["cusum_statistic"] > H_SIGMA
    return monthly


def map_flags_to_transactions(df: pd.DataFrame, monthly_flags: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["transaction_date"].dt.to_period("M").dt.to_timestamp()
    monthly_flags = monthly_flags.copy()
    monthly_flags["month"] = pd.to_datetime(monthly_flags["month"])
    merged = df.merge(
        monthly_flags[["employee_id", "merchant_category", "month", "cusum_statistic", "flagged"]],
        on=["employee_id", "merchant_category", "month"],
        how="left",
    )
    merged["flagged"] = merged["flagged"].fillna(False)
    merged["cusum_statistic"] = merged["cusum_statistic"].fillna(0.0)
    return merged[["flagged", "cusum_statistic"]]
