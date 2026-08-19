"""Cohort-level (department x category x month) CUSUM drift detector —
a second, additional variant alongside the original per-employee CUSUM in
cusum.py, added to test the granularity-mismatch hypothesis documented in
the README's "Spend" results section: the project's original design intent
was cohort-level drift detection ("a department's aggregate spend velocity
accelerating over time"), not single-employee detection, and per-employee
monthly aggregates are inherently noisy (only a handful of transactions
feed each point).

Does NOT replace the per-employee variant — both are evaluated side by
side in the detector comparison table so the granularity effect itself is
visible, not hidden behind a silent swap.

Uses the same robust CUSUM math (median/MAD, k=0.5, h=5) as the
per-employee variant — see cusum.py's module docstring for why.
"""

import numpy as np
import pandas as pd

from src.models.spend.cusum import H_SIGMA, K_SIGMA, _monthly_series, cusum_statistic_series


def compute_cohort_cusum_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a (department_id, merchant_category, month, cusum_statistic, flagged) frame."""
    monthly = _monthly_series(df, ["department_id", "merchant_category"])
    monthly = monthly.sort_values(["department_id", "merchant_category", "month"])

    stats = monthly.groupby(["department_id", "merchant_category"])["monthly_total"].transform(
        lambda s: pd.Series(cusum_statistic_series(s.values), index=s.index)
    )
    monthly["cusum_statistic"] = stats
    monthly["flagged"] = monthly["cusum_statistic"] > H_SIGMA
    return monthly


def map_flags_to_transactions(df: pd.DataFrame, monthly_flags: pd.DataFrame) -> pd.DataFrame:
    """Every transaction in a flagged department+category+month inherits that
    cohort's score — including coworkers' entirely ordinary transactions in
    the same department/category/month as a genuinely drifting colleague.
    That's the real, honest precision cost of cohort-level aggregation
    (reported in the README), not a bug to paper over."""
    df = df.copy()
    df["month"] = df["transaction_date"].dt.to_period("M").dt.to_timestamp()
    monthly_flags = monthly_flags.copy()
    monthly_flags["month"] = pd.to_datetime(monthly_flags["month"])
    merged = df.merge(
        monthly_flags[["department_id", "merchant_category", "month", "cusum_statistic", "flagged"]],
        on=["department_id", "merchant_category", "month"],
        how="left",
    )
    merged["flagged"] = merged["flagged"].fillna(False)
    merged["cusum_statistic"] = merged["cusum_statistic"].fillna(0.0)
    return merged[["flagged", "cusum_statistic"]]
