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

DIAGNOSED FIX — contaminated baseline estimation (see README's "Spend"
results section for the full investigation and numbers): the original
implementation estimated each series' mean/std from the *entire* series,
including the drift months themselves. Measured empirically against a
clean baseline (non-anomalous months only): that self-referential std was
inflated by a median of ~3.5x (mean ~8.8x) versus the clean baseline std
for actual injected slow_drift cases — the drift was diluting the very
"sigma" it was being measured against, actively working against
detection. This is standard statistical process control practice, not
new to this project: control limits are supposed to be estimated from
an in-control reference, not from data that may itself contain the
shift being monitored for.

The fix: center/scale (mean/std) are now the median and MAD-derived
robust standard deviation (median absolute deviation x 1.4826, the
standard scale-consistent conversion for normally-distributed data)
instead of the raw mean/std. This is a like-for-like substitution of a
better estimator for the same theoretical quantities k and h are
expressed in units of — k=0.5 and h=5 are UNCHANGED, no new free
parameters were introduced or searched, so this doesn't need the
tuning-dataset anti-overfitting procedure used elsewhere in this
investigation (see spend_generator's tuning-dataset note if a k/h
retune is ever attempted). Measured effect: median contamination ratio
dropped from ~3.5x to ~1.37x (not fully eliminated — MAD's robustness
breaks down as the contaminated fraction of a series approaches its
~50% breakdown point, and some slow_drift cases span a large share of
an employee's short window) but a real, principled improvement.

Operates on monthly-aggregated spend per (employee_id, merchant_category)
— a genuinely different granularity from the transaction-level Isolation
Forest / autoencoder detectors. That was the design intent for why this
detector should catch slow_drift better than a point-anomaly detector;
the standalone per-detector evaluation in src/models/spend/evaluate.py
did not bear that out even after this baseline-estimation fix. See
cohort_cusum.py for the granularity-mismatch hypothesis this project
also investigated, and the README for the full honest comparison.
"""

import numpy as np
import pandas as pd

K_SIGMA = 0.5
H_SIGMA = 5.0

MAD_TO_STD = 1.4826


def _monthly_series(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    monthly = (
        df.groupby(group_cols + [pd.Grouper(key="transaction_date", freq="MS")])["amount_usd"]
        .sum()
        .rename("monthly_total")
        .reset_index()
        .rename(columns={"transaction_date": "month"})
    )
    return monthly


def cusum_statistic_series(values: np.ndarray) -> np.ndarray:
    """Shared CUSUM math, reused by both the per-employee (this module)
    and cohort-level (cohort_cusum.py) variants. Robust (median/MAD)
    center and scale — see module docstring."""
    center = np.median(values)
    scale = MAD_TO_STD * np.median(np.abs(values - center))
    scale = scale if scale > 0 else 1.0
    z = (values - center) / scale

    c_plus = np.zeros(len(values))
    running = 0.0
    for i, zi in enumerate(z):
        running = max(0.0, running + zi - K_SIGMA)
        c_plus[i] = running
    return c_plus


def compute_cusum_flags(df: pd.DataFrame, h: float = H_SIGMA) -> pd.DataFrame:
    """Returns a (employee_id, merchant_category, month, cusum_statistic, flagged) frame.

    `h` defaults to the module's documented H_SIGMA=5 (a standard, general-
    purpose SPC default) but can be overridden — see
    src/models/spend/tune_cusum.py's H_SIGMA_TUNED, derived via that
    module's documented tuning-dataset procedure specifically for this
    project's generated data, and used by evaluate.py for the actual
    reported pipeline run. `cusum_statistic` itself (and therefore PR-AUC,
    which only depends on its ranking) is unaffected by h — h only gates
    the boolean `flagged` column, which feeds the drift-delay/detection-
    timing metrics and the `cusum_flag` display field.
    """
    monthly = _monthly_series(df, ["employee_id", "merchant_category"])
    monthly = monthly.sort_values(["employee_id", "merchant_category", "month"])

    stats = monthly.groupby(["employee_id", "merchant_category"])["monthly_total"].transform(
        lambda s: pd.Series(cusum_statistic_series(s.values), index=s.index)
    )
    monthly["cusum_statistic"] = stats
    monthly["flagged"] = monthly["cusum_statistic"] > h
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
