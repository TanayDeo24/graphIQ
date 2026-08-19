"""Generate the synthetic time-varying tables that extend the static IBM HR
Analytics Attrition dataset into a 36-month workforce history:
comp_history, performance_reviews, benefits_enrollment.

The IBM dataset (data/raw/*.csv) is real, public, and loaded unmodified
into `employees` (columns renamed to snake_case only). Everything in this
module is fully synthetic and disclosed as such — see README "Data
provenance" section.

Design choices (documented here since the build spec leaves the exact
numbers to the implementation):

- SEED: all randomness is derived from a single seeded NumPy Generator so
  the generated dataset is reproducible run-to-run.
- REFERENCE window: every employee's synthetic history occupies months
  [1 .. window_length], where window_length = min(duration_months, 36).
  Employees with < 3 years of real tenure get a shorter (fully real)
  window; employees with >= 3 years get the full 36-month window,
  positioned as an arbitrary-but-consistent 3-year lookback. Calendar
  dates are anchored to WINDOW_START so the tables have real DATE values.
- comp_history is event-based (a row per change), not one row per month
  — that matches how a real compensation-history table would be modeled.
  The employee's actual `monthly_income` from the IBM dataset is treated
  as ground truth for "now"; the synthetic path is built backward from
  it so the final logged value always matches exactly.
- Raise magnitude: Normal(mean=5.5%, std=1.5%) truncated at 0, i.e. mass
  concentrated in the 3-8% band called for by the spec, never negative.
- Promotion bump magnitude: Uniform(8%, 15%) — the spec specifies raise
  magnitude but not promotion magnitude, so this is a documented
  assumption (promotions are modeled as a materially larger step than a
  routine raise).
- Train/test split (reused verbatim by src/models/attrition/evaluate.py
  via `assign_temporal_split`): temporal, based on window_length — this
  concentrates short-tenure (higher base-rate attrition) employees in
  train and long-tenure employees in test, which is a harder and more
  realistic generalization test than a random 80/20 split.
- Duration granularity (`duration_months`): the real IBM dataset's
  `YearsAtCompany` is annual-resolution only (integer years), which
  produces heavy ties in the survival ranking task if used directly as
  `tenure_years * 12` — see `assign_fine_grained_duration()` below and
  `validate_synthetic.py`'s distinct-duration-value check. `duration_months`
  is therefore a disclosed synthetic refinement, giving month-level
  precision the real dataset doesn't actually contain, derived from each
  employee's own already-generated comp_history/performance_reviews
  timeline rather than an independent random draw. The real annual value
  is never overwritten or hidden — it stays available as `tenure_years`
  throughout (unmodified IBM `YearsAtCompany`).
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

SEED = 42
WINDOW_START = pd.Timestamp("2022-01-01")
WINDOW_MONTHS = 36
TRAIN_TEST_SPLIT_MONTH = 24  # window_length <= 24 -> train, else -> test

RAISE_PCT_MEAN = 0.055
RAISE_PCT_STD = 0.015
PROMOTION_PCT_LOW = 0.08
PROMOTION_PCT_HIGH = 0.15

# Rough annual T&E-independent review cadence used elsewhere too.
REVIEW_INTERVAL_MONTHS = 6

TENURE_BAND_BINS = [-1, 2, 5, float("inf")]
TENURE_BAND_LABELS = ["0-2", "2-5", "5+"]

IBM_COLUMN_RENAME = {
    "EmployeeNumber": "employee_id",
    "Age": "age",
    "Attrition": "attrition_flag",
    "BusinessTravel": "business_travel",
    "DailyRate": "daily_rate",
    "Department": "department",
    "DistanceFromHome": "distance_from_home",
    "Education": "education",
    "EducationField": "education_field",
    "EmployeeCount": "employee_count",
    "EnvironmentSatisfaction": "environment_satisfaction",
    "Gender": "gender",
    "HourlyRate": "hourly_rate",
    "JobInvolvement": "job_involvement",
    "JobLevel": "job_level",
    "JobRole": "job_role",
    "JobSatisfaction": "job_satisfaction",
    "MaritalStatus": "marital_status",
    "MonthlyIncome": "monthly_income",
    "MonthlyRate": "monthly_rate",
    "NumCompaniesWorked": "num_companies_worked",
    "Over18": "over_18",
    "OverTime": "over_time",
    "PercentSalaryHike": "percent_salary_hike",
    "PerformanceRating": "performance_rating",
    "RelationshipSatisfaction": "relationship_satisfaction",
    "StandardHours": "standard_hours",
    "StockOptionLevel": "stock_option_level",
    "TotalWorkingYears": "total_working_years",
    "TrainingTimesLastYear": "training_times_last_year",
    "WorkLifeBalance": "work_life_balance",
    "YearsAtCompany": "tenure_years",
    "YearsInCurrentRole": "years_in_current_role",
    "YearsSinceLastPromotion": "years_since_last_promotion",
    "YearsWithCurrManager": "years_with_curr_manager",
}


def find_raw_csv(raw_dir: str) -> str:
    candidates = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {raw_dir}. Download the IBM HR Analytics Attrition "
            "dataset (Kaggle: pavansubhash/ibm-hr-analytics-attrition-dataset) into it first."
        )
    return candidates[0]


def load_employees(raw_dir: str) -> pd.DataFrame:
    df = pd.read_csv(find_raw_csv(raw_dir))
    df = df.rename(columns=IBM_COLUMN_RENAME)
    missing = set(IBM_COLUMN_RENAME.values()) - set(df.columns)
    if missing:
        raise ValueError(f"IBM dataset is missing expected columns after rename: {missing}")

    # Placeholder, coarse value (tenure_years * 12) — used only for sizing each
    # employee's synthetic history window in generate_all(). Overwritten there
    # with the month-level-resolution value (see assign_fine_grained_duration)
    # before this column is ever written out or used as a modeling target.
    df["duration_months"] = df["tenure_years"] * 12
    df["event_observed"] = df["attrition_flag"].eq("Yes")
    return df


def build_departments(employees: pd.DataFrame) -> pd.DataFrame:
    names = sorted(employees["department"].unique())
    return pd.DataFrame(
        {"department_id": range(1, len(names) + 1), "department_name": names}
    )


def window_length_months(duration_months: int) -> int:
    return max(1, min(int(duration_months), WINDOW_MONTHS))


def assign_temporal_split(duration_months: int) -> str:
    """Temporal train/test split shared by generation and evaluation code."""
    return "train" if window_length_months(duration_months) <= TRAIN_TEST_SPLIT_MONTH else "test"


def _tenure_band(tenure_years: float) -> str:
    idx = np.digitize([tenure_years], TENURE_BAND_BINS)[0] - 1
    idx = min(max(idx, 0), len(TENURE_BAND_LABELS) - 1)
    return TENURE_BAND_LABELS[idx]


def _sample_raise_pct(rng: np.random.Generator) -> float:
    a = (0.0 - RAISE_PCT_MEAN) / RAISE_PCT_STD
    return float(truncnorm.rvs(a, np.inf, loc=RAISE_PCT_MEAN, scale=RAISE_PCT_STD, random_state=rng))


def _comp_history_offsets(comp_history_df: pd.DataFrame) -> list:
    return [
        (d.year - WINDOW_START.year) * 12 + (d.month - WINDOW_START.month)
        for d in comp_history_df["effective_month"]
    ]


def assign_fine_grained_duration(
    tenure_years: int,
    comp_history_df: pd.DataFrame,
    employee_id: int,
) -> int:
    """Derive a month-level-resolution duration_months from this employee's
    own already-generated 36-month synthetic timeline, anchored to their
    last comp_history event — not an independent random draw.

    Real-world caveat, stated plainly: this does NOT recover the
    employee's actual real-world month of departure/last-observation —
    the real IBM dataset simply doesn't contain that information (only
    annual-resolution YearsAtCompany). What this produces is a
    reproducible, timeline-anchored month-level value that breaks the
    heavy ties an annual-only duration would otherwise create in the
    survival ranking task, while staying internally consistent with each
    employee's own generated events.

    Anchored to comp_history specifically, not performance_reviews: a
    review's *timing* follows a fixed, unjittered 6-month schedule (only
    its score is random), so when it happens to be an employee's latest
    event it collapses every employee sharing the same window length to
    the identical anchor month — exactly the tie problem this function
    exists to fix. comp_history raise/promotion timing carries real
    per-employee jitter (see generate_comp_history_for_employee), so it's
    used as the anchor's month component; the exact dollar amount at that
    event (itself a deterministic function of this employee's own
    randomly-sampled raise percentage) breaks residual ties within the
    same clamped month, so two employees who both land on "month 11" by
    jitter alone still get distinct spread from a genuinely-varying,
    timeline-derived value rather than an arbitrary independent draw.

    Falls back to a per-employee seeded draw only when comp_history has no
    variation to anchor to at all — i.e. tenure_years == 0, whose 1-month
    window contains just the single 'initial' row.

    For tenure_years == 0, duration_months = month_within_final_year - 1,
    i.e. somewhere in [0, 11] ("year 0" = less than one full year).
    For tenure_years >= 1, duration_months = (tenure_years - 1) * 12 +
    month_within_final_year, i.e. somewhere in the employee's reported
    final year of tenure.
    """
    comp_offsets = _comp_history_offsets(comp_history_df)

    if len(set(comp_offsets)) > 1:
        last_offset = max(comp_offsets)
        last_income = float(comp_history_df["monthly_income"].iloc[-1])
        income_cents = int(round(last_income * 100))
        month_within_final_year = ((last_offset * 97 + income_cents) % 12) + 1
    else:
        rng = np.random.default_rng(SEED + employee_id)
        month_within_final_year = int(rng.integers(1, 13))

    if tenure_years <= 0:
        return month_within_final_year - 1
    return (tenure_years - 1) * 12 + month_within_final_year


def generate_comp_history_for_employee(
    employee_id: int,
    target_income: float,
    window_len: int,
    is_promoted: bool,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Backward-consistent event-based compensation path ending at target_income."""
    num_raises = window_len // 12
    raise_months = []
    for i in range(1, num_raises + 1):
        jitter = int(rng.integers(-3, 4))
        month = min(max(i * 12 + jitter, 1), window_len - 1) if window_len > 1 else 0
        raise_months.append(month)

    promotion_month = None
    if is_promoted and window_len > 2:
        promotion_month = int(rng.integers(1, max(2, int(window_len * 0.6))))

    # De-duplicate/collision-resolve event months, keep within (0, window_len - 1].
    events = [(m, "raise") for m in raise_months]
    if promotion_month is not None:
        events.append((promotion_month, "promotion_adjustment"))
    seen_months = set()
    resolved = []
    for month, change_type in sorted(events, key=lambda x: x[0]):
        while month in seen_months and month < window_len - 1:
            month += 1
        if month <= 0 or month in seen_months or month > window_len - 1:
            continue
        seen_months.add(month)
        resolved.append((month, change_type))
    resolved.sort(key=lambda x: x[0])

    raise_pcts = [_sample_raise_pct(rng) for _, t in resolved if t == "raise"]
    promo_pct = rng.uniform(PROMOTION_PCT_LOW, PROMOTION_PCT_HIGH) if promotion_month is not None and any(
        t == "promotion_adjustment" for _, t in resolved
    ) else None

    total_multiplier = 1.0
    for pct in raise_pcts:
        total_multiplier *= 1.0 + pct
    if promo_pct is not None:
        total_multiplier *= 1.0 + promo_pct
    base_income = target_income / total_multiplier if total_multiplier > 0 else target_income

    rows = [{"employee_id": employee_id, "month_offset": 0, "monthly_income": round(base_income, 2),
             "change_type": "initial"}]

    running_income = base_income
    raise_iter = iter(raise_pcts)
    for i, (month, change_type) in enumerate(resolved):
        is_last_event = i == len(resolved) - 1
        if change_type == "raise":
            running_income *= 1.0 + next(raise_iter)
        else:
            running_income *= 1.0 + promo_pct
        income_value = target_income if is_last_event else running_income
        rows.append(
            {
                "employee_id": employee_id,
                "month_offset": month,
                "monthly_income": round(income_value, 2),
                "change_type": change_type,
            }
        )

    if len(resolved) == 0:
        rows[0]["monthly_income"] = round(target_income, 2)

    out = pd.DataFrame(rows)
    out["effective_month"] = out["month_offset"].apply(
        lambda m: (WINDOW_START + pd.DateOffset(months=int(m))).normalize()
    )
    return out[["employee_id", "effective_month", "monthly_income", "change_type"]]


def generate_performance_reviews_for_employee(
    employee_id: int, performance_rating_4pt: int, window_len: int, rng: np.random.Generator
) -> pd.DataFrame:
    center_5pt = 1.0 + (performance_rating_4pt - 1) * 4.0 / 3.0
    review_months = list(range(REVIEW_INTERVAL_MONTHS, window_len + 1, REVIEW_INTERVAL_MONTHS))
    rows = []
    for month in review_months:
        score = center_5pt + rng.normal(0, 0.4)
        score = float(np.clip(score, 1.0, 5.0))
        rows.append(
            {
                "employee_id": employee_id,
                "review_month": (WINDOW_START + pd.DateOffset(months=month)).normalize(),
                "review_score": round(score, 2),
            }
        )
    return pd.DataFrame(rows, columns=["employee_id", "review_month", "review_score"])


def assign_benefits_tier(job_level: int, monthly_income: float, income_p33: float, income_p66: float,
                          rng: np.random.Generator) -> str:
    score = 0
    score += {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}.get(int(job_level), 1)
    score += 0 if monthly_income < income_p33 else (1 if monthly_income < income_p66 else 2)

    if score >= 3:
        probs = [0.10, 0.30, 0.60]
    elif score >= 1:
        probs = [0.25, 0.50, 0.25]
    else:
        probs = [0.60, 0.35, 0.05]
    return str(rng.choice(["basic", "standard", "premium"], p=probs))


def generate_all(raw_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    employees = load_employees(raw_dir)
    departments = build_departments(employees)
    employees = employees.merge(
        departments, left_on="department", right_on="department_name", how="left"
    ).drop(columns=["department_name"])

    tenure_band_avg_job_level = (
        employees.assign(tenure_band=employees["tenure_years"].apply(_tenure_band))
        .groupby("tenure_band")["job_level"]
        .mean()
    )

    income_p33, income_p66 = employees["monthly_income"].quantile([0.33, 0.66])

    comp_history_frames = []
    performance_frames = []
    benefits_rows = []
    fine_grained_durations = []

    for row in employees.itertuples(index=False):
        # Window sizing uses the coarse tenure_years * 12 value (how much history to
        # synthesize is a separate concern from the survival target's precision) — see
        # assign_fine_grained_duration() below, which derives the actual duration_months
        # used for modeling from the events generated here, after they're generated.
        window_len = window_length_months(row.duration_months)
        band_avg = tenure_band_avg_job_level.get(_tenure_band(row.tenure_years), employees["job_level"].mean())
        is_above_band_avg = row.job_level > band_avg
        promotion_prob = 0.60 if is_above_band_avg else 0.15
        is_promoted = bool(rng.random() < promotion_prob)

        comp_history_df = generate_comp_history_for_employee(
            row.employee_id, float(row.monthly_income), window_len, is_promoted, rng
        )
        performance_df = generate_performance_reviews_for_employee(
            row.employee_id, int(row.performance_rating), window_len, rng
        )
        comp_history_frames.append(comp_history_df)
        performance_frames.append(performance_df)
        fine_grained_durations.append(
            assign_fine_grained_duration(int(row.tenure_years), comp_history_df, row.employee_id)
        )

        benefits_rows.append(
            {
                "employee_id": row.employee_id,
                "plan_tier": assign_benefits_tier(
                    row.job_level, row.monthly_income, income_p33, income_p66, rng
                ),
                "enrolled_month": WINDOW_START,
            }
        )

    comp_history = pd.concat(comp_history_frames, ignore_index=True)
    performance_reviews = pd.concat(
        [f for f in performance_frames if not f.empty], ignore_index=True
    )
    benefits_enrollment = pd.DataFrame(benefits_rows)

    # Replace the coarse (tenure_years * 12) duration used for window sizing above with
    # the month-level-resolution value used as the actual survival modeling target.
    employees["duration_months"] = fine_grained_durations
    employees["data_split"] = employees["duration_months"].apply(assign_temporal_split)

    employees.to_csv(os.path.join(out_dir, "employees.csv"), index=False)
    departments.to_csv(os.path.join(out_dir, "departments.csv"), index=False)
    comp_history.to_csv(os.path.join(out_dir, "comp_history.csv"), index=False)
    performance_reviews.to_csv(os.path.join(out_dir, "performance_reviews.csv"), index=False)
    benefits_enrollment.to_csv(os.path.join(out_dir, "benefits_enrollment.csv"), index=False)

    return {
        "employees": len(employees),
        "departments": len(departments),
        "comp_history_rows": len(comp_history),
        "performance_review_rows": len(performance_reviews),
        "benefits_enrollment_rows": len(benefits_enrollment),
        "train_employees": int((employees["data_split"] == "train").sum()),
        "test_employees": int((employees["data_split"] == "test").sum()),
    }


if __name__ == "__main__":
    stats = generate_all(raw_dir="data/raw", out_dir="data/generated")
    for key, value in stats.items():
        print(f"{key}: {value}")
