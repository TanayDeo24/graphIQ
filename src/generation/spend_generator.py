"""Generate fully synthetic expense_transactions data for the same 1,470
employees and the same 36-month window used by comp_history
(src/generation/attrition_extension.py) — this is what keeps the two
analyses on one unified schema rather than two disconnected datasets.

All numbers in this module are synthetic and disclosed as such. Nothing
here is drawn from or claims to represent any real company's spend data.

Design choices, documented per the build spec's requirement to ground
"invented" numbers in stated reasoning rather than leaving them as magic
constants:

- Transaction frequency (Poisson lambda per department, per month):
  Sales=11 (travel-heavy, frequent client-facing spend), R&D=8.5
  (moderate — mostly software subscriptions + occasional travel),
  HR=7 (lowest — mostly recurring/internal spend). These are illustrative
  values chosen to land the total dataset in the "roughly 400,000+
  transactions" range the spec targets for 1,470 employees over a 36-month
  window, loosely anchored to commonly cited corporate T&E benchmark
  ranges (~5-12 transactions/employee/month, higher for travel-heavy
  roles) — not a claim about any specific real benchmark source.
- Category mix per department: travel/client-facing categories skew
  toward Sales, software_saas skews toward R&D, general/office categories
  are more even for HR — reflecting typical role-based spend patterns.
- Amount distributions are lognormal per category (right-skewed, matches
  how real expense amounts behave — many small transactions, a long tail
  of large ones). mu/sigma per category are chosen so the median lands in
  a plausible band and the tail extends far enough for spikes to be
  visually and statistically distinguishable pre-injection:
    software_saas: median ~$60,  tight spread  (mu=ln(60),  sigma=0.40)
    meals:         median ~$35,  moderate      (mu=ln(35),  sigma=0.50)
    office_supplies: median ~$40, moderate     (mu=ln(40),  sigma=0.55)
    client_entertainment: median ~$120, wider  (mu=ln(120), sigma=0.60)
    other:         median ~$50,  wide          (mu=ln(50),  sigma=0.65)
    travel:        median ~$450, widest        (mu=ln(450), sigma=0.70)
- Each employee's transactions are bounded to their own
  window_length_months(duration_months) (same helper as comp_history) —
  an employee cannot log expenses before they joined or after they left
  within the modeled window.
- Anomaly injection never adds transactions: it selects existing
  transactions and overwrites their amount (and marks them), so the
  injection rate is exactly "injected / total" as specified, and total
  transaction volume matches the (larger) base generation target.
"""

import os

import numpy as np
import pandas as pd

from src.generation.attrition_extension import WINDOW_START, window_length_months

SEED = 43

DEPARTMENT_LAMBDA = {
    "Sales": 11.0,
    "Research & Development": 8.5,
    "Human Resources": 7.0,
}

CATEGORIES = ["travel", "software_saas", "meals", "office_supplies", "client_entertainment", "other"]

DEPARTMENT_CATEGORY_PROBS = {
    "Sales": {"travel": 0.30, "client_entertainment": 0.20, "meals": 0.20, "software_saas": 0.10,
              "office_supplies": 0.10, "other": 0.10},
    "Research & Development": {"software_saas": 0.35, "meals": 0.15, "travel": 0.15,
                                "office_supplies": 0.15, "client_entertainment": 0.05, "other": 0.15},
    "Human Resources": {"office_supplies": 0.25, "meals": 0.20, "software_saas": 0.15, "travel": 0.15,
                         "client_entertainment": 0.05, "other": 0.20},
}

CATEGORY_LOGNORMAL_PARAMS = {
    "software_saas": (np.log(60), 0.40),
    "meals": (np.log(35), 0.50),
    "office_supplies": (np.log(40), 0.55),
    "client_entertainment": (np.log(120), 0.60),
    "other": (np.log(50), 0.65),
    "travel": (np.log(450), 0.70),
}

HOT_EMPLOYEE_FRACTION = 0.10
HOT_ANOMALY_SHARE = 0.70
ANOMALY_TYPE_SHARE = {"point_spike": 1 / 3, "slow_drift": 1 / 3, "coordinated_pattern": 1 / 3}
POINT_SPIKE_MULT_RANGE = (5.0, 15.0)
SLOW_DRIFT_MONTHS_RANGE = (4, 8)
SLOW_DRIFT_TOTAL_INCREASE_RANGE = (0.6, 1.8)  # cumulative increase by final drift month
COORDINATED_TXN_COUNT_RANGE = (2, 4)
COORDINATED_MULT_RANGE = (1.5, 2.5)

INJECTION_RATES = {"5pct": 0.05, "1pct": 0.01, "10pct": 0.10}


def generate_base_transactions(employees: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    transaction_id = 1
    for row in employees.itertuples(index=False):
        dept = row.department
        lam = DEPARTMENT_LAMBDA[dept]
        cat_probs = DEPARTMENT_CATEGORY_PROBS[dept]
        cats = list(cat_probs.keys())
        probs = list(cat_probs.values())
        window_len = window_length_months(row.duration_months)

        for month_offset in range(window_len):
            n_txns = rng.poisson(lam)
            if n_txns == 0:
                continue
            month_start = WINDOW_START + pd.DateOffset(months=month_offset)
            days_in_month = (month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1)).day
            categories = rng.choice(cats, size=n_txns, p=probs)
            days = rng.integers(0, days_in_month, size=n_txns)
            for category, day in zip(categories, days):
                mu, sigma = CATEGORY_LOGNORMAL_PARAMS[category]
                amount = float(rng.lognormal(mu, sigma))
                rows.append(
                    {
                        "transaction_id": transaction_id,
                        "employee_id": row.employee_id,
                        "department_id": row.department_id,
                        "transaction_date": month_start + pd.Timedelta(days=int(day)),
                        "merchant_category": category,
                        "amount_usd": round(amount, 2),
                        "is_injected_anomaly": False,
                        "anomaly_type": None,
                    }
                )
                transaction_id += 1
    return pd.DataFrame(rows)


def _apply_point_spike(df, available_idx, full_idx, rng):
    candidates = available_idx
    if len(candidates) == 0:
        return []
    idx = int(rng.choice(candidates))
    category = df.at[idx, "merchant_category"]
    emp_txns = df.loc[full_idx]
    own_mean = emp_txns.loc[emp_txns["merchant_category"] == category, "amount_usd"].mean()
    mult = rng.uniform(*POINT_SPIKE_MULT_RANGE)
    df.at[idx, "amount_usd"] = round(float(own_mean * mult), 2)
    df.at[idx, "is_injected_anomaly"] = True
    df.at[idx, "anomaly_type"] = "point_spike"
    return [idx]


def _apply_slow_drift(df, employee_id, available_idx, rng):
    if not available_idx:
        return []
    emp_txns = df.loc[available_idx]
    category = str(rng.choice(emp_txns["merchant_category"].unique()))
    cat_txns = emp_txns[emp_txns["merchant_category"] == category].copy()
    cat_txns["month"] = cat_txns["transaction_date"].dt.to_period("M")
    months = sorted(cat_txns["month"].unique())
    span = int(rng.integers(*SLOW_DRIFT_MONTHS_RANGE))
    if len(months) < 2:
        return []
    span = min(span, len(months))
    start_idx = int(rng.integers(0, len(months) - span + 1))
    drift_months = months[start_idx:start_idx + span]

    total_increase = rng.uniform(*SLOW_DRIFT_TOTAL_INCREASE_RANGE)
    touched = []
    for i, month in enumerate(drift_months):
        step_mult = 1.0 + total_increase * ((i + 1) / len(drift_months))
        month_idx = cat_txns[cat_txns["month"] == month].index.tolist()
        for idx in month_idx:
            df.at[idx, "amount_usd"] = round(float(df.at[idx, "amount_usd"] * step_mult), 2)
            df.at[idx, "is_injected_anomaly"] = True
            df.at[idx, "anomaly_type"] = "slow_drift"
            touched.append(idx)
    return touched


def _apply_coordinated_pattern(df, employee_id, available_idx, rng):
    if len(available_idx) < 2:
        return []
    emp_txns = df.loc[available_idx].sort_values("transaction_date")
    dates = emp_txns["transaction_date"].tolist()
    idxs = emp_txns.index.tolist()

    start_order = list(range(len(dates)))
    rng.shuffle(start_order)
    best_window = None
    for i in start_order:
        window_idxs = [idxs[i]]
        for j in range(i + 1, len(dates)):
            if (dates[j] - dates[i]).days <= 7:
                window_idxs.append(idxs[j])
        if len(window_idxs) >= 2:
            best_window = window_idxs
            break
    if best_window is None:
        return []

    n = min(len(best_window), int(rng.integers(*COORDINATED_TXN_COUNT_RANGE)) if len(best_window) > 2 else len(best_window))
    n = max(n, 2)
    chosen = list(rng.choice(best_window, size=min(n, len(best_window)), replace=False))

    anchor_date = df.at[chosen[0], "transaction_date"]
    for idx in chosen:
        offset_days = int(rng.integers(0, 7))
        df.at[idx, "transaction_date"] = anchor_date + pd.Timedelta(days=offset_days)
        mult = rng.uniform(*COORDINATED_MULT_RANGE)
        df.at[idx, "amount_usd"] = round(float(df.at[idx, "amount_usd"] * mult), 2)
        df.at[idx, "is_injected_anomaly"] = True
        df.at[idx, "anomaly_type"] = "coordinated_pattern"
    return chosen


def inject_anomalies(base_df: pd.DataFrame, employee_ids: list, injection_rate: float,
                      hot_employees: set, rng: np.random.Generator) -> pd.DataFrame:
    df = base_df.copy()
    total = len(df)
    target_anomalies = round(injection_rate * total)
    cold_employees = [e for e in employee_ids if e not in hot_employees]
    hot_list = list(hot_employees)

    by_employee_idx = {emp: idx.tolist() for emp, idx in df.groupby("employee_id").indices.items()}
    # Shrinking pool of not-yet-anomalous transaction indices per employee. Drawing from
    # this (instead of re-sampling the full, static per-employee index list every time)
    # is what makes the sampler actually converge on the target budget: without it,
    # repeated draws keep colliding with rows a *previous* attempt already flagged, and
    # attempts get burned on structurally-unusable candidates faster than any attempts
    # cap can compensate for.
    available_by_employee = {emp: list(idx) for emp, idx in by_employee_idx.items()}

    anomalous_count = 0
    budget_per_type = {t: round(target_anomalies * share) for t, share in ANOMALY_TYPE_SHARE.items()}

    # If the hot-employee pool (a fixed 10% of the population, by design) runs out of
    # usable transactions before a type's budget is filled, blind 70/30 sampling would
    # burn through attempts on structurally-exhausted employees. After a run of
    # consecutive misses we widen the draw to the full population for a few attempts to
    # make progress, then return to the designed 70/30 split — this keeps the *intended*
    # concentration wherever the hot pool has capacity, and only degrades gracefully
    # (rather than under-filling the anomaly budget) once it doesn't.
    STALL_THRESHOLD = 100

    for anomaly_type, budget in budget_per_type.items():
        count = 0
        attempts = 0
        consecutive_misses = 0
        max_attempts = max(budget * 400, 2000)
        while count < budget and attempts < max_attempts:
            attempts += 1
            widen_pool = consecutive_misses >= STALL_THRESHOLD
            use_hot = rng.random() < HOT_ANOMALY_SHARE
            if widen_pool:
                pool = employee_ids
            else:
                pool = hot_list if (use_hot and hot_list) else cold_employees
            if not pool:
                pool = employee_ids
            employee_id = int(rng.choice(pool))

            full_idx = by_employee_idx.get(employee_id, [])
            available_idx = available_by_employee.get(employee_id, [])
            if anomaly_type == "point_spike":
                touched = _apply_point_spike(df, available_idx, full_idx, rng)
            elif anomaly_type == "slow_drift":
                touched = _apply_slow_drift(df, employee_id, available_idx, rng)
            else:
                touched = _apply_coordinated_pattern(df, employee_id, available_idx, rng)

            if touched:
                touched_set = set(touched)
                available_by_employee[employee_id] = [i for i in available_idx if i not in touched_set]
                count += len(touched)
                anomalous_count += len(touched)
                consecutive_misses = 0
            else:
                consecutive_misses += 1

    return df


def generate_all(employees_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    employees = pd.read_csv(employees_path, parse_dates=False)
    rng = np.random.default_rng(SEED)

    base_df = generate_base_transactions(employees, rng)

    n_hot = max(1, int(round(len(employees) * HOT_EMPLOYEE_FRACTION)))
    hot_employees = set(rng.choice(employees["employee_id"].values, size=n_hot, replace=False).tolist())
    employee_ids = employees["employee_id"].tolist()

    # Audit-only artifact: cohort membership must never be used as a model
    # feature (see src/models/spend/features.py's check_no_cohort_leakage).
    # Written here, not derived from any persisted transaction/employee
    # column, specifically so it stays clearly separate from anything a
    # detector could ever have access to.
    pd.DataFrame(
        {"employee_id": employee_ids, "is_designated_cohort_member": [e in hot_employees for e in employee_ids]}
    ).to_csv(os.path.join(out_dir, "_hot_employees_audit.csv"), index=False)

    stats = {"base_transactions": len(base_df), "hot_employees": n_hot}

    for label, rate in INJECTION_RATES.items():
        variant_rng = np.random.default_rng(SEED + hash(label) % 1000)
        variant_df = inject_anomalies(base_df, employee_ids, rate, hot_employees, variant_rng)
        out_path = os.path.join(out_dir, f"expense_transactions_{label}.csv")
        variant_df.to_csv(out_path, index=False)
        stats[f"{label}_total"] = len(variant_df)
        stats[f"{label}_anomalies"] = int(variant_df["is_injected_anomaly"].sum())
        stats[f"{label}_anomaly_rate"] = round(
            variant_df["is_injected_anomaly"].sum() / len(variant_df), 4
        )
        for atype in ANOMALY_TYPE_SHARE:
            stats[f"{label}_{atype}_count"] = int((variant_df["anomaly_type"] == atype).sum())

    return stats


if __name__ == "__main__":
    stats = generate_all(
        employees_path="data/generated/employees.csv",
        out_dir="data/generated",
    )
    for key, value in stats.items():
        print(f"{key}: {value}")
