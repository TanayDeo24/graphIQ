"""Load all generated data (Phase 3 outputs) into the shared Postgres
schema (sql/schema.sql). Idempotent: truncates and reloads every table it
touches, so it can be re-run after regenerating data.

Run order matters for foreign keys: departments -> employees ->
{comp_history, performance_reviews, benefits_enrollment, expense_transactions}.
"""

import os

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine

GENERATED_DIR = "data/generated"

EMPLOYEES_COLUMNS = [
    "employee_id", "age", "attrition_flag", "business_travel", "daily_rate", "department",
    "department_id", "distance_from_home", "education", "education_field", "employee_count",
    "environment_satisfaction", "gender", "hourly_rate", "job_involvement", "job_level", "job_role",
    "job_satisfaction", "marital_status", "monthly_income", "monthly_rate", "num_companies_worked",
    "over_18", "over_time", "percent_salary_hike", "performance_rating", "relationship_satisfaction",
    "standard_hours", "stock_option_level", "total_working_years", "training_times_last_year",
    "work_life_balance", "tenure_years", "years_in_current_role", "years_since_last_promotion",
    "years_with_curr_manager", "duration_months", "event_observed",
]

TRUNCATE_ORDER = [
    "expense_transactions",
    "benefits_enrollment",
    "performance_reviews",
    "comp_history",
    "employees",
    "departments",
]


def apply_results_schema(engine, path: str = "sql/results_schema.sql"):
    with open(path) as f:
        ddl = f.read()
    with engine.begin() as conn:
        conn.execute(text(ddl))


def truncate_all(engine):
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {', '.join(TRUNCATE_ORDER)} RESTART IDENTITY CASCADE"))


def load_departments(engine):
    df = pd.read_csv(os.path.join(GENERATED_DIR, "departments.csv"))
    df.to_sql("departments", engine, if_exists="append", index=False)
    return len(df)


def load_employees(engine):
    df = pd.read_csv(os.path.join(GENERATED_DIR, "employees.csv"))
    df = df[EMPLOYEES_COLUMNS]
    df.to_sql("employees", engine, if_exists="append", index=False)
    return len(df)


def load_comp_history(engine):
    df = pd.read_csv(os.path.join(GENERATED_DIR, "comp_history.csv"), parse_dates=["effective_month"])
    df.to_sql("comp_history", engine, if_exists="append", index=False)
    return len(df)


def load_performance_reviews(engine):
    df = pd.read_csv(os.path.join(GENERATED_DIR, "performance_reviews.csv"), parse_dates=["review_month"])
    df.to_sql("performance_reviews", engine, if_exists="append", index=False)
    return len(df)


def load_benefits_enrollment(engine):
    df = pd.read_csv(os.path.join(GENERATED_DIR, "benefits_enrollment.csv"), parse_dates=["enrolled_month"])
    df.to_sql("benefits_enrollment", engine, if_exists="append", index=False)
    return len(df)


def load_expense_transactions(engine, variant: str = "5pct"):
    df = pd.read_csv(
        os.path.join(GENERATED_DIR, f"expense_transactions_{variant}.csv"),
        parse_dates=["transaction_date"],
    )
    df = df.drop(columns=["transaction_id"])
    df.to_sql("expense_transactions", engine, if_exists="append", index=False)
    return len(df)


def main():
    engine = get_engine()
    apply_results_schema(engine)
    truncate_all(engine)

    counts = {
        "departments": load_departments(engine),
        "employees": load_employees(engine),
        "comp_history": load_comp_history(engine),
        "performance_reviews": load_performance_reviews(engine),
        "benefits_enrollment": load_benefits_enrollment(engine),
        "expense_transactions": load_expense_transactions(engine, variant="5pct"),
    }
    for table, count in counts.items():
        print(f"Loaded {count} rows into {table}")


if __name__ == "__main__":
    main()
