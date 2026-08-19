"""Mandatory EDA for the attrition component, run against the shared
Postgres `employees` table (same table the spend component also reads
from) before any modeling happens.

Outputs saved to data/generated/eda_outputs/attrition/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter

from src.db.connection import get_engine

OUT_DIR = "data/generated/eda_outputs/attrition"

NUMERIC_COLUMNS = [
    "age", "daily_rate", "distance_from_home", "education", "environment_satisfaction",
    "hourly_rate", "job_involvement", "job_level", "job_satisfaction", "monthly_income",
    "monthly_rate", "num_companies_worked", "percent_salary_hike", "performance_rating",
    "relationship_satisfaction", "stock_option_level", "total_working_years",
    "training_times_last_year", "work_life_balance", "tenure_years", "years_in_current_role",
    "years_since_last_promotion", "years_with_curr_manager", "duration_months",
]


def load_employees() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM employees", get_engine())


def plot_numeric_distributions(df: pd.DataFrame, out_dir: str):
    n_cols = 4
    n_rows = -(-len(NUMERIC_COLUMNS) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLUMNS):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    for j in range(len(NUMERIC_COLUMNS), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "numeric_distributions.png"), dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str):
    corr = df[NUMERIC_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Attrition dataset: numeric feature correlation matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)


def plot_class_balance(df: pd.DataFrame, out_dir: str):
    counts = df["attrition_flag"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Attrition class balance")
    ax.set_xlabel("attrition_flag")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_balance.png"), dpi=120)
    plt.close(fig)
    return counts.to_dict()


def plot_kaplan_meier(df: pd.DataFrame, out_dir: str):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["duration_months"], event_observed=df["event_observed"], label="Whole population")
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Survival probability (still employed)")
    ax.set_title("Kaplan-Meier survival curve — whole population")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kaplan_meier_overall.png"), dpi=120)
    plt.close(fig)


def check_missingness(df: pd.DataFrame, out_dir: str) -> pd.Series:
    missing = df.isna().sum()
    missing.to_csv(os.path.join(out_dir, "missingness_report.csv"), header=["n_missing"])
    return missing


def run() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_employees()

    plot_numeric_distributions(df, OUT_DIR)
    plot_correlation_heatmap(df, OUT_DIR)
    class_balance = plot_class_balance(df, OUT_DIR)
    plot_kaplan_meier(df, OUT_DIR)
    missing = check_missingness(df, OUT_DIR)

    summary = {
        "n_employees": len(df),
        "class_balance": class_balance,
        "total_missing_values": int(missing.sum()),
        "columns_with_missing": missing[missing > 0].to_dict(),
    }
    print(summary)
    return summary


if __name__ == "__main__":
    run()
