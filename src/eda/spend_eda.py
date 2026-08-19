"""Mandatory EDA for the spend component, run against the shared Postgres
`expense_transactions` table (joined to `employees`/`departments` via
employee_id — same join key the attrition component uses) before any
modeling happens.

Outputs saved to data/generated/eda_outputs/spend/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.db.connection import get_engine

OUT_DIR = "data/generated/eda_outputs/spend"

NUMERIC_COLUMNS = ["amount_usd"]


def load_transactions() -> pd.DataFrame:
    query = """
        SELECT t.*, e.department AS employee_department
        FROM expense_transactions t
        JOIN employees e ON e.employee_id = t.employee_id
    """
    df = pd.read_sql(query, get_engine())
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def plot_amount_distribution(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df["amount_usd"], kde=True, ax=ax, bins=80)
    ax.set_title("Transaction amount distribution (all categories)")
    ax.set_xlabel("amount_usd")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "amount_distribution.png"), dpi=120)
    plt.close(fig)


def plot_category_distributions(df: pd.DataFrame, out_dir: str):
    categories = sorted(df["merchant_category"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, category in enumerate(categories):
        sns.histplot(df.loc[df["merchant_category"] == category, "amount_usd"], kde=True, ax=axes[i], bins=50)
        axes[i].set_title(category)
    for j in range(len(categories), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "category_distributions.png"), dpi=120)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str):
    numeric = df.copy()
    numeric["day_of_week"] = numeric["transaction_date"].dt.dayofweek
    numeric["is_injected_anomaly_int"] = numeric["is_injected_anomaly"].astype(int)
    cat_dummies = pd.get_dummies(numeric["merchant_category"], prefix="cat")
    corr_df = pd.concat(
        [numeric[["amount_usd", "day_of_week", "is_injected_anomaly_int"]], cat_dummies], axis=1
    )
    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Spend dataset: feature correlation matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)


def plot_anomaly_class_balance(df: pd.DataFrame, out_dir: str):
    counts = df["is_injected_anomaly"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v / counts.sum():.2%})", ha="center", va="bottom")
    ax.set_title("Injected-anomaly class balance")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "anomaly_class_balance.png"), dpi=120)
    plt.close(fig)
    return counts.to_dict()


def plot_monthly_spend_timeseries(df: pd.DataFrame, out_dir: str):
    monthly = df.set_index("transaction_date").resample("MS")["amount_usd"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(ax=ax, marker="o")
    ax.set_title("Aggregate monthly spend (sanity check for seasonality)")
    ax.set_ylabel("Total amount_usd")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "monthly_spend_timeseries.png"), dpi=120)
    plt.close(fig)


def check_missingness(df: pd.DataFrame, out_dir: str) -> pd.Series:
    missing = df.isna().sum()
    missing.to_csv(os.path.join(out_dir, "missingness_report.csv"), header=["n_missing"])
    return missing


def run() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_transactions()

    plot_amount_distribution(df, OUT_DIR)
    plot_category_distributions(df, OUT_DIR)
    plot_correlation_heatmap(df, OUT_DIR)
    anomaly_balance = plot_anomaly_class_balance(df, OUT_DIR)
    plot_monthly_spend_timeseries(df, OUT_DIR)
    missing = check_missingness(df, OUT_DIR)

    summary = {
        "n_transactions": len(df),
        "anomaly_class_balance": anomaly_balance,
        "total_missing_values": int(missing.sum()),
        "columns_with_missing": missing[missing > 0].to_dict(),
    }
    print(summary)
    return summary


if __name__ == "__main__":
    run()
