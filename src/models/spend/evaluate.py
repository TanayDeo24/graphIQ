"""All spend evaluation metrics (build spec Section 6, items 1-6).

Run as a script: fits all three detectors + ensemble on the primary 5%
injection-rate dataset, evaluates, saves artifacts to
data/generated/spend_outputs/, writes result tables into Postgres, and
also runs the 1%/10% robustness comparison (item 5).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sqlalchemy import text

from src.db.connection import get_engine
from src.models.spend import autoencoder, cusum, ensemble, isolation_forest
from src.models.spend.features import CATEGORIES, build_feature_frame, load_transactions

OUT_DIR = "data/generated/spend_outputs"
N_BOOTSTRAP = 1000
RNG_SEED = 11
OPERATING_QUANTILE = 0.95  # top 5% ensemble score treated as "alert", ~matches primary injection rate
ANOMALY_TYPES = ["point_spike", "slow_drift", "coordinated_pattern"]


# ---------------------------------------------------------------------
# Fit all detectors + build the unified scored frame
# ---------------------------------------------------------------------
def fit_and_score(df_features: pd.DataFrame) -> pd.DataFrame:
    isf_model = isolation_forest.fit_isolation_forest(df_features)
    isf_scores = isolation_forest.score(isf_model, df_features)

    ae_model, ae_mean, ae_std = autoencoder.fit_autoencoder(df_features)
    ae_scores = autoencoder.score(ae_model, df_features, ae_mean, ae_std)

    monthly_cusum = cusum.compute_cusum_flags(df_features)
    cusum_mapped = cusum.map_flags_to_transactions(df_features, monthly_cusum)

    combined = ensemble.combine(isf_scores, ae_scores, cusum_mapped["cusum_statistic"].values)
    combined["cusum_flag"] = cusum_mapped["flagged"].values

    out = pd.concat([df_features.reset_index(drop=True), combined.reset_index(drop=True)], axis=1)
    return out, (isf_model, ae_model, ae_mean, ae_std)


# ---------------------------------------------------------------------
# 1. Precision / Recall / PR-AUC per anomaly type
# ---------------------------------------------------------------------
def precision_recall_pr_auc_by_type(scored: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    rows = []
    for anomaly_type in ANOMALY_TYPES + ["overall"]:
        if anomaly_type == "overall":
            eval_set = scored
            y_true = eval_set["is_injected_anomaly"].astype(int).values
        else:
            eval_set = scored[(scored["anomaly_type"].isna()) | (scored["anomaly_type"] == anomaly_type)]
            y_true = (eval_set["anomaly_type"] == anomaly_type).astype(int).values

        if y_true.sum() == 0:
            continue
        y_score = eval_set[score_col].values
        y_pred = (y_score >= threshold).astype(int)

        pr_auc = float(average_precision_score(y_true, y_score))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        rows.extend(
            [
                {"anomaly_type": anomaly_type, "metric_name": "precision", "metric_value": precision},
                {"anomaly_type": anomaly_type, "metric_name": "recall", "metric_value": recall},
                {"anomaly_type": anomaly_type, "metric_name": "pr_auc", "metric_value": pr_auc},
            ]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. Dollar-weighted gains curve
# ---------------------------------------------------------------------
def dollar_weighted_gains_curve(scored: pd.DataFrame, score_col: str = "ensemble_score", n_points: int = 100) -> pd.DataFrame:
    df = scored.sort_values(score_col, ascending=False).reset_index(drop=True)
    total_anomalous_dollars = df.loc[df["is_injected_anomaly"], "amount_usd"].sum()

    rows = []
    n = len(df)
    for pct in np.linspace(0.01, 1.0, n_points):
        cutoff = max(1, int(round(pct * n)))
        subset = df.iloc[:cutoff]
        captured = subset.loc[subset["is_injected_anomaly"], "amount_usd"].sum()
        pct_captured = float(captured / total_anomalous_dollars) if total_anomalous_dollars > 0 else 0.0
        rows.append({"pct_alerts_raised": float(pct), "pct_dollar_volume_captured": pct_captured})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. CUSUM drift detection delay with bootstrapped CI
# ---------------------------------------------------------------------
def cusum_drift_delay(scored: pd.DataFrame, monthly_cusum: pd.DataFrame) -> pd.DataFrame:
    drift_txns = scored[scored["anomaly_type"] == "slow_drift"]
    cases = drift_txns.groupby(["employee_id", "merchant_category"])["transaction_date"].min().reset_index()
    cases = cases.rename(columns={"transaction_date": "onset_date"})
    cases["onset_month"] = cases["onset_date"].dt.to_period("M").dt.to_timestamp()

    rows = []
    for row in cases.itertuples(index=False):
        candidates = monthly_cusum[
            (monthly_cusum["employee_id"] == row.employee_id)
            & (monthly_cusum["merchant_category"] == row.merchant_category)
            & (monthly_cusum["month"] >= row.onset_month)
            & (monthly_cusum["flagged"])
        ].sort_values("month")

        flagged_month = candidates["month"].iloc[0] if not candidates.empty else None
        delay = None
        if flagged_month is not None:
            delay = int(round((flagged_month.year - row.onset_month.year) * 12 + (flagged_month.month - row.onset_month.month)))

        rows.append(
            {
                "employee_id": row.employee_id,
                "merchant_category": row.merchant_category,
                "onset_month": row.onset_month,
                "flagged_month": flagged_month,
                "delay_months": delay,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, statistic_fn, n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED) -> tuple:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(statistic_fn(values))
    boot_stats = [statistic_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    return point, float(ci_low), float(ci_high)


# ---------------------------------------------------------------------
# 4. Alert-fatigue audit
# ---------------------------------------------------------------------
def alert_fatigue(scored: pd.DataFrame, threshold: float) -> dict:
    predicted_flag = scored["ensemble_score"] >= threshold
    alerts = int(predicted_flag.sum())
    total = len(scored)
    y_true = scored["is_injected_anomaly"].astype(int).values
    precision = float(precision_score(y_true, predicted_flag.astype(int), zero_division=0))
    return {
        "operating_threshold": float(threshold),
        "alerts_raised": alerts,
        "total_transactions": total,
        "alerts_per_1000_txns": round(alerts / total * 1000, 2),
        "precision_at_threshold": precision,
    }


# ---------------------------------------------------------------------
# 5. Robustness across injection rates
# ---------------------------------------------------------------------
def robustness_across_rates(engine) -> pd.DataFrame:
    rows = []
    for label in ["1pct", "5pct", "10pct"]:
        raw = load_transactions(variant=label, engine=engine)
        featured = build_feature_frame(raw)
        scored, _ = fit_and_score(featured)
        threshold = scored["ensemble_score"].quantile(OPERATING_QUANTILE)
        metrics = precision_recall_pr_auc_by_type(scored, "ensemble_score", threshold)
        metrics["injection_rate"] = label
        rows.append(metrics)
    combined = pd.concat(rows, ignore_index=True)
    pivot = combined.pivot_table(index=["injection_rate", "anomaly_type"], columns="metric_name", values="metric_value").reset_index()
    return pivot.rename(columns={"pr_auc": "pr_auc", "precision": "precision", "recall": "recall"})


# ---------------------------------------------------------------------
# 6. Per-transaction explainability
# ---------------------------------------------------------------------
SUBSIGNAL_GROUPS = {
    "amount_deviation": ["amount_usd", "deviation_from_own_mean", "deviation_from_dept_mean"],
    "frequency": ["trailing_30d_frequency"],
    "merchant_novelty": [f"cat_{c}" for c in CATEGORIES],
}


def explain_transactions(scored: pd.DataFrame, feature_columns: list, ae_model, ae_mean, ae_std,
                          flagged_mask: np.ndarray) -> pd.DataFrame:
    flagged = scored[flagged_mask].reset_index(drop=True)
    if flagged.empty:
        return pd.DataFrame(columns=["transaction_id", "sub_signal", "contribution"])

    recon_error = autoencoder.per_feature_reconstruction_error(ae_model, flagged, ae_mean, ae_std)
    col_idx = {c: i for i, c in enumerate(feature_columns)}

    z_score_cols = ["deviation_from_own_mean", "deviation_from_dept_mean", "trailing_30d_frequency"]
    z_matrix = flagged[z_score_cols].abs().values

    rows = []
    for i in range(len(flagged)):
        group_scores = {}
        for group, cols in SUBSIGNAL_GROUPS.items():
            ae_component = sum(recon_error[i, col_idx[c]] for c in cols if c in col_idx)
            z_component = 0.0
            if group == "amount_deviation":
                z_component = abs(flagged.iloc[i]["deviation_from_own_mean"]) + abs(flagged.iloc[i]["deviation_from_dept_mean"])
            elif group == "frequency":
                z_component = abs(flagged.iloc[i]["trailing_30d_frequency"] - flagged["trailing_30d_frequency"].mean())
            group_scores[group] = 0.5 * ae_component + 0.5 * z_component

        total = sum(group_scores.values()) or 1.0
        for group, val in group_scores.items():
            rows.append(
                {
                    "transaction_id": int(flagged.iloc[i]["transaction_id"]),
                    "sub_signal": group,
                    "contribution": float(val / total),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Persistence + plots
# ---------------------------------------------------------------------
def _write_table(engine, df: pd.DataFrame, table: str):
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY"))
    if not df.empty:
        df.to_sql(table, engine, if_exists="append", index=False)


def plot_gains_curve(gains_df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(gains_df["pct_alerts_raised"] * 100, gains_df["pct_dollar_volume_captured"] * 100)
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray", label="random")
    ax.set_xlabel("% of alerts raised")
    ax.set_ylabel("% of anomalous dollar volume captured")
    ax.set_title("Dollar-weighted gains curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gains_curve.png"), dpi=120)
    plt.close(fig)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = get_engine()

    raw = load_transactions(variant="5pct", engine=engine)
    featured = build_feature_frame(raw)
    scored, (isf_model, ae_model, ae_mean, ae_std) = fit_and_score(featured)

    monthly_cusum = cusum.compute_cusum_flags(featured)

    threshold = scored["ensemble_score"].quantile(OPERATING_QUANTILE)

    # --- 1. precision/recall/PR-AUC by type ---
    metrics_isf = precision_recall_pr_auc_by_type(scored, "isolation_forest_score", scored["isolation_forest_score"].quantile(OPERATING_QUANTILE))
    metrics_isf["detector"] = "isolation_forest"
    metrics_ae = precision_recall_pr_auc_by_type(scored, "autoencoder_score", scored["autoencoder_score"].quantile(OPERATING_QUANTILE))
    metrics_ae["detector"] = "autoencoder"
    metrics_cusum = precision_recall_pr_auc_by_type(scored, "cusum_statistic", scored["cusum_statistic"].quantile(OPERATING_QUANTILE))
    metrics_cusum["detector"] = "cusum"
    metrics_ensemble = precision_recall_pr_auc_by_type(scored, "ensemble_score", threshold)
    metrics_ensemble["detector"] = "ensemble"
    eval_metrics_df = pd.concat([metrics_isf, metrics_ae, metrics_cusum, metrics_ensemble], ignore_index=True)
    eval_metrics_df = eval_metrics_df[["detector", "anomaly_type", "metric_name", "metric_value"]]
    eval_metrics_df.to_csv(os.path.join(OUT_DIR, "eval_metrics_by_type.csv"), index=False)

    # --- 2. gains curve ---
    gains_df = dollar_weighted_gains_curve(scored)
    gains_df.to_csv(os.path.join(OUT_DIR, "gains_curve.csv"), index=False)
    plot_gains_curve(gains_df, OUT_DIR)
    top10 = gains_df.iloc[(gains_df["pct_alerts_raised"] - 0.10).abs().argmin()]
    headline = f"Top {top10['pct_alerts_raised']:.0%} of alerts capture {top10['pct_dollar_volume_captured']:.1%} of flagged dollar volume."
    with open(os.path.join(OUT_DIR, "gains_curve_headline.txt"), "w") as f:
        f.write(headline)

    # --- 3. CUSUM drift delay ---
    drift_delay_df = cusum_drift_delay(scored, monthly_cusum)
    drift_delay_df.to_csv(os.path.join(OUT_DIR, "drift_delay_raw.csv"), index=False)
    resolved_delays = drift_delay_df["delay_months"].dropna().values.astype(float)
    drift_summary_rows = []
    for stat_name, fn in [("mean", np.mean), ("median", np.median)]:
        point, ci_low, ci_high = bootstrap_ci(resolved_delays, fn)
        drift_summary_rows.append(
            {
                "statistic": stat_name,
                "point_estimate": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_bootstrap": N_BOOTSTRAP,
                "n_cases": len(resolved_delays),
            }
        )
    drift_summary_df = pd.DataFrame(drift_summary_rows)
    drift_summary_df.to_csv(os.path.join(OUT_DIR, "drift_delay_summary.csv"), index=False)

    # --- 4. alert fatigue ---
    fatigue = alert_fatigue(scored, threshold)
    fatigue_df = pd.DataFrame([fatigue])
    fatigue_df.to_csv(os.path.join(OUT_DIR, "alert_fatigue.csv"), index=False)

    # --- 5. robustness across injection rates ---
    robustness_df = robustness_across_rates(engine)
    robustness_df.to_csv(os.path.join(OUT_DIR, "robustness_by_injection_rate.csv"), index=False)
    robustness_long = robustness_df.melt(
        id_vars=["injection_rate", "anomaly_type"], value_vars=["precision", "recall", "pr_auc"],
        var_name="metric_name", value_name="value"
    ).rename(columns={"value": "precision"})

    # --- 6. explainability ---
    from src.models.spend.features import FEATURE_COLUMNS
    predicted_flag = (scored["ensemble_score"] >= threshold).values
    explain_df = explain_transactions(scored, FEATURE_COLUMNS, ae_model, ae_mean, ae_std, predicted_flag)
    explain_df.to_csv(os.path.join(OUT_DIR, "transaction_explanations.csv"), index=False)

    # --- anomaly_scores table ---
    anomaly_scores_df = scored[
        ["transaction_id", "employee_id", "department_id", "isolation_forest_score", "autoencoder_score",
         "cusum_flag", "ensemble_score", "is_injected_anomaly", "anomaly_type"]
    ].copy()
    anomaly_scores_df["predicted_flag"] = predicted_flag
    anomaly_scores_df.to_csv(os.path.join(OUT_DIR, "anomaly_scores.csv"), index=False)

    # --- write to Postgres ---
    _write_table(engine, anomaly_scores_df, "spend_anomaly_scores")

    spend_eval_metrics_db = eval_metrics_df.copy()
    _write_table(engine, spend_eval_metrics_db, "spend_eval_metrics")

    _write_table(engine, gains_df.rename(columns={}), "spend_gains_curve")

    drift_delay_db = drift_delay_df.dropna(subset=["employee_id"]).copy()
    _write_table(engine, drift_delay_db, "spend_drift_delay")
    _write_table(engine, drift_summary_df, "spend_drift_delay_summary")

    emp_dept = scored[["employee_id", "department_id"]].drop_duplicates()
    cusum_series_db = monthly_cusum.merge(emp_dept, on="employee_id", how="left")
    cusum_series_db = cusum_series_db[
        ["employee_id", "department_id", "merchant_category", "month", "monthly_total", "cusum_statistic", "flagged"]
    ]
    _write_table(engine, cusum_series_db, "spend_cusum_series")

    _write_table(engine, fatigue_df, "spend_alert_fatigue")

    robustness_db = robustness_df.melt(
        id_vars=["injection_rate", "anomaly_type"], value_vars=["precision", "recall", "pr_auc"],
        var_name="metric_name", value_name="metric_value"
    )
    robustness_wide = robustness_df[["injection_rate", "anomaly_type", "precision", "recall", "pr_auc"]]
    _write_table(engine, robustness_wide, "spend_robustness")

    _write_table(engine, explain_df, "spend_transaction_explain")

    print("Spend evaluation complete.")
    print(headline)
    print(fatigue)
    return {"headline": headline, "alert_fatigue": fatigue}


if __name__ == "__main__":
    run()
