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
from src.models.spend import autoencoder, cohort_cusum, cusum, ensemble, isolation_forest
from src.models.spend.features import CATEGORIES, build_feature_frame, check_no_cohort_leakage, load_transactions
from src.models.spend.tune_cusum import H_SIGMA_TUNED

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

    # H_SIGMA_TUNED (not the module's general-purpose H_SIGMA=5 default) is used
    # for the boolean `flagged` column here -- see tune_cusum.py for the tuning-
    # dataset procedure that derived it. Does not affect cusum_statistic itself
    # (and therefore not PR-AUC), only the drift-delay/detection-timing metrics
    # and the cusum_flag display field, which depend on the flag.
    monthly_cusum = cusum.compute_cusum_flags(df_features, h=H_SIGMA_TUNED)
    cusum_mapped = cusum.map_flags_to_transactions(df_features, monthly_cusum)

    monthly_cohort_cusum = cohort_cusum.compute_cohort_cusum_flags(df_features)
    cohort_cusum_mapped = cohort_cusum.map_flags_to_transactions(df_features, monthly_cohort_cusum)

    combined = ensemble.combine(isf_scores, ae_scores, cusum_mapped["cusum_statistic"].values)
    combined["cusum_flag"] = cusum_mapped["flagged"].values
    combined["cohort_cusum_statistic"] = cohort_cusum_mapped["cusum_statistic"].values
    combined["cohort_cusum_flag"] = cohort_cusum_mapped["flagged"].values

    out = pd.concat([df_features.reset_index(drop=True), combined.reset_index(drop=True)], axis=1)
    return out, (isf_model, ae_model, ae_mean, ae_std)


# ---------------------------------------------------------------------
# 1. Precision / Recall / PR-AUC per anomaly type (+ lift-over-random)
# ---------------------------------------------------------------------
def compute_prevalence_rates(scored: pd.DataFrame) -> dict:
    """Actual injected rate of each anomaly type within the full dataset
    (e.g. ~1.667% each for a 5%-overall injection split evenly three ways;
    computed from the real counts, not assumed even). 'overall' is the
    primary injection rate itself (~5%). Used as the denominator for
    lift-over-random: PR-AUC / prevalence_rate, so a detector no better
    than random guessing would score ~1.0 regardless of anomaly type, and
    higher numbers are directly comparable across types with very
    different base rates.
    """
    total = len(scored)
    rates = {"overall": float(scored["is_injected_anomaly"].sum()) / total}
    for anomaly_type in ANOMALY_TYPES:
        rates[anomaly_type] = float((scored["anomaly_type"] == anomaly_type).sum()) / total
    return rates


def precision_recall_pr_auc_by_type(scored: pd.DataFrame, score_col: str, threshold: float,
                                     prevalence_rates: dict) -> pd.DataFrame:
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
        lift = pr_auc / prevalence_rates[anomaly_type]
        rows.extend(
            [
                {"anomaly_type": anomaly_type, "metric_name": "precision", "metric_value": precision},
                {"anomaly_type": anomaly_type, "metric_name": "recall", "metric_value": recall},
                {"anomaly_type": anomaly_type, "metric_name": "pr_auc", "metric_value": pr_auc},
                {"anomaly_type": anomaly_type, "metric_name": "lift_over_random", "metric_value": lift},
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
# 3. CUSUM drift detection delay with bootstrapped CI, plus whether each
# detected case was caught while the drift was still active or only after
# it had already ended (see drift_detection_timing() below).
# ---------------------------------------------------------------------
def cusum_drift_delay(scored: pd.DataFrame, monthly_cusum: pd.DataFrame) -> pd.DataFrame:
    drift_txns = scored[scored["anomaly_type"] == "slow_drift"]
    onset = drift_txns.groupby(["employee_id", "merchant_category"])["transaction_date"].min()
    end = drift_txns.groupby(["employee_id", "merchant_category"])["transaction_date"].max()
    cases = pd.DataFrame({"onset_date": onset, "end_date": end}).reset_index()
    cases["onset_month"] = cases["onset_date"].dt.to_period("M").dt.to_timestamp()
    cases["end_month"] = cases["end_date"].dt.to_period("M").dt.to_timestamp()

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
        caught_during_active_window = None
        if flagged_month is not None:
            delay = int(round((flagged_month.year - row.onset_month.year) * 12 + (flagged_month.month - row.onset_month.month)))
            caught_during_active_window = bool(flagged_month <= row.end_month)

        rows.append(
            {
                "employee_id": row.employee_id,
                "merchant_category": row.merchant_category,
                "onset_month": row.onset_month,
                "end_month": row.end_month,
                "flagged_month": flagged_month,
                "delay_months": delay,
                "caught_during_active_window": caught_during_active_window,
            }
        )
    return pd.DataFrame(rows)


def drift_detection_timing(drift_delay_df: pd.DataFrame) -> dict:
    """Honest complement to the delay mean/median: of the slow_drift cases
    CUSUM actually caught, what fraction were caught while the drift was
    still active vs. only after it had already ended (i.e. the delay was
    longer than the drift's own 4-8 month duration)? Framing this only as
    a mean delay number can look better than it is if a large share of
    "successful" detections actually landed after the drift was already
    over."""
    n_total = len(drift_delay_df)
    detected = drift_delay_df[drift_delay_df["flagged_month"].notna()]
    n_detected = len(detected)
    n_undetected = n_total - n_detected

    n_during = int(detected["caught_during_active_window"].sum()) if n_detected else 0
    n_after = n_detected - n_during

    return {
        "n_total_cases": n_total,
        "n_detected": n_detected,
        "n_undetected": n_undetected,
        "n_caught_during_active": n_during,
        "n_caught_after_ended": n_after,
        "pct_caught_during_active": (n_during / n_detected) if n_detected else float("nan"),
        "pct_caught_after_ended": (n_after / n_detected) if n_detected else float("nan"),
    }


# ---------------------------------------------------------------------
# Annotated CUSUM trajectories (dashboard): a small, curated set of real
# detected slow_drift cases with their full monthly trajectory, for the
# "annotated CUSUM trajectory" chart. The full trajectory data already
# exists for every (employee, category) pair in spend_cusum_series — this
# does not duplicate that; it selects and labels a handful of specific,
# illustrative cases from it, joined with their onset/end/flagged months.
# ---------------------------------------------------------------------
def select_annotated_cusum_cases(drift_delay_df: pd.DataFrame, rng_seed: int = RNG_SEED) -> pd.DataFrame:
    detected = drift_delay_df[drift_delay_df["flagged_month"].notna()].copy()
    detected["delay_months"] = detected["delay_months"].astype(float)
    # caught_during_active_window comes through as object dtype (mixed None/bool
    # in the source column before this filter) -- `~` on a plain-object column of
    # Python bools does bitwise NOT (~True == -2), not logical negation. Cast first.
    detected["caught_during_active_window"] = detected["caught_during_active_window"].astype(bool)
    during = detected[detected["caught_during_active_window"]]
    after = detected[~detected["caught_during_active_window"]]

    rng = np.random.default_rng(rng_seed)
    picks = []
    if len(during) >= 2:
        picks.append(during.sample(n=2, random_state=rng_seed))
    if len(after) >= 1:
        picks.append(after.sample(n=1, random_state=rng_seed))
    # Borderline: caught during the active window, but with the smallest margin
    # (flagged closest to end_month) -- a real "nearly missed it" case.
    if not during.empty:
        during_with_margin = during.copy()
        during_with_margin["margin_months"] = (
            (during_with_margin["end_month"] - during_with_margin["flagged_month"]).dt.days / 30.44
        )
        borderline = during_with_margin.nsmallest(1, "margin_months")
        picks.append(borderline[during.columns])
    # One more "caught while active" case for a fuller set of 4-5, if available.
    remaining_during = during.drop(index=pd.concat(picks[:1]).index if picks else [], errors="ignore")
    if len(remaining_during) >= 1:
        picks.append(remaining_during.sample(n=1, random_state=rng_seed + 1))

    if not picks:
        return pd.DataFrame(columns=["case_label", "employee_id", "merchant_category", "onset_month", "end_month", "flagged_month"])

    selected = pd.concat(picks).drop_duplicates(subset=["employee_id", "merchant_category"])
    selected = selected.reset_index(drop=True)
    selected["case_label"] = [f"case_{i + 1}" for i in range(len(selected))]
    return selected[["case_label", "employee_id", "merchant_category", "onset_month", "end_month", "flagged_month", "caught_during_active_window"]]


def annotated_cusum_trajectories(annotated_cases: pd.DataFrame, monthly_cusum: pd.DataFrame) -> pd.DataFrame:
    if annotated_cases.empty:
        return pd.DataFrame(columns=["case_label", "employee_id", "merchant_category", "month", "cusum_statistic", "flagged"])
    rows = []
    for row in annotated_cases.itertuples(index=False):
        series = monthly_cusum[
            (monthly_cusum["employee_id"] == row.employee_id) & (monthly_cusum["merchant_category"] == row.merchant_category)
        ].sort_values("month")
        for s in series.itertuples(index=False):
            rows.append(
                {
                    "case_label": row.case_label,
                    "employee_id": row.employee_id,
                    "merchant_category": row.merchant_category,
                    "month": s.month,
                    "cusum_statistic": s.cusum_statistic,
                    "flagged": s.flagged,
                    "onset_month": row.onset_month,
                    "end_month": row.end_month,
                    "flagged_month": row.flagged_month,
                    "caught_during_active_window": row.caught_during_active_window,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Detector comparison table: rows = detector, one PR-AUC column and one
# lift-over-random column (PR-AUC / that type's actual injected prevalence
# rate — see compute_prevalence_rates) per anomaly type. Lift is a new
# column pair alongside the existing PR-AUC ones in this same table, not a
# separate table, so a type's very different base rates (point_spike/
# slow_drift/coordinated_pattern are injected in equal counts but "overall"
# has ~3x their individual prevalence) stay directly comparable at a
# glance without leaving this view.
# ---------------------------------------------------------------------
def detector_comparison_table(eval_metrics_df: pd.DataFrame) -> pd.DataFrame:
    pr_auc_rows = eval_metrics_df[eval_metrics_df["metric_name"] == "pr_auc"]
    lift_rows = eval_metrics_df[eval_metrics_df["metric_name"] == "lift_over_random"]

    pr_auc_pivot = pr_auc_rows.pivot_table(index="detector", columns="anomaly_type", values="metric_value")
    lift_pivot = lift_rows.pivot_table(index="detector", columns="anomaly_type", values="metric_value")

    ordered_types = [c for c in ANOMALY_TYPES + ["overall"] if c in pr_auc_pivot.columns]
    combined = pd.DataFrame(index=pr_auc_pivot.index)
    for t in ordered_types:
        combined[f"{t}_pr_auc"] = pr_auc_pivot[t]
        combined[f"{t}_lift"] = lift_pivot[t]
    return combined.reset_index()


# ---------------------------------------------------------------------
# Detector overlap (dashboard): at the existing "50 alerts / 1,000 txns"
# operating threshold, how many transactions are flagged by exactly 1, 2,
# or 3 of {Isolation Forest, Autoencoder, CUSUM}? Cohort CUSUM excluded —
# dominated by the other three, would just add noise to this comparison.
# ---------------------------------------------------------------------
def detector_overlap(scored: pd.DataFrame) -> pd.DataFrame:
    isf_flag = scored["isolation_forest_score"] >= scored["isolation_forest_score"].quantile(OPERATING_QUANTILE)
    ae_flag = scored["autoencoder_score"] >= scored["autoencoder_score"].quantile(OPERATING_QUANTILE)
    cusum_flag = scored["cusum_statistic"] >= scored["cusum_statistic"].quantile(OPERATING_QUANTILE)

    combo_labels = pd.Series("none", index=scored.index)
    combo_labels[isf_flag & ~ae_flag & ~cusum_flag] = "only_IF"
    combo_labels[~isf_flag & ae_flag & ~cusum_flag] = "only_AE"
    combo_labels[~isf_flag & ~ae_flag & cusum_flag] = "only_CUSUM"
    combo_labels[isf_flag & ae_flag & ~cusum_flag] = "IF+AE"
    combo_labels[isf_flag & ~ae_flag & cusum_flag] = "IF+CUSUM"
    combo_labels[~isf_flag & ae_flag & cusum_flag] = "AE+CUSUM"
    combo_labels[isf_flag & ae_flag & cusum_flag] = "all_three"

    counts = combo_labels[combo_labels != "none"].value_counts()
    order = ["only_IF", "only_AE", "only_CUSUM", "IF+AE", "IF+CUSUM", "AE+CUSUM", "all_three"]
    return pd.DataFrame(
        {"combination": order, "transaction_count": [int(counts.get(c, 0)) for c in order]}
    )


# ---------------------------------------------------------------------
# Dollar treemap (dashboard): department x category, sized by anomalous
# dollar volume, colored by (dominant) anomaly type.
# ---------------------------------------------------------------------
def dollar_treemap(scored: pd.DataFrame) -> pd.DataFrame:
    anomalous = scored[scored["is_injected_anomaly"]].copy()
    if anomalous.empty:
        return pd.DataFrame(columns=["department_id", "merchant_category", "anomaly_type", "dollar_volume", "transaction_count"])
    grouped = (
        anomalous.groupby(["department_id", "merchant_category", "anomaly_type"])["amount_usd"]
        .agg(dollar_volume="sum", transaction_count="count")
        .reset_index()
    )
    return grouped


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
        rate_prevalence = compute_prevalence_rates(scored)
        metrics = precision_recall_pr_auc_by_type(scored, "ensemble_score", threshold, rate_prevalence)
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
    check_no_cohort_leakage(featured)
    scored, (isf_model, ae_model, ae_mean, ae_std) = fit_and_score(featured)

    monthly_cusum = cusum.compute_cusum_flags(featured)

    threshold = scored["ensemble_score"].quantile(OPERATING_QUANTILE)
    prevalence_rates = compute_prevalence_rates(scored)

    # --- 1. precision/recall/PR-AUC by type (+ lift-over-random) ---
    metrics_isf = precision_recall_pr_auc_by_type(scored, "isolation_forest_score", scored["isolation_forest_score"].quantile(OPERATING_QUANTILE), prevalence_rates)
    metrics_isf["detector"] = "isolation_forest"
    metrics_ae = precision_recall_pr_auc_by_type(scored, "autoencoder_score", scored["autoencoder_score"].quantile(OPERATING_QUANTILE), prevalence_rates)
    metrics_ae["detector"] = "autoencoder"
    metrics_cusum = precision_recall_pr_auc_by_type(scored, "cusum_statistic", scored["cusum_statistic"].quantile(OPERATING_QUANTILE), prevalence_rates)
    metrics_cusum["detector"] = "cusum"
    metrics_cohort_cusum = precision_recall_pr_auc_by_type(
        scored, "cohort_cusum_statistic", scored["cohort_cusum_statistic"].quantile(OPERATING_QUANTILE), prevalence_rates
    )
    metrics_cohort_cusum["detector"] = "cohort_cusum"
    metrics_ensemble = precision_recall_pr_auc_by_type(scored, "ensemble_score", threshold, prevalence_rates)
    metrics_ensemble["detector"] = "ensemble"
    eval_metrics_df = pd.concat(
        [metrics_isf, metrics_ae, metrics_cusum, metrics_cohort_cusum, metrics_ensemble], ignore_index=True
    )
    eval_metrics_df = eval_metrics_df[["detector", "anomaly_type", "metric_name", "metric_value"]]
    eval_metrics_df.to_csv(os.path.join(OUT_DIR, "eval_metrics_by_type.csv"), index=False)

    comparison_df = detector_comparison_table(eval_metrics_df)
    comparison_df.to_csv(os.path.join(OUT_DIR, "detector_comparison_pr_auc.csv"), index=False)

    # --- dashboard: detector overlap + dollar treemap ---
    overlap_df = detector_overlap(scored)
    overlap_df.to_csv(os.path.join(OUT_DIR, "detector_overlap.csv"), index=False)

    treemap_df = dollar_treemap(scored)
    treemap_df.to_csv(os.path.join(OUT_DIR, "dollar_treemap.csv"), index=False)

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

    timing = drift_detection_timing(drift_delay_df)
    timing_df = pd.DataFrame([timing])
    timing_df.to_csv(os.path.join(OUT_DIR, "drift_detection_timing.csv"), index=False)

    # --- dashboard: annotated CUSUM trajectories ---
    annotated_cases = select_annotated_cusum_cases(drift_delay_df)
    annotated_trajectories = annotated_cusum_trajectories(annotated_cases, monthly_cusum)
    annotated_trajectories.to_csv(os.path.join(OUT_DIR, "cusum_annotated_trajectories.csv"), index=False)

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
    _write_table(engine, timing_df, "spend_drift_timing_summary")

    emp_dept = scored[["employee_id", "department_id"]].drop_duplicates()
    cusum_series_db = monthly_cusum.merge(emp_dept, on="employee_id", how="left")
    cusum_series_db = cusum_series_db[
        ["employee_id", "department_id", "merchant_category", "month", "monthly_total", "cusum_statistic", "flagged"]
    ]
    _write_table(engine, cusum_series_db, "spend_cusum_series")

    _write_table(engine, fatigue_df, "spend_alert_fatigue")

    _write_table(engine, overlap_df, "spend_detector_overlap")
    _write_table(engine, treemap_df, "spend_dollar_treemap")
    _write_table(engine, annotated_trajectories, "spend_cusum_annotated_trajectory")

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
    print("Detector comparison (PR-AUC):")
    print(comparison_df.to_string(index=False))
    print(
        f"Drift timing: {timing['n_detected']}/{timing['n_total_cases']} slow_drift cases detected; "
        f"of those, {timing['pct_caught_during_active']:.0%} caught while still active, "
        f"{timing['pct_caught_after_ended']:.0%} only caught after the drift had already ended."
    )
    return {"headline": headline, "alert_fatigue": fatigue, "drift_timing": timing}


if __name__ == "__main__":
    run()
