"""All attrition evaluation metrics (build spec Section 5, items 1-8).

Run as a script: fits both models on the temporal train split, evaluates
on the temporal test split, saves artifacts to
data/generated/attrition_outputs/, and writes result tables into Postgres
for the API layer.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv
from sqlalchemy import text

from src.db.connection import get_engine
from src.models.attrition import cox_model, gbm_survival
from src.models.attrition.features import build_feature_frame, check_no_leakage, encode_features

OUT_DIR = "data/generated/attrition_outputs"
N_BOOTSTRAP = 1000
RNG_SEED = 7

TIME_HORIZONS_DAYS = [30, 60, 90, 180]
DAYS_PER_MONTH = 30
CALIBRATION_HORIZON_MONTHS = 12
TOP_RISK_QUANTILE = 0.75
# Cumulative-risk crossing threshold for the lead-time metric (1 - S(t) >= threshold).
# Calibrated to this dataset's overall attrition base rate (~16-20%): individual
# predicted cumulative risk over the observed window rarely exceeds ~40% even for the
# highest-risk true positives, so a 50%-style threshold would (and in testing, did)
# leave the lead-time cohort empty. 10% is chosen so it captures essentially the full
# true-positive cohort while still requiring a real, non-trivial accumulation of risk
# (not just "flagged from day one").
LEAD_TIME_CUMULATIVE_RISK_THRESHOLD = 0.10
SENSITIVITY_INCOME_BUMP = 0.10


def _tenure_band(tenure_years) -> str:
    if tenure_years <= 2:
        return "0-2"
    if tenure_years <= 5:
        return "2-5"
    return "5+"


def _comp_band(monthly_income: pd.Series) -> pd.Series:
    return pd.qcut(monthly_income, 3, labels=["low", "mid", "high"])


# ---------------------------------------------------------------------
# 1. Proportional hazards assumption check
# ---------------------------------------------------------------------
def check_ph_assumption(cph, train_encoded: pd.DataFrame) -> pd.DataFrame:
    model_df = train_encoded.copy()
    model_df["event_observed"] = model_df["event_observed"].astype(int)
    result = proportional_hazard_test(cph, model_df, time_transform="rank")
    summary = result.summary.reset_index().rename(columns={"index": "feature"})
    summary["violated"] = summary["p"] < 0.05
    return summary[["feature", "test_statistic", "p", "violated"]].rename(columns={"p": "p_value"})


# ---------------------------------------------------------------------
# 2. Concordance index
# ---------------------------------------------------------------------
def concordance(event_observed, duration_months, risk_score) -> float:
    result = concordance_index_censored(
        event_observed.astype(bool), duration_months.astype(float), risk_score.astype(float)
    )
    return float(result[0])


def within_tenure_band_concordance(event_observed, duration_months, risk_score, tenure_years) -> dict:
    """Concordance computed separately within each real tenure_years band,
    then pooled (event-count-weighted). Diagnostic added after the
    feature-leakage fix: duration_months is built as
    (tenure_years - 1) * 12 + month_within_final_year (see
    attrition_extension.assign_fine_grained_duration) -- a between-year
    spread of up to ~470 months against only ~12 months of within-year
    noise, so duration_months is algebraically dominated by tenure_years.
    Any feature even moderately correlated with tenure_years (job_level,
    monthly_income, benefits_tier all are, through ordinary real-world
    career progression -- none of them individually or literally leaked)
    lets a flexible model reconstruct most of that between-year ranking
    without any single feature ever looking like a copy of the target.
    Overall concordance is structurally inflated by that regardless of
    whether the feature set is clean. This metric asks the fairer
    question instead: among people who've already been there about the
    same length of time, can the model tell who's more likely to leave
    sooner? That's the part of the ranking task that isn't just "does the
    model know how long you've been here already."
    """
    band = pd.Series(tenure_years).apply(_tenure_band)
    df = pd.DataFrame(
        {"event_observed": event_observed, "duration_months": duration_months, "risk_score": risk_score, "band": band.values}
    )
    rows = []
    weighted_sum, weight_total = 0.0, 0
    for band_name, group in df.groupby("band"):
        n_events = int(group["event_observed"].sum())
        if n_events < 2 or len(group) < 5:
            rows.append({"tenure_band": band_name, "n": len(group), "n_events": n_events, "concordance": None})
            continue
        c = concordance(group["event_observed"].values, group["duration_months"].values, group["risk_score"].values)
        rows.append({"tenure_band": band_name, "n": len(group), "n_events": n_events, "concordance": c})
        weighted_sum += c * n_events
        weight_total += n_events

    pooled = (weighted_sum / weight_total) if weight_total > 0 else float("nan")
    return {"by_band": rows, "pooled": pooled, "n_events_pooled": weight_total}


# ---------------------------------------------------------------------
# Interaction risk heatmap (dashboard): baseline_tenure_band x
# review_score_trend bucket, cell = mean baseline GBM risk score.
# ---------------------------------------------------------------------
REVIEW_TREND_DECLINING_MAX = -0.2
REVIEW_TREND_IMPROVING_MIN = 0.2
INTERACTION_HEATMAP_MIN_N = 10


def _review_trend_bucket(value: float) -> str:
    if value <= REVIEW_TREND_DECLINING_MAX:
        return "declining"
    if value >= REVIEW_TREND_IMPROVING_MIN:
        return "improving"
    return "stable"


def interaction_risk_heatmap(full_df: pd.DataFrame, gbm_risk_score: np.ndarray) -> pd.DataFrame:
    """3x3 grid for the dashboard: baseline_tenure_band x review_score_trend
    bucket (declining: <= -0.2, stable: (-0.2, 0.2), improving: >= 0.2 --
    thresholds chosen relative to review_score_trend's observed spread,
    std ~0.54, so "stable" covers roughly the middle ~0.7 std and each tail
    bucket is a real, not marginal, trend direction), cell = mean baseline
    GBM risk score for employees in that cell. Cells with n < 10 are
    flagged low_confidence (same INTERACTION_HEATMAP_MIN_N convention as
    segment_calibration's low-confidence flag) rather than presented with
    the same visual weight as a well-populated cell.
    """
    df = full_df.copy()
    df["gbm_risk_score"] = gbm_risk_score
    df["review_trend_bucket"] = df["review_score_trend"].apply(_review_trend_bucket)

    rows = []
    for tenure_band in ["0-2", "2-5", "5+"]:
        for trend_bucket in ["declining", "stable", "improving"]:
            cell = df[(df["baseline_tenure_band"] == tenure_band) & (df["review_trend_bucket"] == trend_bucket)]
            n = len(cell)
            rows.append(
                {
                    "tenure_band": tenure_band,
                    "review_trend_bucket": trend_bucket,
                    "n": n,
                    "mean_gbm_risk_score": float(cell["gbm_risk_score"].mean()) if n > 0 else None,
                    "low_confidence": n < INTERACTION_HEATMAP_MIN_N,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. Time-dependent AUC
# ---------------------------------------------------------------------
def time_dependent_auc(y_train, y_test, risk_score_test) -> dict:
    """Time-dependent AUC at horizons corresponding to 30/60/90/180 days
    (build spec Section 5 item 3), expressed in months since our data's
    native granularity is monthly.

    Uses the rolling-origin fit/eval views built in run(): the model is
    fit on outcomes as administratively censored at month 24, and
    evaluated against the same population's outcomes as censored at
    month 36. This keeps `y_train`'s censoring distribution (needed for
    cumulative_dynamic_auc's IPCW weighting) supported across the full
    evaluation range — a strict "short-tenure employees train, long-
    tenure employees test" partition instead starves that censoring
    distribution of support beyond month 24, which makes IPCW undefined
    at any of these horizons; see run()'s docstring note for the earlier
    (rejected) design and why.
    """
    horizons_months = sorted({max(1, round(d / DAYS_PER_MONTH)) for d in TIME_HORIZONS_DAYS})
    min_valid = float(y_test["time"].min())
    max_valid = min(float(y_test["time"].max()), float(y_train["time"].max())) - 1

    valid_pairs = [(d, h) for d, h in zip(TIME_HORIZONS_DAYS, horizons_months) if min_valid < h < max_valid]
    if not valid_pairs:
        return {}

    valid_horizons = [h for _, h in valid_pairs]
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_score_test, valid_horizons)
    result = {f"{d}d (~{h}mo)": float(a) for (d, h), a in zip(valid_pairs, auc)}
    result["mean_auc"] = float(mean_auc)
    return result


# ---------------------------------------------------------------------
# 4. Integrated Brier score
# ---------------------------------------------------------------------
def eval_survival_flat_extrapolated(sf, t: float) -> float:
    """Evaluate a fitted step-function survival curve at t, holding the
    curve flat beyond its fitted domain.

    A GBM survival curve's domain is bounded by the *training* split's
    max observed time (24 months, per the temporal split — see
    time_dependent_auc's docstring for why). The test split's minimum
    follow-up is 36 months, so integrated_brier_score's required
    evaluation grid unavoidably falls outside that domain. Rather than
    leave integrated Brier score uncomputed, we extrapolate flat (hold
    the last fitted survival probability constant past t_max) — a
    standard, explicitly-disclosed approximation, not a claim that the
    model has real information about risk beyond month 24.
    """
    t_clipped = min(max(t, sf.x[0]), sf.x[-1])
    return float(sf(t_clipped))


def integrated_brier(y_train, y_test, survival_probs: np.ndarray, times: np.ndarray) -> float:
    return float(integrated_brier_score(y_train, y_test, survival_probs, times))


# ---------------------------------------------------------------------
# 5. Segment calibration drift
# ---------------------------------------------------------------------
def _calibration_rows_for_dimension(df: pd.DataFrame, dimension_name: str, group_cols: list) -> list:
    group_key = df[group_cols[0]] if len(group_cols) == 1 else df[group_cols].agg(" / ".join, axis=1)
    try:
        lr = multivariate_logrank_test(df["duration_months"], group_key, df["event_observed"])
        p_value = float(lr.p_value)
    except Exception:
        p_value = float("nan")

    rows = []
    for value in sorted(group_key.unique()):
        segment = df[group_key == value]
        if segment.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(segment["duration_months"], segment["event_observed"])
        try:
            observed = float(kmf.survival_function_at_times(CALIBRATION_HORIZON_MONTHS).iloc[0])
        except Exception:
            observed = float(kmf.survival_function_.iloc[-1, 0])
        predicted = float(segment["predicted_survival_12m"].mean())
        n_at_risk = int(len(segment))
        event_count = int(
            (segment["event_observed"] & (segment["duration_months"] <= CALIBRATION_HORIZON_MONTHS)).sum()
        )
        rows.append(
            {
                "segment_dimension": dimension_name,
                "segment_value": str(value),
                "horizon_months": CALIBRATION_HORIZON_MONTHS,
                "predicted_survival": predicted,
                "observed_survival": observed,
                "calibration_error": abs(predicted - observed),
                "logrank_p_value": p_value,
                "n_at_risk": n_at_risk,
                "event_count": event_count,
            }
        )
    return rows


def segment_calibration(test_df: pd.DataFrame, predicted_survival_12m: np.ndarray) -> pd.DataFrame:
    df = test_df.copy()
    df["predicted_survival_12m"] = predicted_survival_12m
    df["tenure_band"] = df["tenure_years"].apply(_tenure_band)
    df["comp_band"] = _comp_band(df["monthly_income"]).astype(str)

    rows = []
    for dimension in ["department", "tenure_band", "comp_band"]:
        rows.extend(_calibration_rows_for_dimension(df, dimension, [dimension]))
    # Cross-tab used by the dashboard's department x tenure-band calibration heatmap.
    rows.extend(_calibration_rows_for_dimension(df, "department_x_tenure_band", ["department", "tenure_band"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 6. Lead-time distribution with bootstrapped CI
# ---------------------------------------------------------------------
def compute_lead_times(test_df: pd.DataFrame, survival_functions, risk_threshold: float) -> pd.DataFrame:
    rows = []
    excluded = 0
    for i, row in enumerate(test_df.itertuples(index=False)):
        if not row.event_observed or row.gbm_risk_score < risk_threshold:
            continue
        sf = survival_functions[i]
        times = sf.x
        survival = sf.y
        cumulative_risk = 1 - survival
        crossing_idx = np.argmax(cumulative_risk >= LEAD_TIME_CUMULATIVE_RISK_THRESHOLD)
        if cumulative_risk[crossing_idx] < LEAD_TIME_CUMULATIVE_RISK_THRESHOLD:
            excluded += 1
            continue
        flagged_month = float(times[crossing_idx])
        departure_month = float(row.duration_months)
        if flagged_month > departure_month:
            excluded += 1
            continue
        rows.append(
            {
                "employee_id": row.employee_id,
                "flagged_month": flagged_month,
                "departure_month": departure_month,
                "lead_time_months": departure_month - flagged_month,
            }
        )
    return pd.DataFrame(rows), excluded


def bootstrap_ci(values: np.ndarray, statistic_fn, n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED) -> tuple:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(statistic_fn(values))
    boot_stats = [
        statistic_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)
    ]
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    return point, float(ci_low), float(ci_high)


# ---------------------------------------------------------------------
# 7. Counterfactual sensitivity (CORRELATIONAL — never causal)
# ---------------------------------------------------------------------
SENSITIVITY_DISCLAIMER = (
    "This is predicted risk sensitivity to a simulated compensation change, estimated from a "
    "correlational model. It is NOT a causal effect of a raise on attrition."
)


def counterfactual_sensitivity(gbm, feature_columns, full_df: pd.DataFrame) -> pd.DataFrame:
    censored = full_df[~full_df["event_observed"]].copy()
    base_risk = gbm_survival.predict_risk(gbm, censored, feature_columns)
    threshold = np.quantile(base_risk, TOP_RISK_QUANTILE)
    high_risk = censored[base_risk >= threshold].copy()
    high_risk_base_risk = base_risk[base_risk >= threshold]

    perturbed = high_risk.copy()
    perturbed["monthly_income"] = perturbed["monthly_income"] * (1 + SENSITIVITY_INCOME_BUMP)
    perturbed_risk = gbm_survival.predict_risk(gbm, perturbed, feature_columns)

    out = pd.DataFrame(
        {
            "employee_id": high_risk["employee_id"].values,
            "base_risk": high_risk_base_risk,
            "perturbed_risk": perturbed_risk,
        }
    )
    out["risk_change"] = out["perturbed_risk"] - out["base_risk"]
    out["risk_change_pct"] = out["risk_change"] / out["base_risk"].replace(0, np.nan)
    out["disclaimer"] = SENSITIVITY_DISCLAIMER
    return out


# ---------------------------------------------------------------------
# 8. SHAP
# ---------------------------------------------------------------------
def compute_shap(gbm, feature_columns, test_df: pd.DataFrame, background_size=100, top_n_sample=200):
    X_test = gbm_survival._to_matrix(test_df, feature_columns)
    rng = np.random.default_rng(RNG_SEED)

    background_idx = rng.choice(len(X_test), size=min(background_size, len(X_test)), replace=False)
    background = X_test[background_idx]

    explainer = shap.Explainer(gbm.predict, background, feature_names=feature_columns)

    sample_idx = rng.choice(len(X_test), size=min(top_n_sample, len(X_test)), replace=False)
    shap_values = explainer(X_test[sample_idx])

    global_importance = pd.DataFrame(
        {"feature": feature_columns, "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)

    risk_scores = gbm.predict(X_test)
    top_decile_threshold = np.quantile(risk_scores, 0.90)
    top_decile_idx = np.where(risk_scores >= top_decile_threshold)[0]
    top_decile_X = X_test[top_decile_idx]
    top_decile_shap = explainer(top_decile_X)

    employee_rows = []
    employee_ids = test_df["employee_id"].values[top_decile_idx]
    for i, emp_id in enumerate(employee_ids):
        for j, feature in enumerate(feature_columns):
            employee_rows.append(
                {
                    "employee_id": int(emp_id),
                    "feature": feature,
                    "feature_value": float(top_decile_X[i, j]),
                    "shap_value": float(top_decile_shap.values[i, j]),
                    "base_value": float(np.atleast_1d(top_decile_shap.base_values[i])[0]),
                }
            )
    return global_importance, pd.DataFrame(employee_rows)


def plot_shap_global(global_importance: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    top = global_importance.head(15).iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("GBM survival model — global feature importance (SHAP)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "shap_global_importance.png"), dpi=120)
    plt.close(fig)


def plot_shap_waterfall_examples(employee_shap: pd.DataFrame, out_dir: str, n_examples=3):
    if employee_shap.empty:
        return
    sample_ids = employee_shap["employee_id"].unique()[:n_examples]
    for emp_id in sample_ids:
        row = employee_shap[employee_shap["employee_id"] == emp_id].sort_values(
            "shap_value", key=np.abs, ascending=True
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#C44E52" if v > 0 else "#4C72B0" for v in row["shap_value"]]
        ax.barh(row["feature"], row["shap_value"], color=colors)
        ax.set_title(f"SHAP contribution — employee {emp_id}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"shap_waterfall_employee_{emp_id}.png"), dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------
def _write_table(engine, df: pd.DataFrame, table: str):
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY"))
    df.to_sql(table, engine, if_exists="append", index=False)


def run():
    """
    Fit/eval design note: build spec Section 3.1 describes the train/test
    split as "employees whose synthetic history ends in the first 24 of
    36 months are train; the remaining chronological window is test."
    Taken as a strict, mutually-exclusive employee partition by tenure
    length, this is mathematically incompatible with the IPCW-based
    metrics Section 5 also requires (time-dependent AUC, integrated
    Brier score): those need the *training* censoring distribution to
    have support at the evaluation times, and a "long-tenure-only" test
    split's minimum follow-up (36mo, since duration_months is always a
    multiple of 12 given real integer-year tenure) falls entirely outside
    a "short-tenure-only" training split's observed range (max 24mo) —
    sksurv raises a hard error ("censoring survival function is zero")
    rather than silently producing a wrong number.

    We implement the same underlying intent — temporal, not random,
    validation; no leakage from outcomes not yet knowable at fit time —
    as a standard rolling-origin design instead: the SAME population is
    used for both fit and eval, with outcomes administratively censored
    at two different snapshots (fit: as of month 24; eval: as of month
    36). This is the standard way time-to-event models are validated
    temporally, and it keeps every metric below mathematically valid.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = get_engine()

    full_df = build_feature_frame(engine)
    check_no_leakage(full_df)

    FIT_CENSOR_MONTHS = 24
    EVAL_CENSOR_MONTHS = 36

    fit_view = full_df.copy()
    fit_view["duration_months"] = fit_view["duration_months"].clip(upper=FIT_CENSOR_MONTHS)
    fit_view["event_observed"] = full_df["event_observed"] & (full_df["duration_months"] <= FIT_CENSOR_MONTHS)

    eval_view = full_df.copy()
    eval_view["duration_months"] = eval_view["duration_months"].clip(upper=EVAL_CENSOR_MONTHS)
    eval_view["event_observed"] = full_df["event_observed"] & (full_df["duration_months"] <= EVAL_CENSOR_MONTHS)

    cph, cox_features = cox_model.fit_cox_model(fit_view)
    gbm, gbm_features = gbm_survival.fit_gbm_model(fit_view)

    # --- 1. PH assumption ---
    fit_encoded = encode_features(fit_view)
    ph_summary = check_ph_assumption(cph, fit_encoded[cox_features + ["duration_months", "event_observed"]])
    ph_summary.to_csv(os.path.join(OUT_DIR, "ph_assumption_check.csv"), index=False)

    # --- risk scores on eval view ---
    cox_risk_eval = cox_model.predict_risk(cph, eval_view, cox_features).values
    gbm_risk_eval = gbm_survival.predict_risk(gbm, eval_view, gbm_features)

    # --- 2. Concordance index ---
    cox_cindex = concordance(eval_view["event_observed"].values, eval_view["duration_months"].values, cox_risk_eval)
    gbm_cindex = concordance(eval_view["event_observed"].values, eval_view["duration_months"].values, gbm_risk_eval)

    cox_within_band = within_tenure_band_concordance(
        eval_view["event_observed"].values, eval_view["duration_months"].values, cox_risk_eval, eval_view["tenure_years"].values
    )
    gbm_within_band = within_tenure_band_concordance(
        eval_view["event_observed"].values, eval_view["duration_months"].values, gbm_risk_eval, eval_view["tenure_years"].values
    )

    # --- 3/4. time-dependent AUC + integrated brier score (GBM) ---
    # Both cumulative_dynamic_auc and integrated_brier_score require every EVAL
    # observation's own time to fall strictly within the FIT view's observed range
    # (they estimate inverse-probability-of-censoring weights from fit_view's
    # censoring distribution, which has no support past fit_view's max time). Since
    # eval_view intentionally extends to month 36 (see run()'s docstring), we
    # restrict these two metrics to the subset of eval_view within fit's range —
    # standard practice for IPCW-based time-dependent metrics; everyone with
    # longer follow-up is exactly the additional right-censored info that made
    # eval_view worth having (it still speaks to concordance/PH-assumption/other
    # metrics above), just not to these two specifically.
    y_train = Surv.from_arrays(fit_view["event_observed"].values.astype(bool), fit_view["duration_months"].values.astype(float))

    fit_max_time = float(fit_view["duration_months"].max())
    auc_mask = eval_view["duration_months"].values < fit_max_time
    eval_auc_view = eval_view[auc_mask]
    gbm_risk_auc = gbm_risk_eval[auc_mask]
    y_test = Surv.from_arrays(eval_auc_view["event_observed"].values.astype(bool), eval_auc_view["duration_months"].values.astype(float))

    auc_results = time_dependent_auc(y_train, y_test, gbm_risk_auc)

    sfs_auc = gbm_survival.predict_survival_functions(gbm, eval_auc_view, gbm_features)
    grid_low = float(y_test["time"].min()) + 1
    grid_high = min(float(y_test["time"].max()), fit_max_time) - 1
    times_grid = np.linspace(grid_low, max(grid_low + 1, grid_high), 15)
    survival_probs = np.array([[eval_survival_flat_extrapolated(sf, t) for t in times_grid] for sf in sfs_auc])
    ibs = integrated_brier(y_train, y_test, survival_probs, times_grid)

    sfs = gbm_survival.predict_survival_functions(gbm, eval_view, gbm_features)

    predicted_survival_12m = np.array(
        [sf(min(CALIBRATION_HORIZON_MONTHS, sf.x[-1])) for sf in sfs]
    )

    metrics_rows = [
        {"model": "cox_ph", "metric_name": "concordance_index", "horizon_months": None, "metric_value": cox_cindex, "ci_low": None, "ci_high": None},
        {"model": "gbm_survival", "metric_name": "concordance_index", "horizon_months": None, "metric_value": gbm_cindex, "ci_low": None, "ci_high": None},
        {"model": "gbm_survival", "metric_name": "integrated_brier_score", "horizon_months": None, "metric_value": ibs, "ci_low": None, "ci_high": None},
        {"model": "cox_ph", "metric_name": "within_tenure_band_concordance", "horizon_months": None, "metric_value": cox_within_band["pooled"], "ci_low": None, "ci_high": None},
        {"model": "gbm_survival", "metric_name": "within_tenure_band_concordance", "horizon_months": None, "metric_value": gbm_within_band["pooled"], "ci_low": None, "ci_high": None},
    ]
    for horizon_label, value in auc_results.items():
        if horizon_label == "mean_auc":
            metrics_rows.append({"model": "gbm_survival", "metric_name": "mean_time_dependent_auc", "horizon_months": None, "metric_value": value, "ci_low": None, "ci_high": None})
        else:
            days = int(horizon_label.split("d")[0])
            metrics_rows.append({"model": "gbm_survival", "metric_name": "time_dependent_auc", "horizon_months": days, "metric_value": value, "ci_low": None, "ci_high": None})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "model_metrics.csv"), index=False)

    within_band_rows = [dict(r, model="cox_ph") for r in cox_within_band["by_band"]] + [
        dict(r, model="gbm_survival") for r in gbm_within_band["by_band"]
    ]
    within_band_df = pd.DataFrame(within_band_rows)
    within_band_df.to_csv(os.path.join(OUT_DIR, "within_tenure_band_concordance.csv"), index=False)

    # --- 5. Segment calibration ---
    calibration_df = segment_calibration(eval_view, predicted_survival_12m)
    calibration_df.to_csv(os.path.join(OUT_DIR, "segment_calibration.csv"), index=False)

    # --- 6. Lead time ---
    eval_with_scores = eval_view.copy()
    eval_with_scores["gbm_risk_score"] = gbm_risk_eval
    risk_threshold = np.quantile(gbm_risk_eval, TOP_RISK_QUANTILE)
    lead_time_df, excluded = compute_lead_times(eval_with_scores, sfs, risk_threshold)
    lead_time_df.to_csv(os.path.join(OUT_DIR, "lead_time_raw.csv"), index=False)

    lead_time_summary_rows = []
    note = (
        f"Flagged cohort = top {(1 - TOP_RISK_QUANTILE):.0%} predicted GBM risk in the eval view "
        f"(risk_score >= {risk_threshold:.4f}). Crossing threshold = first month where the individual's "
        f"predicted cumulative risk (1 - S(t)) >= {LEAD_TIME_CUMULATIVE_RISK_THRESHOLD:.0%}."
    )
    if not lead_time_df.empty:
        for stat_name, fn in [("mean", np.mean), ("median", np.median)]:
            point, ci_low, ci_high = bootstrap_ci(lead_time_df["lead_time_months"].values, fn)
            lead_time_summary_rows.append(
                {
                    "statistic": stat_name,
                    "point_estimate": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_bootstrap": N_BOOTSTRAP,
                    "n_true_positives": len(lead_time_df),
                    "n_excluded_no_crossing": excluded,
                    "risk_threshold_note": note,
                }
            )
    lead_time_summary_df = pd.DataFrame(lead_time_summary_rows)
    lead_time_summary_df.to_csv(os.path.join(OUT_DIR, "lead_time_summary.csv"), index=False)

    # --- 7. Counterfactual sensitivity ---
    sensitivity_df = counterfactual_sensitivity(gbm, gbm_features, full_df)
    sensitivity_df.to_csv(os.path.join(OUT_DIR, "counterfactual_sensitivity.csv"), index=False)

    # --- 8. SHAP ---
    global_importance, employee_shap = compute_shap(gbm, gbm_features, eval_view)
    global_importance.to_csv(os.path.join(OUT_DIR, "shap_global_importance.csv"), index=False)
    employee_shap.to_csv(os.path.join(OUT_DIR, "shap_employee.csv"), index=False)
    plot_shap_global(global_importance, OUT_DIR)
    plot_shap_waterfall_examples(employee_shap, OUT_DIR)

    # --- risk scores table (all employees; data_split is an informational tag —
    # whether the employee's real tenure was originally <=24mo — not a model
    # fit/eval partition, per the rolling-origin design above) ---
    all_cox_risk = cox_model.predict_risk(cph, full_df, cox_features).values
    all_gbm_risk = gbm_survival.predict_risk(gbm, full_df, gbm_features)
    all_sfs = gbm_survival.predict_survival_functions(gbm, full_df, gbm_features)
    all_survival_12m = np.array([sf(min(CALIBRATION_HORIZON_MONTHS, sf.x[-1])) for sf in all_sfs])
    full_risk_threshold = np.quantile(all_gbm_risk, TOP_RISK_QUANTILE)

    risk_scores_df = pd.DataFrame(
        {
            "employee_id": full_df["employee_id"].values,
            "department": full_df["department"].values,
            "tenure_band": full_df["tenure_years"].apply(_tenure_band).values,
            "data_split": full_df["data_split"].values,
            "cox_risk_score": all_cox_risk,
            "gbm_risk_score": all_gbm_risk,
            "gbm_predicted_survival_12m": all_survival_12m,
            "is_top_risk_quartile": all_gbm_risk >= full_risk_threshold,
        }
    )
    risk_scores_df.to_csv(os.path.join(OUT_DIR, "risk_scores.csv"), index=False)

    # --- interaction heatmap (dashboard) ---
    interaction_heatmap_df = interaction_risk_heatmap(full_df, all_gbm_risk)
    interaction_heatmap_df.to_csv(os.path.join(OUT_DIR, "interaction_risk_heatmap.csv"), index=False)

    # --- write to Postgres ---
    _write_table(engine, risk_scores_df, "attrition_risk_scores")
    _write_table(engine, interaction_heatmap_df, "attrition_interaction_heatmap")
    _write_table(engine, metrics_df, "attrition_model_metrics")
    _write_table(engine, ph_summary, "attrition_ph_assumption")
    _write_table(engine, calibration_df, "attrition_calibration")
    _write_table(engine, lead_time_df, "attrition_lead_time")
    _write_table(engine, lead_time_summary_df, "attrition_lead_time_summary")
    _write_table(engine, sensitivity_df, "attrition_sensitivity")
    _write_table(engine, global_importance, "attrition_shap_global")
    _write_table(engine, employee_shap, "attrition_shap_employee")

    print("Attrition evaluation complete.")
    print(f"Cox c-index: {cox_cindex:.3f} | GBM c-index: {gbm_cindex:.3f} | GBM IBS: {ibs:.3f}")
    print(
        f"Within-tenure-band c-index (controls for tenure_years dominance): "
        f"Cox {cox_within_band['pooled']:.3f} | GBM {gbm_within_band['pooled']:.3f}"
    )
    return {
        "cox_cindex": cox_cindex,
        "gbm_cindex": gbm_cindex,
        "integrated_brier_score": ibs,
        "time_dependent_auc": auc_results,
    }


if __name__ == "__main__":
    run()
