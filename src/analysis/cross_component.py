"""Cross-component analysis: does an employee's attrition risk relate to
their spend-anomaly signal? Both components read from the same
`employees` table (the project's core architectural thesis — see README)
but have never been directly compared against each other until now.

Employee-level spend-anomaly score: count of an employee's transactions
flagged at the operating threshold (predicted_flag = TRUE in
spend_anomaly_scores).

A first attempt used max(ensemble_score) across ALL of an employee's
transactions instead. That turned out to be a real bug, not just a
different reasonable choice: ensemble_score is rank-normalized globally
across ~400k transactions, and employees average ~275 transactions each —
the expected max of that many draws from a rank-normalized [0,1]
distribution is already ~0.996 for nearly everyone, regardless of whether
their spending actually looked anomalous. It's an order-statistics
artifact of "take the max over many draws," not a meaningful per-employee
signal — confirmed by inspecting the quadrant scatter, where nearly every
point sat within a few pixels of the very top of the y-axis. Flagged-
transaction count doesn't have this problem (it's a genuine count, not an
order statistic of a globally-normalized score) and is one of the two
options this analysis was scoped to choose between.

"High risk" / "high anomaly" thresholds reuse the project's own existing
top-quartile convention (TOP_RISK_QUANTILE = 0.75 in
src/models/attrition/evaluate.py) rather than inventing a new one — applied
to the employee-level spend-anomaly score the same way it's already applied
to gbm_risk_score.

CORRELATIONAL, NOT CAUSAL — same framing as the counterfactual sensitivity
analysis elsewhere in this project. Any relationship found here describes
an observed association between two model outputs on synthetic/real-hybrid
data, never a claim that attrition risk causes (or is caused by) anomalous
spend, or that either is a real signal about any actual person.

PARTIAL CORRELATION CHECK: the bivariate Spearman correlation above is
vulnerable to confounding -- department and income both plausibly drive
*both* scores independently (e.g. some departments may have both higher
attrition risk and different spend patterns for reasons unrelated to any
real relationship between the two). `partial_spearman_with_permutation_test()`
controls for department and monthly_income (the continuous compensation
field the attrition model actually keys off -- see
FEATURE_COLUMNS_NUMERIC in src/models/attrition/features.py; comp_band is
a derived bucket of the same field and isn't separately consumed by the
model, so monthly_income is the more direct confound to control for) via
OLS-residualized partial Spearman correlation, no pingouin dependency
needed. Reported alongside the bivariate figure in the summary table, API
response, and README -- whichever way the partial result comes out.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sqlalchemy import text

from src.db.connection import get_engine

OUT_DIR = "data/generated/cross_component_outputs"
TOP_QUANTILE = 0.75  # reused from src/models/attrition/evaluate.py's TOP_RISK_QUANTILE
N_PERMUTATIONS = 1000
RNG_SEED = 17

CORRELATIONAL_DISCLAIMER = (
    "This is an observed association between two model outputs (attrition risk score and spend-anomaly "
    "score) on synthetic/real-hybrid data. It is NOT a causal claim that one relates to or predicts the "
    "other in any real workforce, and is not a real finding about any person or company."
)


def load_attrition_risk(engine) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT ars.employee_id, ars.department, ars.tenure_band, ars.gbm_risk_score, "
        "ars.is_top_risk_quartile, e.monthly_income "
        "FROM attrition_risk_scores ars JOIN employees e ON e.employee_id = ars.employee_id",
        engine,
    )


def load_spend_anomaly_by_employee(engine) -> pd.DataFrame:
    """Count of flagged transactions per employee -- includes employees
    with zero flagged transactions (a real, common value, not missing
    data). Employees entirely absent from spend_anomaly_scores (no
    transactions at all) are handled separately in build_joined_frame."""
    all_employees = pd.read_sql("SELECT DISTINCT employee_id FROM spend_anomaly_scores", engine)
    flagged_counts = pd.read_sql(
        "SELECT employee_id, COUNT(*) AS spend_anomaly_score FROM spend_anomaly_scores "
        "WHERE predicted_flag = TRUE GROUP BY employee_id",
        engine,
    )
    merged = all_employees.merge(flagged_counts, on="employee_id", how="left")
    merged["spend_anomaly_score"] = merged["spend_anomaly_score"].fillna(0).astype(int)
    return merged


def build_joined_frame(engine) -> pd.DataFrame:
    attrition = load_attrition_risk(engine)
    spend = load_spend_anomaly_by_employee(engine)
    df = attrition.merge(spend, on="employee_id", how="left")

    n_missing = df["spend_anomaly_score"].isna().sum()
    if n_missing:
        # Every employee has >=1 generated transaction under this project's
        # generation design (window_length_months floors at 1) -- if any are
        # missing here it means they had zero transactions in the primary
        # 5% dataset. Documented, not silently imputed: excluded from the
        # analysis rather than back-filled with a fabricated score.
        df = df.dropna(subset=["spend_anomaly_score"])

    threshold = np.quantile(df["spend_anomaly_score"], TOP_QUANTILE)
    df["is_top_spend_quartile"] = df["spend_anomaly_score"] >= threshold

    def _quadrant(row):
        risk = "high_risk" if row["is_top_risk_quartile"] else "low_risk"
        anomaly = "high_anomaly" if row["is_top_spend_quartile"] else "low_anomaly"
        return f"{risk}_{anomaly}"

    df["quadrant"] = df.apply(_quadrant, axis=1)
    return df, n_missing


def spearman_with_permutation_test(x: np.ndarray, y: np.ndarray, n_permutations: int = N_PERMUTATIONS,
                                    seed: int = RNG_SEED) -> dict:
    """Spearman rank correlation (scores aren't assumed normally
    distributed) with a permutation-test p-value: shuffle y n_permutations
    times, recompute the correlation each time, and see how extreme the
    real (unshuffled) correlation is relative to that null distribution.
    Chosen over the parametric Spearman p-value because it makes no
    distributional assumption at all beyond exchangeability under the
    null -- appropriate here since neither score's distribution is
    otherwise characterized or assumed.
    """
    observed_corr, _ = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    permuted_corrs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_y = rng.permutation(y)
        permuted_corrs[i], _ = spearmanr(x, shuffled_y)
    p_value = float((np.abs(permuted_corrs) >= np.abs(observed_corr)).mean())
    return {
        "spearman_correlation": float(observed_corr),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


PARTIAL_CORR_METHOD_NOTE = (
    "Partial Spearman correlation controlling for department and monthly_income "
    "(no pingouin dependency -- implemented directly via OLS residualization, documented "
    "here): rank-transform gbm_risk_score, spend_anomaly_score, and monthly_income; "
    "one-hot encode department; regress each of ranked-risk and ranked-anomaly on "
    "[department dummies, ranked income] and take residuals; the partial Spearman "
    "correlation is the Pearson correlation of the two residual series. Significance via "
    "a two-sided permutation test on the risk-score residual (Freedman-Lane-style: "
    "residuals, not raw scores, are permuted, since under the null of 'no partial "
    "association given the confounds' the residuals -- not the raw, confound-carrying "
    "scores -- are exchangeable)."
)


def _confound_design_matrix(confounds: pd.DataFrame) -> np.ndarray:
    department_dummies = pd.get_dummies(confounds["department"], drop_first=True).astype(float).values
    income_rank = rankdata(confounds["monthly_income"].values)
    return np.column_stack([department_dummies, income_rank])


def partial_spearman_with_permutation_test(x: np.ndarray, y: np.ndarray, confounds: pd.DataFrame,
                                            n_permutations: int = N_PERMUTATIONS, seed: int = RNG_SEED) -> dict:
    """See PARTIAL_CORR_METHOD_NOTE for the full method. `confounds` must
    have `department` and `monthly_income` columns, row-aligned with x/y."""
    rank_x = rankdata(x)
    rank_y = rankdata(y)
    z = _confound_design_matrix(confounds)
    design = np.column_stack([np.ones(len(x)), z])
    # QR gives an orthonormal basis for the confound column space; projecting
    # off it (v - Q @ Q.T @ v) is the OLS residual, and reusing the same Q for
    # every permutation keeps each permuted residual an O(n*k) mat-vec instead
    # of refitting a regression per permutation.
    q, _ = np.linalg.qr(design)

    def _residualize(v):
        return v - q @ (q.T @ v)

    resid_x = _residualize(rank_x)
    resid_y = _residualize(rank_y)
    observed_corr, _ = pearsonr(resid_x, resid_y)

    rng = np.random.default_rng(seed)
    permuted_corrs = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_resid_x = rng.permutation(resid_x)
        permuted_corrs[i], _ = pearsonr(shuffled_resid_x, resid_y)
    p_value = float((np.abs(permuted_corrs) >= np.abs(observed_corr)).mean())

    return {
        "partial_spearman_correlation": float(observed_corr),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "confounds": "department, monthly_income",
        "method_note": PARTIAL_CORR_METHOD_NOTE,
    }


def quadrant_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quadrant, group in df.groupby("quadrant"):
        n = len(group)
        for dimension in ["department", "tenure_band"]:
            counts = group[dimension].value_counts()
            for value, count in counts.items():
                rows.append(
                    {
                        "quadrant": quadrant,
                        "dimension": dimension,
                        "dimension_value": value,
                        "count": int(count),
                        "pct_of_quadrant": float(count / n),
                    }
                )
    return pd.DataFrame(rows)


def _write_table(engine, df: pd.DataFrame, table: str):
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY"))
    if not df.empty:
        df.to_sql(table, engine, if_exists="append", index=False)


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    engine = get_engine()

    df, n_missing = build_joined_frame(engine)

    stats = spearman_with_permutation_test(df["gbm_risk_score"].values, df["spend_anomaly_score"].values)
    method_note = (
        "Spearman rank correlation (ranked, not raw scores -- neither is assumed normally distributed); "
        f"significance via a two-sided permutation test ({N_PERMUTATIONS} permutations of the spend-anomaly "
        "score, p = fraction of permuted |correlation| >= observed |correlation|)."
    )

    partial_stats = partial_spearman_with_permutation_test(
        df["gbm_risk_score"].values, df["spend_anomaly_score"].values, df[["department", "monthly_income"]]
    )

    summary_df = pd.DataFrame(
        [
            {
                "spearman_correlation": stats["spearman_correlation"],
                "p_value": stats["p_value"],
                "n_permutations": stats["n_permutations"],
                "n_employees": len(df),
                "method_note": method_note,
                "disclaimer": CORRELATIONAL_DISCLAIMER,
                "partial_spearman_correlation": partial_stats["partial_spearman_correlation"],
                "partial_p_value": partial_stats["p_value"],
                "partial_confounds": partial_stats["confounds"],
                "partial_method_note": partial_stats["method_note"],
            }
        ]
    )

    characteristics_df = quadrant_characteristics(df)
    quadrant_counts = df["quadrant"].value_counts().to_dict()

    quadrant_df = df[
        ["employee_id", "department", "tenure_band", "gbm_risk_score", "is_top_risk_quartile",
         "spend_anomaly_score", "is_top_spend_quartile", "quadrant"]
    ]

    quadrant_df.to_csv(os.path.join(OUT_DIR, "quadrant_assignments.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_DIR, "correlation_summary.csv"), index=False)
    characteristics_df.to_csv(os.path.join(OUT_DIR, "quadrant_characteristics.csv"), index=False)

    _write_table(engine, quadrant_df, "cross_component_quadrant")
    _write_table(engine, summary_df, "cross_component_summary")
    _write_table(engine, characteristics_df, "cross_component_quadrant_characteristics")

    print(f"n_employees={len(df)} (excluded {n_missing} with no transactions)")
    print(f"Spearman correlation (bivariate): {stats['spearman_correlation']:.4f} (p={stats['p_value']:.4f}, {N_PERMUTATIONS} permutations)")
    print(f"Spearman correlation (partial, controlling for department + monthly_income): "
          f"{partial_stats['partial_spearman_correlation']:.4f} (p={partial_stats['p_value']:.4f}, {N_PERMUTATIONS} permutations)")
    print("Quadrant counts:", quadrant_counts)
    print(CORRELATIONAL_DISCLAIMER)
    return {"stats": stats, "partial_stats": partial_stats, "quadrant_counts": quadrant_counts, "n_employees": len(df)}


if __name__ == "__main__":
    run()
