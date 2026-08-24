"""One-off, targeted recompute of ONLY attrition_sensitivity, after
counterfactual_sensitivity() in evaluate.py was changed to cover every
employee instead of a top-risk-quantile subset of censored employees
only. Replicates evaluate.py's own fit_view construction and GBM fit
exactly (same RNG_SEED, same temporal-censoring setup) so the sensitivity
numbers stay consistent with the rest of the attrition_* tables, without
re-running and rewriting every other table those don't need to change.
Run as: python -m src.models.attrition.recompute_sensitivity
"""

from src.db.connection import get_engine
from src.models.attrition import gbm_survival
from src.models.attrition.evaluate import _write_table, counterfactual_sensitivity
from src.models.attrition.features import build_feature_frame, check_no_leakage

FIT_CENSOR_MONTHS = 24


def main():
    engine = get_engine()
    full_df = build_feature_frame(engine)
    check_no_leakage(full_df)

    fit_view = full_df.copy()
    fit_view["duration_months"] = fit_view["duration_months"].clip(upper=FIT_CENSOR_MONTHS)
    fit_view["event_observed"] = full_df["event_observed"] & (full_df["duration_months"] <= FIT_CENSOR_MONTHS)

    gbm, gbm_features = gbm_survival.fit_gbm_model(fit_view)

    sensitivity_df = counterfactual_sensitivity(gbm, gbm_features, full_df)
    print(f"Recomputed sensitivity for {len(sensitivity_df)} employees (full_df has {len(full_df)}).")

    _write_table(engine, sensitivity_df, "attrition_sensitivity")
    print("attrition_sensitivity table rewritten.")


if __name__ == "__main__":
    main()
