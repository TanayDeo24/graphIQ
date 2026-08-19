"""Cox Proportional Hazards attrition model (lifelines.CoxPHFitter)."""

import pandas as pd
from lifelines import CoxPHFitter

from src.models.attrition.features import NON_FEATURE_COLUMNS, encode_features


def get_feature_columns(encoded_df: pd.DataFrame) -> list:
    return [c for c in encoded_df.columns if c not in NON_FEATURE_COLUMNS]


def fit_cox_model(train_df: pd.DataFrame, penalizer: float = 0.1) -> tuple:
    encoded = encode_features(train_df)
    feature_columns = get_feature_columns(encoded)

    model_df = encoded[feature_columns + ["duration_months", "event_observed"]].copy()
    model_df["event_observed"] = model_df["event_observed"].astype(int)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(model_df, duration_col="duration_months", event_col="event_observed")
    return cph, feature_columns


def predict_risk(cph: CoxPHFitter, df: pd.DataFrame, feature_columns: list) -> pd.Series:
    encoded = encode_features(df)
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    return cph.predict_partial_hazard(encoded[feature_columns])


def predict_survival_function(cph: CoxPHFitter, df: pd.DataFrame, feature_columns: list, times=None) -> pd.DataFrame:
    encoded = encode_features(df)
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    return cph.predict_survival_function(encoded[feature_columns], times=times)
