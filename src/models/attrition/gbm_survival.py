"""Gradient-boosted survival attrition model
(sksurv.ensemble.GradientBoostingSurvivalAnalysis) — same train split and
same feature set as the Cox model, for a like-for-like comparison."""

import numpy as np
import pandas as pd
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv

from src.models.attrition.features import NON_FEATURE_COLUMNS, encode_features


def get_feature_columns(encoded_df: pd.DataFrame) -> list:
    return [c for c in encoded_df.columns if c not in NON_FEATURE_COLUMNS]


def fit_gbm_model(train_df: pd.DataFrame, random_state: int = 42) -> tuple:
    encoded = encode_features(train_df)
    feature_columns = get_feature_columns(encoded)

    X = encoded[feature_columns].astype(float).values
    y = Surv.from_arrays(
        event=train_df["event_observed"].astype(bool).values,
        time=train_df["duration_months"].astype(float).values,
    )

    model = GradientBoostingSurvivalAnalysis(
        n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=random_state
    )
    model.fit(X, y)
    return model, feature_columns


def _to_matrix(df: pd.DataFrame, feature_columns: list) -> np.ndarray:
    encoded = encode_features(df)
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded[feature_columns].astype(float).values


def predict_risk(model: GradientBoostingSurvivalAnalysis, df: pd.DataFrame, feature_columns: list) -> np.ndarray:
    X = _to_matrix(df, feature_columns)
    return model.predict(X)


def predict_survival_functions(model: GradientBoostingSurvivalAnalysis, df: pd.DataFrame, feature_columns: list):
    X = _to_matrix(df, feature_columns)
    return model.predict_survival_function(X)
