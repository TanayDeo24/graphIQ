"""Isolation Forest point-anomaly detector for expense transactions."""

import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.spend.features import get_feature_matrix


def fit_isolation_forest(df, random_state: int = 42) -> IsolationForest:
    X = get_feature_matrix(df)
    model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=random_state, n_jobs=-1
    )
    model.fit(X)
    return model


def score(model: IsolationForest, df) -> np.ndarray:
    """Higher score = more anomalous (sklearn's score_samples is the opposite convention)."""
    X = get_feature_matrix(df)
    return -model.score_samples(X)
