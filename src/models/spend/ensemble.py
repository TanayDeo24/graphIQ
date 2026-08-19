"""Combine the three spend detectors (Isolation Forest, CUSUM, autoencoder)
into one ensemble anomaly score: rank-normalize each detector's score to
[0, 1], then average. Individual detector scores are kept alongside the
ensemble score so each one's standalone contribution/performance stays
visible rather than being hidden inside the combined number.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    ranks = rankdata(scores, method="average")
    return (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else np.zeros_like(ranks, dtype=float)


def combine(isolation_forest_score: np.ndarray, autoencoder_score: np.ndarray, cusum_statistic: np.ndarray) -> pd.DataFrame:
    isf_rank = rank_normalize(isolation_forest_score)
    ae_rank = rank_normalize(autoencoder_score)
    cusum_rank = rank_normalize(cusum_statistic)

    ensemble_score = (isf_rank + ae_rank + cusum_rank) / 3.0

    return pd.DataFrame(
        {
            "isolation_forest_score": isolation_forest_score,
            "isolation_forest_rank": isf_rank,
            "autoencoder_score": autoencoder_score,
            "autoencoder_rank": ae_rank,
            "cusum_statistic": cusum_statistic,
            "cusum_rank": cusum_rank,
            "ensemble_score": ensemble_score,
        }
    )
