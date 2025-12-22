# anomaly_detection/isolation_forest.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.02,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.fitted = False

    def fit(self, X: pd.DataFrame):
        """
        Train Isolation Forest on normal traffic
        """
        self.model.fit(X)
        self.fitted = True

    def score(self, X: pd.DataFrame) -> pd.Series:
        """
        Return normalized anomaly score ∈ [0, 1]
        Higher = more anomalous
        """
        if not self.fitted:
            raise RuntimeError("IsolationForestModel is not fitted")

        # sklearn: higher = more normal → invert
        raw_scores = -self.model.score_samples(X)

        # Min-max normalization
        min_s, max_s = raw_scores.min(), raw_scores.max()
        norm_scores = (raw_scores - min_s) / (max_s - min_s + 1e-6)

        return pd.Series(norm_scores, index=X.index)
