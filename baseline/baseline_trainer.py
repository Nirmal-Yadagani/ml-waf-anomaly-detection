import pandas as pd
import numpy as np
from typing import Dict


class BaselineTrainer:
    def __init__(self):
        self.baseline = {}

    def fit(self, features: pd.DataFrame):
        """
        Learn baseline statistics for each feature
        """
        for col in features.columns:
            self.baseline[col] = {
                "mean": features[col].mean(),
                "std": features[col].std() + 1e-6,
                "p95": features[col].quantile(0.95),
                "p99": features[col].quantile(0.99),
            }

    def get_baseline(self) -> Dict[str, Dict[str, float]]:
        return self.baseline

    def score_deviation(self, features: pd.DataFrame) -> pd.Series:
        """
        Compute baseline deviation score ∈ [0, 1]
        """
        scores = []

        for col in features.columns:
            baseline = self.baseline[col]
            z = (features[col] - baseline["mean"]) / baseline["std"]
            score = np.clip(np.abs(z) / 3.0, 0, 1)
            scores.append(score)

        return pd.concat(scores, axis=1).mean(axis=1)
