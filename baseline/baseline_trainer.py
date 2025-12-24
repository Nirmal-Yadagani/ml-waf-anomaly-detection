import pandas as pd
import numpy as np
from typing import Dict


class BaselineTrainer:
    def __init__(self):
        self.baseline: Dict[str, Dict[str, float]] = {}

    # ------------------------------
    # Initial fit (cold start)
    # ------------------------------
    def fit(self, features: pd.DataFrame):
        """
        Learn baseline statistics from scratch
        """
        self.baseline = self._compute_stats(features)

    # ------------------------------
    # Incremental adaptive update
    # ------------------------------
    def update(self, features: pd.DataFrame, alpha: float = 0.1):
        """
        Slowly adapt baseline using exponential moving average
        alpha: adaptation rate (0.05–0.2 recommended)
        """
        new_stats = self._compute_stats(features)

        if not self.baseline:
            self.baseline = new_stats
            return

        for feature in self.baseline:
            for k in ["mean", "std", "p95", "p99"]:
                self.baseline[feature][k] = (
                    (1 - alpha) * self.baseline[feature][k]
                    + alpha * new_stats[feature][k]
                )

    # ------------------------------
    # Scoring
    # ------------------------------
    def score_deviation(self, features: pd.DataFrame) -> pd.Series:
        scores = []

        for col in features.columns:
            b = self.baseline[col]
            z = (features[col] - b["mean"]) / (b["std"] + 1e-6)
            score = np.clip(np.abs(z) / 3.0, 0, 1)
            scores.append(score)

        return pd.concat(scores, axis=1).mean(axis=1)

    def get_baseline(self) -> Dict[str, Dict[str, float]]:
        return self.baseline

    # ------------------------------
    # Internal helper
    # ------------------------------
    def _compute_stats(self, features: pd.DataFrame):
        stats = {}
        for col in features.columns:
            stats[col] = {
                "mean": features[col].mean(),
                "std": features[col].std() + 1e-6,
                "p95": features[col].quantile(0.95),
                "p99": features[col].quantile(0.99),
            }
        return stats
