# anomaly_detection/scorer.py

import pandas as pd
from typing import Dict


class AnomalyScorer:
    def __init__(
        self,
        if_weight: float = 0.6,
        baseline_weight: float = 0.4,
        anomaly_threshold: float = 0.75,
    ):
        assert abs(if_weight + baseline_weight - 1.0) < 1e-6
        self.if_weight = if_weight
        self.baseline_weight = baseline_weight
        self.anomaly_threshold = anomaly_threshold

    def score(
        self,
        if_scores: pd.Series,
        baseline_scores: pd.Series,
    ) -> pd.DataFrame:
        """
        Combine Isolation Forest + baseline deviation
        """

        final_score = (
            self.if_weight * if_scores
            + self.baseline_weight * baseline_scores
        )

        result = pd.DataFrame(
            {
                "if_score": if_scores,
                "baseline_score": baseline_scores,
                "final_score": final_score,
                "is_anomaly": final_score > self.anomaly_threshold,
            }
        )

        return result
