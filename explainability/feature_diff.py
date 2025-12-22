import pandas as pd
from typing import Dict


class FeatureDiff:
    def __init__(self, baseline: Dict[str, Dict[str, float]]):
        """
        baseline format:
        {
          feature: {
            mean: float,
            std: float,
            p95: float,
            p99: float
          }
        }
        """
        self.baseline = baseline

    def diff(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Returns per-feature deviation metrics for each row
        """
        diffs = {}

        for col in features.columns:
            b = self.baseline[col]

            diffs[f"{col}_z"] = (
                (features[col] - b["mean"]) / b["std"]
            )
            diffs[f"{col}_above_p95"] = features[col] > b["p95"]
            diffs[f"{col}_above_p99"] = features[col] > b["p99"]

        return pd.DataFrame(diffs, index=features.index)
