import pandas as pd
from typing import List, Dict

from explainability.templates import ExplanationTemplates


class ExplanationBuilder:
    def __init__(self, baseline: Dict[str, Dict[str, float]]):
        self.baseline = baseline

    def build(
        self,
        features: pd.DataFrame,
        diffs: pd.DataFrame,
        min_z: float = 3.0,
    ) -> Dict[int, List[str]]:
        """
        Build explanations for each row
        Returns:
          { index: [explanations...] }
        """

        explanations = {}

        for idx in features.index:
            reasons = []

            for feature in self.baseline.keys():
                z_col = f"{feature}_z"
                p99_col = f"{feature}_above_p99"

                if z_col not in diffs.columns:
                    continue

                z = diffs.loc[idx, z_col]
                above_p99 = diffs.loc[idx, p99_col]

                # Force scalar values
                if hasattr(z, "__len__"):
                    z = float(z.iloc[0]) if hasattr(z, "iloc") else float(z)

                if hasattr(above_p99, "__len__"):
                    above_p99 = bool(above_p99.iloc[0]) if hasattr(above_p99, "iloc") else bool(above_p99)

                if abs(z) >= min_z or above_p99:
                    text = ExplanationTemplates.render(
                        feature=feature,
                        value=features.loc[idx, feature],
                        p99=self.baseline[feature]["p99"],
                    )
                    if text:
                        reasons.append(text)

            explanations[idx] = reasons

        return explanations
