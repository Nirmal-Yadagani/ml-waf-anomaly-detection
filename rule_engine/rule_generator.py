# rule_engine/rule_generator.py

from collections import defaultdict, Counter
from typing import Dict, List
import yaml
from datetime import datetime

class RuleGenerator:
    def __init__(
        self,
        template_path: str,
        min_occurrences: int = 3,
        min_avg_score: float = 0.8,
    ):
        with open(template_path, "r") as f:
            self.templates = yaml.safe_load(f)

        self.min_occurrences = min_occurrences
        self.min_avg_score = min_avg_score

    def generate(
        self,
        context: Dict,
        results,
    ) -> List[Dict]:
        """
        context: DataFrame-like dict with uri_path, src_ip, method
        results: DataFrame with is_anomaly, final_score, explanations
        """

        grouped = defaultdict(list)

        # Group anomalies by endpoint + IP
        for idx, row in results.iterrows():
            if not row["is_anomaly"]:
                continue

            key = (
                context.loc[idx, "uri_path"],
                context.loc[idx, "src_ip"],
            )

            grouped[key].append(row)

        rules = []

        for (endpoint, src_ip), rows in grouped.items():
            if len(rows) < self.min_occurrences:
                continue

            avg_score = sum(r["final_score"] for r in rows) / len(rows)
            if avg_score < self.min_avg_score:
                continue

            explanations = []
            for r in rows:
                explanations.extend(r["explanations"])

            rule = self._build_rule(
                endpoint=endpoint,
                src_ip=src_ip,
                explanations=explanations,
                confidence=avg_score,
            )

            if rule:
                rules.append(rule)

        return rules

    # -------------------------------------------------------------

    def _build_rule(
    self,
    endpoint: str,
    src_ip: str,
    explanations: list,
    confidence: float,
) -> dict:

        explanation_counts = Counter(explanations)

        for rule_type, spec in self.templates.items():
            for trigger in spec["triggers"]:
                if any(trigger in e for e in explanation_counts):

                    return {
                        "rule_type": rule_type,

                        # --- match condition ---
                        "match": {
                            "endpoint": endpoint,
                            "src_ip": src_ip,
                        },

                        # --- action ---
                        "action": spec["action"],

                        # --- lifecycle metadata ---
                        "confidence": round(confidence, 2),
                        "confidence_decayed": round(confidence, 2),
                        "created_at": datetime.utcnow(),
                        "status": "proposed",

                        # --- explainability ---
                        "evidence": explanation_counts.most_common(3),
                    }

        return None