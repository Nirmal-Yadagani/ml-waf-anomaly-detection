from datetime import datetime
import math


class ConfidenceDecay:
    def __init__(self, decay_rate: float = 0.1):
        """
        decay_rate: higher = faster decay
        """
        self.decay_rate = decay_rate

    def apply(self, rule: dict, now: datetime = None) -> dict:
        now = now or datetime.utcnow()

        created_at = rule.get("created_at")
        if not created_at:
            return rule

        age_days = (now - created_at).days
        decayed = rule["confidence"] * math.exp(
            -self.decay_rate * age_days
        )

        rule["confidence_decayed"] = round(decayed, 3)
        rule["age_days"] = age_days

        # Auto-expire suggestion
        if rule["confidence_decayed"] < 0.5:
            rule["status"] = "expired"

        return rule
