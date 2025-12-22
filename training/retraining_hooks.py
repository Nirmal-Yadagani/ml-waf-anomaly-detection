from datetime import datetime, timedelta


class RetrainingHooks:
    def __init__(
        self,
        max_fp_rate: float = 0.2,
        drift_threshold: float = 0.4,
        retrain_interval_days: int = 7,
    ):
        self.max_fp_rate = max_fp_rate
        self.drift_threshold = drift_threshold
        self.retrain_interval = timedelta(days=retrain_interval_days)
        self.last_retrain = None

    def should_retrain(
        self,
        fp_rate: float,
        avg_baseline_score: float,
        now: datetime = None,
    ) -> bool:
        now = now or datetime.utcnow()

        if fp_rate > self.max_fp_rate:
            return True

        if avg_baseline_score > self.drift_threshold:
            return True

        if self.last_retrain is None:
            return True

        if now - self.last_retrain > self.retrain_interval:
            return True

        return False

    def mark_retrained(self):
        self.last_retrain = datetime.utcnow()
