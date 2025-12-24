import pickle
from pathlib import Path


class BaselineStore:
    def __init__(self, path="data/baseline.pkl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, baseline: dict):
        with open(self.path, "wb") as f:
            pickle.dump(baseline, f)

    def load(self):
        if not self.path.exists():
            return None
        with open(self.path, "rb") as f:
            return pickle.load(f)
