"""
Loads the saved model and exposes predict()
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90


@dataclass(frozen=True)
class Prediction:
    quality: float
    low: float
    high: float
    warnings: list[str] = field(default_factory=list)


class Predictor:
    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found at {path}. Train it first: python -m wine_api.train")
        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        self.features = artifact["features"]
        self.version = artifact["version"]
        self.metrics = artifact.get("metrics", {})
        self.feature_ranges = artifact.get("feature_ranges", {})
        self._preprocess = self.pipeline[:-1]
        self._forest = self.pipeline[-1]

    @property
    def feature_importances(self) -> dict[str, float]:
        pairs = zip(self.features, self._forest.feature_importances_)
        ranked = sorted(pairs, key=lambda p: p[1], reverse=True)
        return {name: round(float(value), 4) for name, value in ranked}

    def predict(self, values: dict) -> float:
        return self.predict_many([values])[0].quality

    def predict_many(self, rows: list[dict]) -> list[Prediction]:
        frame = pd.DataFrame([[r[f] for f in self.features] for r in rows], columns=self.features)
        X = self._preprocess.transform(frame)
        # The spread across individual trees gives a cheap uncertainty band
        per_tree = np.stack([tree.predict(X) for tree in self._forest.estimators_])
        mean = per_tree.mean(axis=0)
        low, high = np.percentile(per_tree, [LOW_PERCENTILE, HIGH_PERCENTILE], axis=0)
        return [
            Prediction(float(m), float(lo), float(hi), self._range_warnings(row))
            for m, lo, hi, row in zip(mean, low, high, rows)
        ]

    def _range_warnings(self, row: dict) -> list[str]:
        return [
            f"{name}={row[name]} is outside the training range [{lo}, {hi}]"
            for name, (lo, hi) in self.feature_ranges.items()
            if not lo <= row[name] <= hi
        ]
