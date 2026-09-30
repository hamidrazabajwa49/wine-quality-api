"""
Loads the saved model and exposes predict()
"""

import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_path: str):
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.features = artifact["features"]
        self.version = artifact["version"]
        self.metrics = artifact.get("metrics", {})

    def predict(self, values: dict) -> float:
        row = pd.DataFrame([[values[f] for f in self.features]], columns=self.features)
        return float(self.pipeline.predict(row)[0])
