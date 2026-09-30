import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .config import settings
from .data import FEATURES, TARGET, load_data
from .logger import get_logger

log = get_logger(__name__)

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def evaluate(y_true, preds, baseline_pred: float) -> dict:
    err = np.abs(np.asarray(y_true) - preds)
    return {
        "baseline_mae": round(float(mean_absolute_error(y_true, np.full(len(y_true), baseline_pred))), 3),
        "mae": round(float(mean_absolute_error(y_true, preds)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, preds))), 3),
        "r2": round(float(r2_score(y_true, preds)), 3),
        "within_half_point": round(float((err <= 0.5).mean()), 3),
    }


def main() -> None:
    df = load_data(settings.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    pipeline = build_pipeline().fit(X_train, y_train)
    metrics = evaluate(y_test, pipeline.predict(X_test), baseline_pred=float(y_train.mean()))
    metrics.update(n_train=len(X_train), n_test=len(X_test))
    log.info("metrics: %s", metrics)

    joblib.dump(
        {"pipeline": pipeline, "features": FEATURES, "version": __version__, "metrics": metrics},
        settings.model_path,
    )
    Path(settings.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.metrics_path).write_text(json.dumps(metrics, indent=2))
    log.info("saved model to %s", settings.model_path)


if __name__ == "__main__":
    main()
