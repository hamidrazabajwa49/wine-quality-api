import joblib
import numpy as np
import pandas as pd
import pytest

from wine_api.data import FEATURES
from wine_api.predictor import Predictor
from wine_api.train import build_pipeline


@pytest.fixture
def model_path(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, len(FEATURES))), columns=FEATURES)
    y = rng.integers(3, 9, 50)
    ranges = {f: [0.0, 1.0] for f in FEATURES}
    path = tmp_path / "m.joblib"
    joblib.dump(
        {"pipeline": build_pipeline().fit(X, y), "features": FEATURES, "version": "t", "feature_ranges": ranges},
        path,
    )
    return str(path)


def sample(value=0.5) -> dict:
    return {f: value for f in FEATURES}


def test_predictor_roundtrip(model_path):
    p = Predictor(model_path)
    assert isinstance(p.predict(sample()), float)


def test_predict_many_band_contains_mean(model_path):
    results = Predictor(model_path).predict_many([sample(0.2), sample(0.8)])
    assert len(results) == 2
    for r in results:
        assert r.low <= r.quality <= r.high


def test_out_of_range_input_warns(model_path):
    p = Predictor(model_path)
    assert p.predict_many([sample(0.5)])[0].warnings == []
    warnings = p.predict_many([{**sample(0.5), "alcohol": 5.0}])[0].warnings
    assert len(warnings) == 1 and "alcohol" in warnings[0]


def test_feature_importances_sorted(model_path):
    values = list(Predictor(model_path).feature_importances.values())
    assert values == sorted(values, reverse=True)


def test_missing_model_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="wine_api.train"):
        Predictor(str(tmp_path / "nope.joblib"))
