import joblib
import numpy as np
import pandas as pd

from wine_api.data import FEATURES
from wine_api.predictor import Predictor
from wine_api.train import build_pipeline


def test_predictor_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, len(FEATURES))), columns=FEATURES)
    y = rng.integers(3, 9, 50)
    path = tmp_path / "m.joblib"
    joblib.dump({"pipeline": build_pipeline().fit(X, y), "features": FEATURES, "version": "t"}, path)

    p = Predictor(str(path))
    assert isinstance(p.predict(X.iloc[0].to_dict()), float)
