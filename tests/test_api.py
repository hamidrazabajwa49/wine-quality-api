from fastapi.testclient import TestClient

from wine_api.api.main import app

VALID = {
    "fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0.0,
    "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4,
}


class FakePredictor:
    version = "test"
    features = ["alcohol"]
    metrics = {"mae": 0.5}

    def predict(self, values: dict) -> float:
        return 5.456


def make_client() -> TestClient:
    app.state.predictor = FakePredictor()  # no context manager, so real model is not loaded
    return TestClient(app)


def test_health():
    assert make_client().get("/health").json() == {"status": "ok", "model_version": "test"}


def test_predict_ok():
    r = make_client().post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.json() == {"quality": 5.46, "model_version": "test"}


def test_predict_rejects_bad_input():
    bad = {**VALID, "alcohol": 99}
    assert make_client().post("/predict", json=bad).status_code == 422


def test_model_info():
    body = make_client().get("/model-info").json()
    assert body == {"model_version": "test", "features": ["alcohol"], "metrics": {"mae": 0.5}}
