from fastapi.testclient import TestClient

from wine_api.api.main import app
from wine_api.predictor import Prediction

VALID = {
    "fixed_acidity": 7.4, "volatile_acidity": 0.7, "citric_acid": 0.0,
    "residual_sugar": 1.9, "chlorides": 0.076, "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4,
}


class FakePredictor:
    version = "test"
    features = ["alcohol"]
    metrics = {"mae": 0.5, "n_test": 10}
    feature_ranges = {"alcohol": [8.4, 14.9]}
    feature_importances = {"alcohol": 1.0}

    def predict_many(self, rows: list[dict]) -> list[Prediction]:
        return [Prediction(5.456, 5.1, 5.9, ["alcohol=99 is outside the training range [8.4, 14.9]"]) for _ in rows]


def make_client() -> TestClient:
    app.state.predictor = FakePredictor()  # no context manager, so real model is not loaded
    return TestClient(app)


def test_health():
    assert make_client().get("/health").json() == {"status": "ok", "model_version": "test"}


def test_model_not_loaded_returns_503():
    app.state.predictor = None
    assert TestClient(app).get("/health").status_code == 503


def test_predict_ok():
    r = make_client().post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.headers["x-request-id"]
    assert r.json() == {
        "quality": 5.46,
        "rounded_quality": 5,
        "category": "average",
        "low": 5.1,
        "high": 5.9,
        "warnings": ["alcohol=99 is outside the training range [8.4, 14.9]"],
        "model_version": "test",
    }


def test_predict_rejects_bad_input():
    bad = {**VALID, "alcohol": 99}
    assert make_client().post("/predict", json=bad).status_code == 422


def test_predict_rejects_missing_and_unknown_fields():
    client = make_client()
    missing = {k: v for k, v in VALID.items() if k != "pH"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**VALID, "color": "red"}).status_code == 422


def test_predict_batch():
    r = make_client().post("/predict/batch", json={"samples": [VALID, VALID, VALID]})
    body = r.json()
    assert r.status_code == 200
    assert len(body["predictions"]) == 3
    assert body["model_version"] == "test"


def test_predict_batch_limits():
    client = make_client()
    assert client.post("/predict/batch", json={"samples": []}).status_code == 422
    assert client.post("/predict/batch", json={"samples": [VALID] * 101}).status_code == 422


def test_model_info():
    body = make_client().get("/model-info").json()
    assert body == {
        "model_version": "test",
        "features": ["alcohol"],
        "metrics": {"mae": 0.5, "n_test": 10},
        "feature_ranges": {"alcohol": [8.4, 14.9]},
        "feature_importances": {"alcohol": 1.0},
    }
