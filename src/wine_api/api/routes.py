"""
Endpoints of API
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from ..predictor import Predictor
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionOut,
    PredictRequest,
    PredictResponse,
)

router = APIRouter()


def get_predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return predictor


@router.get("/health", response_model=HealthResponse, tags=["system"], summary="Liveness check")
def health(predictor: Predictor = Depends(get_predictor)) -> HealthResponse:
    return HealthResponse(status="ok", model_version=predictor.version)


@router.get("/model-info", response_model=ModelInfoResponse, tags=["system"], summary="Model metadata")
def model_info(predictor: Predictor = Depends(get_predictor)) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=predictor.version,
        features=predictor.features,
        metrics=predictor.metrics,
        feature_ranges=predictor.feature_ranges,
        feature_importances=predictor.feature_importances,
    )


@router.post("/predict", response_model=PredictResponse, tags=["prediction"], summary="Predict one wine")
def predict(body: PredictRequest, predictor: Predictor = Depends(get_predictor)) -> PredictResponse:
    result = predictor.predict_many([body.model_dump()])[0]
    return PredictResponse(**PredictionOut.from_prediction(result).model_dump(), model_version=predictor.version)


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["prediction"],
    summary="Predict up to 100 wines in one call",
)
def predict_batch(body: BatchPredictRequest, predictor: Predictor = Depends(get_predictor)) -> BatchPredictResponse:
    results = predictor.predict_many([s.model_dump() for s in body.samples])
    return BatchPredictResponse(
        predictions=[PredictionOut.from_prediction(r) for r in results],
        model_version=predictor.version,
    )
