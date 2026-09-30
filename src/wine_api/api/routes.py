"""
Endpoints of API
"""
from fastapi import APIRouter, Request
from .schemas import PredictRequest, PredictResponse

router = APIRouter()

@router.get('/health')
def health(request: Request) -> dict:
    return {"status":"OK", "model_version": request.app.state.predictor.version}

@router.get("/model-info")
def model_info(request: Request) -> dict:
    predictor = request.app.state.predictor
    return {"model_version": predictor.version, "features":predictor.features, "metrics": predictor.metrics}

@router.post("/predict",response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    predictor = request.app.state.predictor
    quality = predictor.predict(body.model_dump())
    return PredictResponse(quality=round(quality,3), model_version=predictor.version)
