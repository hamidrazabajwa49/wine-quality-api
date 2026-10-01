"""
Pydantic Model
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..predictor import Prediction

MAX_BATCH = 100


def quality_category(score: float) -> Literal["low", "average", "high"]:
    rounded = int(score + 0.5)
    if rounded <= 4:
        return "low"
    if rounded >= 7:
        return "high"
    return "average"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixed_acidity: float = Field(gt=0, le=30, description="Tartaric acid, g/dm3", examples=[7.4])
    volatile_acidity: float = Field(ge=0, le=5, description="Acetic acid, g/dm3", examples=[0.7])
    citric_acid: float = Field(ge=0, le=2, description="Citric acid, g/dm3", examples=[0.0])
    residual_sugar: float = Field(ge=0, le=100, description="Sugar left after fermentation, g/dm3", examples=[1.9])
    chlorides: float = Field(ge=0, le=1, description="Sodium chloride, g/dm3", examples=[0.076])
    free_sulfur_dioxide: float = Field(ge=0, le=300, description="Free SO2, mg/dm3", examples=[11])
    total_sulfur_dioxide: float = Field(ge=0, le=600, description="Total SO2, mg/dm3", examples=[34])
    density: float = Field(gt=0.9, le=1.1, description="Density, g/cm3", examples=[0.9978])
    pH: float = Field(ge=1, le=14, description="pH", examples=[3.51])
    sulphates: float = Field(ge=0, le=5, description="Potassium sulphate, g/dm3", examples=[0.56])
    alcohol: float = Field(ge=0, le=20, description="Alcohol, % by volume", examples=[9.4])


class PredictionOut(BaseModel):
    quality: float = Field(description="Mean prediction over all trees")
    rounded_quality: int = Field(description="Quality rounded to the nearest whole score")
    category: Literal["low", "average", "high"] = Field(description="low: 4 or less, average: 5-6, high: 7 or more")
    low: float = Field(description="10th percentile of individual tree predictions")
    high: float = Field(description="90th percentile of individual tree predictions")
    warnings: list[str] = Field(description="Inputs that fall outside the training range")

    @classmethod
    def from_prediction(cls, p: Prediction) -> "PredictionOut":
        return cls(
            quality=round(p.quality, 2),
            rounded_quality=int(p.quality + 0.5),
            category=quality_category(p.quality),
            low=round(p.low, 2),
            high=round(p.high, 2),
            warnings=p.warnings,
        )


class PredictResponse(PredictionOut):
    model_version: str


class BatchPredictRequest(BaseModel):
    samples: list[PredictRequest] = Field(min_length=1, max_length=MAX_BATCH)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionOut]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    features: list[str]
    metrics: dict[str, int | float]
    feature_ranges: dict[str, list[float]]
    feature_importances: dict[str, float]
