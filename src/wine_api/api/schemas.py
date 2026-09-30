"""
Pydantic Model
"""

from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    fixed_acidity: float = Field(gt=0, examples=[7.4])
    volatile_acidity: float = Field(ge=0, examples=[0.7])
    citric_acid: float = Field(ge=0, examples=[0.0])
    residual_sugar: float = Field(ge=0, examples=[1.9])
    chlorides: float = Field(ge=0, examples=[0.076])
    free_sulfur_dioxide: float = Field(ge=0, examples=[11])
    total_sulfur_dioxide: float = Field(ge=0, examples=[34])
    density: float = Field(gt=0, examples=[0.9978])
    pH: float = Field(gt=0, le=14, examples=[3.51])
    sulphates: float = Field(ge=0, examples=[0.56])
    alcohol: float = Field(ge=0, le=20, examples=[9.4])


class PredictResponse(BaseModel):
    quality: float
    model_version: str
