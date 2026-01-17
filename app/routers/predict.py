from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.services.predictor import predict_at_risk, predict_final

router = APIRouter()

class FeatureVector(BaseModel):
    features: List[float]

@router.post("/at-risk")
def at_risk(input: FeatureVector):
    return predict_at_risk(input.features)

@router.post("/final")
def final_prediction(input: FeatureVector):
    return predict_final(input.features)
