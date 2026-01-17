from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.services.explainer import explain_risk_shap, explain_final_shap

router = APIRouter()

class FeatureVector(BaseModel):
    features: List[float]

@router.post("/shap/at-risk")
def shap_at_risk(input: FeatureVector):
    return explain_risk_shap(input.features)

@router.post("/shap/final")
def shap_final(input: FeatureVector):
    return explain_final_shap(input.features)
