import json
import os
from fastapi import APIRouter, HTTPException, Query, Depends

from app.dependencies.auth import get_current_teacher
from app.services.predictor import predict_at_risk, predict_final
from app.services.explainer import explain_risk_shap, explain_final_shap

router = APIRouter(
    prefix="/students",
    tags=["Students"],
    dependencies=[Depends(get_current_teacher)],
)

DATA_PATH = "app/data/students_registry.json"

# ------------------------------------------------------
# Load registry once
# ------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise RuntimeError("Student registry not found")

with open(DATA_PATH, "r") as f:
    STUDENTS = json.load(f)

# ------------------------------------------------------
# GET /students
# ------------------------------------------------------
@router.get("/")
def list_students():
    return [{"student_id": sid} for sid in STUDENTS.keys()]

# ------------------------------------------------------
# GET /students/{student_id}
# Full student profile (NO features, NO predictions)
# ------------------------------------------------------
@router.get("/{student_id}")
def get_student(student_id: str):
    if student_id not in STUDENTS:
        raise HTTPException(status_code=404, detail="Student not found")

    student = STUDENTS[student_id]

    return {
        "student_id": student["student_id"],
        "label": student.get("label"),
        "oulad_id": student.get("oulad_id"),
        "demographics": student.get("demographics", {}),
        "stats": student.get("stats", {}),
    }

# ------------------------------------------------------
# GET /students/{student_id}/early
# ------------------------------------------------------
@router.get("/{student_id}/early")
def early_at_risk(student_id: str):
    if student_id not in STUDENTS:
        raise HTTPException(status_code=404, detail="Student not found")

    features = STUDENTS[student_id]["early_features"]

    return {
        "student_id": student_id,
        "stage": "early",
        "prediction": predict_at_risk(features),
        "explanation": explain_risk_shap(features),
    }

# ------------------------------------------------------
# GET /students/{student_id}/final
# ------------------------------------------------------
@router.get("/{student_id}/final")
def final_outcome(student_id: str):
    if student_id not in STUDENTS:
        raise HTTPException(status_code=404, detail="Student not found")

    features = STUDENTS[student_id]["final_features"]

    return {
        "student_id": student_id,
        "stage": "final",
        "prediction": predict_final(features),
        "explanation": explain_final_shap(features),
    }

# ------------------------------------------------------
# DEBUG ONLY — features
# ------------------------------------------------------
@router.get("/{student_id}/features")
def get_student_features(
    student_id: str,
    feature_type: str = Query(..., enum=["early", "final"]),
):
    if student_id not in STUDENTS:
        raise HTTPException(status_code=404, detail="Student not found")

    key = "early_features" if feature_type == "early" else "final_features"

    return {
        "student_id": student_id,
        "feature_type": feature_type,
        "features": STUDENTS[student_id][key],
    }
