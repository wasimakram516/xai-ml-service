import json
import os
from fastapi import APIRouter, HTTPException, Query, Depends

from app.dependencies.auth import get_current_teacher
from app.services.predictor import predict_at_risk, predict_final
from app.services.explainer import explain_risk_shap, explain_final_shap, get_global_shap_summary

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
# GET /students/global/early
# ------------------------------------------------------
@router.get("/global/early")
def global_early_explanations(top_k: int = Query(15, ge=5, le=30)):
    return get_global_shap_summary("early", top_k=top_k)


# ------------------------------------------------------
# GET /students/global/final
# ------------------------------------------------------
@router.get("/global/final")
def global_final_explanations(top_k: int = Query(15, ge=5, le=30)):
    return get_global_shap_summary("final", top_k=top_k)

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
# GET /students/{student_id}/insights
# Combined payload for UI: prediction + local + global context
# ------------------------------------------------------
@router.get(
    "/{student_id}/insights",
    summary="Student Insights (Recommended)",
    description="Recommended endpoint for UI integration. Returns prediction, "
                "local SHAP explanation, and global cohort context in one payload.",
)
def student_insights(
    student_id: str,
    stage: str = Query(..., enum=["early", "final"]),
    top_k: int = Query(10, ge=5, le=30),
):
    if student_id not in STUDENTS:
        raise HTTPException(status_code=404, detail="Student not found")

    if stage == "early":
        features = STUDENTS[student_id]["early_features"]
        prediction = predict_at_risk(features)
        local_explanation = explain_risk_shap(features)
    else:
        features = STUDENTS[student_id]["final_features"]
        prediction = predict_final(features)
        local_explanation = explain_final_shap(features)

    global_explanation = get_global_shap_summary(stage, top_k=top_k)

    return {
        "student_id": student_id,
        "stage": stage,
        "prediction": prediction,
        "local_explanation": local_explanation,
        "global_context": global_explanation,
    }
