import json
import os
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

DATA_PATH = "app/data/students_registry.json"

# ------------------------------------------------------
# Load registry once (safe for demo / thesis)
# ------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise RuntimeError("Student registry not found. Run build_student_registry.py")

with open(DATA_PATH, "r") as f:
    STUDENTS = json.load(f)

# ------------------------------------------------------
# GET /students
# ------------------------------------------------------
@router.get("/")
def list_students():
    return [
        {
            "student_id": sid
        }
        for sid in STUDENTS.keys()
    ]

# ------------------------------------------------------
# GET /students/{student_id}
# ------------------------------------------------------
@router.get("/{student_id}")
def get_student(student_id: str):
    if student_id not in STUDENTS:
        raise HTTPException(status_code=404, detail="Student not found")

    return {
        "student_id": student_id
    }

# ------------------------------------------------------
# GET /students/{student_id}/features
# ------------------------------------------------------
@router.get("/{student_id}/features")
def get_student_features(
    student_id: str,
    feature_type: str = Query(..., enum=["early", "final"])
):
    if student_id not in STUDENTS:
        raise HTTPException(status_code=404, detail="Student not found")

    key = "early_features" if feature_type == "early" else "final_features"

    return {
        "student_id": student_id,
        "feature_type": feature_type,
        "features": STUDENTS[student_id][key]
    }
