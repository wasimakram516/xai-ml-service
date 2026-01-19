import sys
import os
import json
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from app.utils.preprocess import load_oulad, build_full_features

# ======================================================
# CONFIG
# ======================================================
OUTPUT_PATH = "app/data/students_registry.json"
MAX_STUDENTS = 1000
RANDOM_SEED = 42

# ======================================================
# HELPERS
# ======================================================
def to_python(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v


def sanitize_list(values):
    return [to_python(v) for v in values]


# ======================================================
# MAIN
# ======================================================
def build_registry():
    print("Loading OULAD data...")
    student_info, reg, assess, vle, vle_meta, assess_meta, courses = load_oulad()

    print("Building EARLY features...")
    X_early, _ = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta, courses,
        early_only=True
    )

    print("Building FINAL features...")
    X_final, _ = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta, courses,
        early_only=False
    )

    assert len(X_early) == len(X_final)

    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(
        len(X_early),
        size=min(MAX_STUDENTS, len(X_early)),
        replace=False
    )

    registry = {}

    for i, idx in enumerate(indices):
        row = student_info.iloc[idx]
        student_key = f"STU_{i:04d}"

        registry[student_key] = {
            "student_id": student_key,
            "oulad_id": to_python(row.get("id_student")),
            "label": f"Student {i + 1}",

            "demographics": {
                "gender": to_python(row.get("gender")),
                "age_band": to_python(row.get("age_band")),
                "highest_education": to_python(row.get("highest_education")),
                "region": to_python(row.get("region")),
                "disability": to_python(row.get("disability")),
            },

            "stats": {
                "num_assessments": to_python(row.get("num_of_prev_attempts", 0)),
                "avg_score": to_python(row.get("studied_credits")),
                "total_clicks": int(
                    sum(v for v in X_early.iloc[idx].tolist() if isinstance(v, (int, float)))
                ),
                "engagement_variance": to_python(
                    np.var(X_early.iloc[idx].tolist())
                ),
            },

            "early_features": sanitize_list(
                X_early.iloc[idx].tolist()
            ),
            "final_features": sanitize_list(
                X_final.iloc[idx].tolist()
            ),
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"Student registry created: {OUTPUT_PATH}")
    print(f"Students included: {len(registry)}")

if __name__ == "__main__":
    build_registry()
