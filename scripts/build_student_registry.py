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
MAX_STUDENTS = 200   # Maximum number of students to include
RANDOM_SEED = 42     # For reproducibility

# ======================================================
# MAIN
# ======================================================
def build_registry():
    print("Loading OULAD data...")
    student_info, reg, assess, vle, vle_meta, assess_meta = load_oulad()

    print("Building EARLY features...")
    X_early, y_early = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta,
        early_only=True
    )

    print("Building FINAL features...")
    X_final, y_final = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta,
        early_only=False
    )

    assert len(X_early) == len(X_final), "Feature row mismatch"

    # --------------------------------------------
    # Select a reproducible subset of real students
    # --------------------------------------------
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(
        len(X_early),
        size=min(MAX_STUDENTS, len(X_early)),
        replace=False
    )

    registry = {}

    for i, idx in enumerate(indices):
        student_key = f"STU_{i:04d}"

        registry[student_key] = {
            "early_features": X_early.iloc[idx].tolist()
            if hasattr(X_early, "iloc")
            else X_early[idx].tolist(),

            "final_features": X_final.iloc[idx].tolist()
            if hasattr(X_final, "iloc")
            else X_final[idx].tolist()
        }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"Student registry created: {OUTPUT_PATH}")
    print(f"Students included: {len(registry)}")

if __name__ == "__main__":
    build_registry()
