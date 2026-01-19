import sys
import os

# ------------------------------------------------
# Make app/ importable
# ------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.preprocess import load_oulad, build_full_features
from app.services.explainer import generate_global_shap_plots


def main():
    print("Loading OULAD dataset...")
    student_info, reg, assess, vle, vle_meta, assess_meta, courses = load_oulad()

    print("Building EARLY features...")
    X_early, _ = build_full_features(
        student_info,
        reg,
        assess,
        vle,
        vle_meta,
        assess_meta,
        courses,
        early_only=True
    )

    print("Building FINAL features...")
    X_final, _ = build_full_features(
        student_info,
        reg,
        assess,
        vle,
        vle_meta,
        assess_meta,
        courses,
        early_only=False
    )

    print("Generating SHAP figures...")
    generate_global_shap_plots(
        X_early.values,
        X_final.values
    )

    print("SHAP figures successfully generated.")


if __name__ == "__main__":
    main()
