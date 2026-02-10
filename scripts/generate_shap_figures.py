import sys
import os
import joblib

# ------------------------------------------------
# Make app/ importable
# ------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.preprocess import load_oulad, build_full_features
from app.services.explainer import generate_global_shap_plots


def main():
    model_dir = os.environ.get("MODEL_DIR", "model")
    early_cache = os.path.join(model_dir, "at_risk_model_features_full.pkl")
    final_cache = os.path.join(model_dir, "final_model_features_full.pkl")

    if os.path.exists(early_cache) and os.path.exists(final_cache):
        print("Loading cached feature matrices from model/ ...")
        X_early = joblib.load(early_cache)
        X_final = joblib.load(final_cache)
    else:
        print("Cached features not found. Rebuilding from OULAD...")
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
