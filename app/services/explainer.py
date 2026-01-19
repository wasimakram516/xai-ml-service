import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# LOAD FEATURE METADATA (SAFE & EXPLICIT)
# ======================================================
def load_feature_columns(path, model_name):
    meta = joblib.load(path)

    if not isinstance(meta, dict):
        raise TypeError(
            f"[SHAP ERROR] {model_name} feature file must be a dict, got {type(meta)}"
        )

    if "columns" not in meta:
        raise KeyError(
            f"[SHAP ERROR] {model_name} feature file missing 'columns'"
        )

    columns = list(meta["columns"])

    if len(columns) != meta.get("n_features", len(columns)):
        raise ValueError(
            f"[SHAP ERROR] {model_name} feature count mismatch "
            f"(columns={len(columns)}, n_features={meta.get('n_features')})"
        )

    return columns

# ======================================================
# LOAD MODELS (TREE-SHAP COMPATIBLE)
# ======================================================
risk_model = joblib.load("model/at_risk_model_xgb_shap.pkl")
final_model = joblib.load("model/final_model_xgb_shap.pkl")

# ======================================================
# LOAD FEATURE NAMES (AUTHORITATIVE)
# ======================================================
risk_feature_names = load_feature_columns(
    "model/at_risk_model_features.pkl",
    "Early Risk"
)

final_feature_names = load_feature_columns(
    "model/final_model_features.pkl",
    "Final Outcome"
)

# ======================================================
# TREE SHAP EXPLAINERS (EXACT, STABLE)
# ======================================================
risk_explainer = shap.TreeExplainer(
    risk_model,
    feature_perturbation="tree_path_dependent"
)

final_explainer = shap.TreeExplainer(
    final_model,
    feature_perturbation="tree_path_dependent"
)

# ======================================================
# FEATURE MEANINGS (HUMAN READABLE)
# ======================================================
RISK_FEATURE_MEANINGS = {
    "mean_clicks": "Average engagement with learning materials",
    "std_clicks": "Consistency of engagement over time",
    "max_gap": "Longest period of inactivity",
    "late_ratio": "Frequency of late assessment submissions",
    "avg_score": "Average assessment performance",
    "min_score": "Lowest assessment score",
    "score_std": "Variability in assessment performance",
    "activity_weeks": "Number of weeks with learning activity",
    "no_activity_weeks": "Weeks without any engagement",
    "click_trend": "Trend in engagement over time",
    "assessment_trend": "Trend in assessment performance",
    "submission_count": "Total number of submissions",
    "pass_ratio": "Ratio of passed assessments",
    "fail_ratio": "Ratio of failed assessments",
    "late_submissions": "Count of late submissions",
    "early_activity": "Early engagement in the course",
    "final_activity": "Engagement near the end of the course"
}

# ======================================================
# INTERNAL SAFETY CHECK (DO NOT REMOVE)
# ======================================================
def _validate_alignment(X, feature_names, model_name):
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"[SHAP ERROR] Feature mismatch in {model_name}: "
            f"X has {X.shape[1]} columns, "
            f"but feature list has {len(feature_names)} entries."
        )

# ======================================================
# BUILD HUMAN-READABLE EXPLANATIONS
# ======================================================
def build_explanations(shap_values, feature_names, feature_meanings, top_k=5):
    explanations = []

    for name, value in zip(feature_names, shap_values):
        explanations.append({
            "feature": name,
            "impact": float(value),
            "meaning": feature_meanings.get(
                name,
                "Behavioural feature influencing prediction"
            )
        })

    explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return explanations[:top_k]

# ======================================================
# SHAP — AT RISK (API)
# ======================================================
def explain_risk_shap(features: list):
    X = np.array([features])

    shap_values = risk_explainer.shap_values(X)
    shap_row = shap_values[0]

    base_value = float(risk_explainer.expected_value)

    return {
        "base_value": base_value,
        "top_factors": build_explanations(
            shap_row,
            risk_feature_names,
            RISK_FEATURE_MEANINGS
        ),
        "raw_shap_values": shap_row.tolist()
    }

# ======================================================
# SHAP — FINAL OUTCOME (API)
# ======================================================
def explain_final_shap(features: list):
    X = np.array([features])

    shap_values = final_explainer.shap_values(X)
    shap_row = shap_values[0]

    base_value = float(final_explainer.expected_value)

    return {
        "base_value": base_value,
        "top_factors": build_explanations(
            shap_row,
            final_feature_names,
            {},
            top_k=7
        ),
        "raw_shap_values": shap_row.tolist()
    }

# =====================================================================
# OFFLINE GLOBAL SHAP FIGURES (FOR PAPER / THESIS)
# =====================================================================
def generate_global_shap_plots(X_risk, X_final):
    """
    Offline-only function.
    Use after training with the same feature matrices.
    """
    os.makedirs("outputs/figures", exist_ok=True)

    # -----------------------
    # EARLY RISK MODEL
    # -----------------------
    _validate_alignment(X_risk, risk_feature_names, "Early Risk")

    shap_vals_risk = risk_explainer.shap_values(X_risk)

    shap.summary_plot(
        shap_vals_risk,
        X_risk,
        feature_names=risk_feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/early_shap_summary.png", dpi=300)
    plt.close()

    shap.summary_plot(
        shap_vals_risk,
        X_risk,
        feature_names=risk_feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/early_shap_bar.png", dpi=300)
    plt.close()

    # -----------------------
    # FINAL OUTCOME MODEL
    # -----------------------
    _validate_alignment(X_final, final_feature_names, "Final Outcome")

    shap_vals_final = final_explainer.shap_values(X_final)

    shap.summary_plot(
        shap_vals_final,
        X_final,
        feature_names=final_feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/final_shap_summary.png", dpi=300)
    plt.close()

    shap.summary_plot(
        shap_vals_final,
        X_final,
        feature_names=final_feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/final_shap_bar.png", dpi=300)
    plt.close()
