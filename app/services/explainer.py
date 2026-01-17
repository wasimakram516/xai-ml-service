import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# LOAD MODELS (TREE-SHAP COMPATIBLE)
# ======================================================
risk_model = joblib.load("model/at_risk_model_xgb_shap.pkl")
final_model = joblib.load("model/final_model_xgb_shap.pkl")

# ======================================================
# LOAD FEATURE NAMES
# ======================================================
risk_feature_names = joblib.load("model/at_risk_features.pkl")
final_feature_names = joblib.load("model/final_features.pkl")

# ======================================================
# TREE SHAP EXPLAINERS (EXACT)
# ======================================================
risk_explainer = shap.TreeExplainer(risk_model)
final_explainer = shap.TreeExplainer(final_model)

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
# UTILITY: BUILD HUMAN EXPLANATIONS
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
# SHAP EXPLANATIONS — AT RISK (API)
# ======================================================
def explain_risk_shap(features: list):
    X = np.array([features])
    shap_values = risk_explainer.shap_values(X)[0]
    base_value = float(risk_explainer.expected_value)

    return {
        "base_value": base_value,
        "top_factors": build_explanations(
            shap_values,
            risk_feature_names,
            RISK_FEATURE_MEANINGS
        ),
        "raw_shap_values": shap_values.tolist()
    }

# ======================================================
# SHAP EXPLANATIONS — FINAL OUTCOME (API)
# ======================================================
def explain_final_shap(features: list):
    X = np.array([features])
    shap_values = final_explainer.shap_values(X)[0]
    base_value = float(final_explainer.expected_value)

    return {
        "base_value": base_value,
        "top_factors": build_explanations(
            shap_values,
            final_feature_names,
            {},
            top_k=7
        ),
        "raw_shap_values": shap_values.tolist()
    }

# =====================================================================
# OFFLINE GLOBAL SHAP PLOTS (NOT USED BY API)
# =====================================================================
def generate_global_shap_plots(X_risk, X_final):
    """
    Offline-only function.
    Call this from training or a notebook.
    """
    os.makedirs("outputs/figures", exist_ok=True)

    # -----------------------
    # EARLY RISK MODEL
    # -----------------------
    shap_vals_risk = risk_explainer.shap_values(X_risk)

    shap.summary_plot(
        shap_vals_risk,
        X_risk,
        feature_names=risk_feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/early_shap_summary.png")
    plt.close()

    shap.summary_plot(
        shap_vals_risk,
        X_risk,
        feature_names=risk_feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/early_shap_bar.png")
    plt.close()

    # -----------------------
    # FINAL OUTCOME MODEL
    # -----------------------
    shap_vals_final = final_explainer.shap_values(X_final)

    shap.summary_plot(
        shap_vals_final,
        X_final,
        feature_names=final_feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/final_shap_summary.png")
    plt.close()

    shap.summary_plot(
        shap_vals_final,
        X_final,
        feature_names=final_feature_names,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/final_shap_bar.png")
    plt.close()
