import shap
import joblib
import numpy as np

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

# (You can later add FINAL_FEATURE_MEANINGS in same way)

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

    # Sort by absolute impact
    explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return explanations[:top_k]

# ======================================================
# SHAP EXPLANATIONS — AT RISK
# ======================================================
def explain_risk_shap(features: list):
    X = np.array([features])

    shap_values = risk_explainer.shap_values(X)[0]
    base_value = float(risk_explainer.expected_value)

    top_factors = build_explanations(
        shap_values=shap_values,
        feature_names=risk_feature_names,
        feature_meanings=RISK_FEATURE_MEANINGS
    )

    return {
        "base_value": base_value,
        "top_factors": top_factors,
        "raw_shap_values": shap_values.tolist()
    }

# ======================================================
# SHAP EXPLANATIONS — FINAL OUTCOME
# ======================================================
def explain_final_shap(features: list):
    X = np.array([features])

    shap_values = final_explainer.shap_values(X)[0]
    base_value = float(final_explainer.expected_value)

    top_factors = build_explanations(
        shap_values=shap_values,
        feature_names=final_feature_names,
        feature_meanings={},  # add later if needed
        top_k=7
    )

    return {
        "base_value": base_value,
        "top_factors": top_factors,
        "raw_shap_values": shap_values.tolist()
    }
