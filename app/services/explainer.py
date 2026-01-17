import shap
import joblib
import numpy as np

# ======================================================
# LOAD MODELS (TREE-SHAP COMPATIBLE)
# ======================================================
risk_model = joblib.load("model/at_risk_model_xgb_shap.pkl")
final_model = joblib.load("model/final_model_xgb_shap.pkl")

# ======================================================
# TREE SHAP EXPLAINERS (EXACT)
# ======================================================
risk_explainer = shap.TreeExplainer(risk_model)
final_explainer = shap.TreeExplainer(final_model)


# ======================================================
# SHAP EXPLANATIONS — AT RISK
# ======================================================
def explain_risk_shap(features: list):
    X = np.array([features])
    shap_values = risk_explainer.shap_values(X)

    return {
        "shap_values": shap_values.tolist(),
        "base_value": float(risk_explainer.expected_value)
    }


# ======================================================
# SHAP EXPLANATIONS — FINAL OUTCOME
# ======================================================
def explain_final_shap(features: list):
    X = np.array([features])
    shap_values = final_explainer.shap_values(X)

    return {
        "shap_values": shap_values.tolist(),
        "base_value": float(final_explainer.expected_value)
    }
