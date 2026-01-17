import joblib
import numpy as np

# ======================================================
# LOAD TRAINED XGBOOST MODELS
# ======================================================
risk_model = joblib.load("model/at_risk_model.pkl")
final_model = joblib.load("model/final_model.pkl")

# ======================================================
# OPTIMIZED THRESHOLDS (FROM TRAINING)
# ======================================================
RISK_THRESHOLD = 0.40
FINAL_THRESHOLD = 0.54


# ======================================================
# PREDICT AT-RISK
# ======================================================
def predict_at_risk(features: list):
    X = np.array([features])
    prob = float(risk_model.predict_proba(X)[0][1])
    pred = int(prob >= RISK_THRESHOLD)

    return {
        "at_risk": pred,
        "probability": prob,
        "threshold": RISK_THRESHOLD
    }


# ======================================================
# PREDICT FINAL OUTCOME
# ======================================================
def predict_final(features: list):
    X = np.array([features])
    prob = float(final_model.predict_proba(X)[0][1])
    pred = int(prob >= FINAL_THRESHOLD)

    return {
        "final_prediction": pred,
        "probability": prob,
        "threshold": FINAL_THRESHOLD
    }
