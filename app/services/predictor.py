import joblib
import numpy as np

# ======================================================
# LOAD MODELS
# ======================================================
risk_model = joblib.load("model/at_risk_model.pkl")
final_model = joblib.load("model/final_model.pkl")

# ======================================================
# THRESHOLDS (LOCKED FROM METRICS)
# ======================================================
RISK_THRESHOLD = 0.41
FINAL_THRESHOLD = 0.51

# ======================================================
# FEATURE COUNTS (AUTHORITATIVE)
# ======================================================
EXPECTED_RISK_FEATURES = risk_model.get_booster().num_features()
EXPECTED_FINAL_FEATURES = final_model.get_booster().num_features()


# ======================================================
# PREDICT AT-RISK
# ======================================================
def predict_at_risk(features: list):
    if len(features) != EXPECTED_RISK_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_RISK_FEATURES} features, got {len(features)}"
        )

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
    if len(features) != EXPECTED_FINAL_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FINAL_FEATURES} features, got {len(features)}"
        )

    X = np.array([features])
    prob = float(final_model.predict_proba(X)[0][1])
    pred = int(prob >= FINAL_THRESHOLD)

    return {
        "final_prediction": pred,
        "probability": prob,
        "threshold": FINAL_THRESHOLD
    }
