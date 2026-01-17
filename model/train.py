import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib
import optuna

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

from app.utils.preprocess import load_oulad, build_full_features

os.makedirs("model", exist_ok=True)
os.makedirs("outputs/summaries", exist_ok=True)


# ================================================================
# Best Threshold Finder
# ================================================================
def find_best_threshold(y_true, proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1, best_t = -1, 0.5

    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return best_t, best_f1


# ================================================================
# Optuna Optimization Objective
# ================================================================
def optimize_xgb(trial, X_train, y_train, X_valid, y_valid, scale_pos_weight):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.3),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.5),
    }

    model = XGBClassifier(
        **params,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    f1 = f1_score(y_valid, preds)
    return f1


# ================================================================
# Train a Single XGBoost Model (Early or Final)
# ================================================================
def train_single_model(X, y, model_name, n_trials=40):

    print(f"\n🚀 Training {model_name} model")

    # -----------------------------------------
    # Split
    # -----------------------------------------
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    spw = neg / pos if pos > 0 else 1

    # -----------------------------------------
    # Optuna Tuning
    # -----------------------------------------
    print("🔍 Running Optuna hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optimize_xgb(
            trial, X_train, y_train, X_valid, y_valid, spw
        ),
        n_trials=n_trials,
        show_progress_bar=False
    )

    best_params = study.best_params
    print("✨ Best Hyperparameters:", best_params)

    # -----------------------------------------
    # Final Model Training
    # -----------------------------------------
    model = XGBClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    # -----------------------------------------
    # Metrics
    # -----------------------------------------
    preds = model.predict(X_valid)
    proba = model.predict_proba(X_valid)[:, 1]

    acc = accuracy_score(y_valid, preds)
    f1 = f1_score(y_valid, preds)
    prec = precision_score(y_valid, preds)
    rec = recall_score(y_valid, preds)
    auc = roc_auc_score(y_valid, proba)

    best_t, best_f1 = find_best_threshold(y_valid.values, proba)

    print("\n📊 Validation Metrics")
    print(f"Accuracy@0.5: {acc:.4f}")
    print(f"F1@0.5:       {f1:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"AUC:          {auc:.4f}")
    print(f"Best Thresh:  {best_t:.2f}")
    print(f"F1@Best:      {best_f1:.4f}")

    # -----------------------------------------
    # Save Model + SHAP Copy
    # -----------------------------------------
    joblib.dump(model, f"model/{model_name}.pkl")
    joblib.dump(model, f"model/{model_name}_xgb_shap.pkl")

    return {
        "Accuracy@0.5": acc,
        "F1@0.5": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC_AUC": auc,
        "BestThreshold": best_t,
        "F1@Best": best_f1
    }


# ================================================================
# MAIN PIPELINE
# ================================================================
def train_models():

    print("\n📌 Loading OULAD dataset...")
    student_info, reg, assess, vle, vle_meta, assess_meta = load_oulad()

    # -----------------------------------------
    # EARLY MODEL
    # -----------------------------------------
    print("\n📌 Building EARLY features...")
    X_early, y_early = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta,
        early_only=True
    )
    print("Early feature shape:", X_early.shape)

    early_results = train_single_model(
        X_early, y_early, "at_risk_model", n_trials=40
    )

    # -----------------------------------------
    # FINAL MODEL
    # -----------------------------------------
    print("\n📌 Building FINAL features...")
    X_final, y_final = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta,
        early_only=False
    )
    print("Final feature shape:", X_final.shape)

    final_results = train_single_model(
        X_final, y_final, "final_model", n_trials=40
    )

    # -----------------------------------------
    # SAVE METRICS
    # -----------------------------------------
    summary = pd.DataFrame({
        "Model": ["Early Risk", "Final Outcome"],
        "Accuracy@0.5": [early_results["Accuracy@0.5"], final_results["Accuracy@0.5"]],
        "F1@0.5": [early_results["F1@0.5"], final_results["F1@0.5"]],
        "Precision": [early_results["Precision"], final_results["Precision"]],
        "Recall": [early_results["Recall"], final_results["Recall"]],
        "ROC_AUC": [early_results["ROC_AUC"], final_results["ROC_AUC"]],
        "BestThreshold": [early_results["BestThreshold"], final_results["BestThreshold"]],
        "F1@Best": [early_results["F1@Best"], final_results["F1@Best"]],
    })

    summary.to_csv("outputs/summaries/train_metrics.csv", index=False)

    print("\n🎉 FINAL RESULTS")
    print(summary)
    print("\nModels saved successfully.")


if __name__ == "__main__":
    train_models()
