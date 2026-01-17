import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

from app.utils.preprocess import load_oulad, build_full_features

os.makedirs("model", exist_ok=True)
os.makedirs("outputs/summaries", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

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
# Plotting Utilities (OFFLINE — ML LAYER ONLY)
# ================================================================
def plot_roc(y_true, proba, name):
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {name}")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{name}_roc.png")
    plt.close()


def plot_pr(y_true, proba, name):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall — {name}")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{name}_pr.png")
    plt.close()


def plot_confusion(y_true, proba, threshold, name):
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{name}_confusion.png")
    plt.close()


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

    print(f"\nTraining {model_name} model")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    spw = neg / pos if pos > 0 else 1

    print("Running Optuna hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optimize_xgb(
            trial, X_train, y_train, X_valid, y_valid, spw
        ),
        n_trials=n_trials,
        show_progress_bar=False
    )

    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

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

    preds = model.predict(X_valid)
    proba = model.predict_proba(X_valid)[:, 1]

    acc = accuracy_score(y_valid, preds)
    f1 = f1_score(y_valid, preds)
    prec = precision_score(y_valid, preds)
    rec = recall_score(y_valid, preds)
    auc = roc_auc_score(y_valid, proba)

    best_t, best_f1 = find_best_threshold(y_valid.values, proba)

    print("\nValidation Metrics")
    print(f"Accuracy@0.5: {acc:.4f}")
    print(f"F1@0.5:       {f1:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"AUC:          {auc:.4f}")
    print(f"Best Thresh:  {best_t:.2f}")
    print(f"F1@Best:      {best_f1:.4f}")

    # ------------------------------------------------
    # OFFLINE ML-LAYER EVALUATION PLOTS
    # ------------------------------------------------
    plot_roc(y_valid, proba, model_name)
    plot_pr(y_valid, proba, model_name)
    plot_confusion(y_valid, proba, best_t, model_name)

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

    print("\nLoading OULAD dataset...")
    student_info, reg, assess, vle, vle_meta, assess_meta = load_oulad()

    print("\nBuilding EARLY features...")
    X_early, y_early = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta,
        early_only=True
    )

    early_results = train_single_model(
        X_early, y_early, "at_risk_model", n_trials=40
    )

    print("\nBuilding FINAL features...")
    X_final, y_final = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta,
        early_only=False
    )

    final_results = train_single_model(
        X_final, y_final, "final_model", n_trials=40
    )

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

    print("\nFINAL RESULTS")
    print(summary)
    print("\nModels and figures saved successfully.")


if __name__ == "__main__":
    train_models()
