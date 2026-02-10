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
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, average_precision_score
)
from sklearn.model_selection import train_test_split

from app.utils.preprocess import load_oulad, build_full_features

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
OUTPUT_SUMMARIES_DIR = "outputs/summaries"
OUTPUT_FIGURES_DIR = "outputs/figures"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_SUMMARIES_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)


def model_path(name: str) -> str:
    return os.path.join(MODEL_DIR, name)


def summary_path(name: str) -> str:
    return os.path.join(OUTPUT_SUMMARIES_DIR, name)


def figure_path(name: str) -> str:
    return os.path.join(OUTPUT_FIGURES_DIR, name)

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
# Plotting Utilities
# ================================================================
def plot_roc(y_true, proba, name):
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.tight_layout()
    plt.savefig(figure_path(f"{name}_roc.png"))
    plt.close()


def plot_pr(y_true, proba, name):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {name}")
    plt.tight_layout()
    plt.savefig(figure_path(f"{name}_pr.png"))
    plt.close()


def plot_confusion(y_true, proba, threshold, name):
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(figure_path(f"{name}_confusion.png"))
    plt.close()

    # Also save normalized confusion matrix for class-wise readability
    cm_norm = confusion_matrix(y_true, preds, normalize="true")
    disp_norm = ConfusionMatrixDisplay(cm_norm)
    disp_norm.plot(cmap="Blues", values_format=".2f")
    plt.title(f"Normalized Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(figure_path(f"{name}_confusion_normalized.png"))
    plt.close()


def plot_threshold_curve(y_true, proba, name):
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        f1_scores.append(f1_score(y_true, preds))

    best_idx = int(np.argmax(f1_scores))
    best_t = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, f1_scores)
    plt.scatter([best_t], [best_f1], c="red", s=40)
    plt.xlabel("Decision Threshold")
    plt.ylabel("F1 Score")
    plt.title(f"Threshold vs F1 - {name}")
    plt.tight_layout()
    plt.savefig(figure_path(f"{name}_threshold_f1.png"))
    plt.close()


def plot_classification_report_visual(y_true, preds, model_name, suffix):
    report = classification_report(y_true, preds, output_dict=True, zero_division=0)
    classes = ["0", "1"]
    metrics = ["precision", "recall", "f1-score"]
    mat = np.array([[report[c][m] for m in metrics] for c in classes], dtype=float)

    plt.figure(figsize=(6, 4))
    im = plt.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(classes)), [f"class {c}" for c in classes])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt_color = "white" if val > 0.6 else "black"
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color)

    plt.title(f"Classification Report ({suffix}) - {model_name}")
    plt.tight_layout()
    plt.savefig(figure_path(f"{model_name}_classification_report_{suffix}.png"))
    plt.close()


def save_classification_reports(y_true, preds_05, preds_best, model_name):
    os.makedirs(OUTPUT_SUMMARIES_DIR, exist_ok=True)

    report_05 = classification_report(y_true, preds_05, output_dict=True, zero_division=0)
    report_best = classification_report(y_true, preds_best, output_dict=True, zero_division=0)

    pd.DataFrame(report_05).transpose().to_csv(
        summary_path(f"{model_name}_classification_report_05.csv"), index=True
    )
    pd.DataFrame(report_best).transpose().to_csv(
        summary_path(f"{model_name}_classification_report_best.csv"), index=True
    )

    with open(summary_path(f"{model_name}_classification_report_05.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, preds_05, zero_division=0))
    with open(summary_path(f"{model_name}_classification_report_best.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, preds_best, zero_division=0))

    plot_classification_report_visual(y_true, preds_05, model_name, "05")
    plot_classification_report_visual(y_true, preds_best, model_name, "best")


# ================================================================
# Optuna Optimization Objective (NO EARLY STOPPING)
# ================================================================
def optimize_xgb(trial, X_train, y_train, X_valid, y_valid, spw, optimize_target="f1"):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1400),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 4.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }

    model = XGBClassifier(**params)

    # FORCE NumPy (XGBoost 2.x safety)
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_valid_np = X_valid.values
    y_valid_np = y_valid.values

    model.fit(
        X_train_np,
        y_train_np,
        eval_set=[(X_valid_np, y_valid_np)],
        verbose=False
    )

    proba = model.predict_proba(X_valid_np)[:, 1]
    preds = (proba >= 0.5).astype(int)

    if optimize_target == "f1_best":
        _, best_f1 = find_best_threshold(y_valid_np, proba)
        return best_f1
    if optimize_target == "pr_auc":
        return average_precision_score(y_valid_np, proba)
    if optimize_target == "auc":
        return roc_auc_score(y_valid_np, proba)
    return f1_score(y_valid_np, preds)


# ================================================================
# Train a Single Model
# ================================================================
def train_single_model(X, y, model_name, n_trials=40, optimize_target="f1"):

    print(f"\nTraining {model_name} model")

    # ===============================
    # SAVE FEATURES USED (IMPORTANT)
    # ===============================
    joblib.dump(
        {
            "columns": list(X.columns),
            "n_features": X.shape[1]
        },
        model_path(f"{model_name}_features.pkl")
    )

    # Optional but VERY useful for SHAP/debugging
    joblib.dump(
        X,
        model_path(f"{model_name}_features_full.pkl")
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    spw = (y_train == 0).sum() / (y_train == 1).sum()

    print("Running Optuna hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: optimize_xgb(t, X_train, y_train, X_valid, y_valid, spw, optimize_target),
        n_trials=n_trials,
        show_progress_bar=False
    )

    print("Best Hyperparameters:", study.best_params)

    model = XGBClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="logloss"
    )

    # Force NumPy (XGBoost 2.x safety)
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_valid_np = X_valid.values
    y_valid_np = y_valid.values

    model.fit(
        X_train_np,
        y_train_np,
        eval_set=[(X_valid_np, y_valid_np)],
        verbose=False
    )

    preds = model.predict(X_valid_np)
    proba = model.predict_proba(X_valid_np)[:, 1]

    acc = accuracy_score(y_valid, preds)
    f1 = f1_score(y_valid, preds)
    prec = precision_score(y_valid, preds)
    rec = recall_score(y_valid, preds)
    auc = roc_auc_score(y_valid, proba)
    pr_auc = average_precision_score(y_valid, proba)

    best_t, best_f1 = find_best_threshold(y_valid, proba)
    preds_best = (proba >= best_t).astype(int)

    print("\nValidation Metrics")
    print(f"Accuracy@0.5: {acc:.4f}")
    print(f"F1@0.5:       {f1:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Recall:       {rec:.4f}")
    print(f"AUC:          {auc:.4f}")
    print(f"PR_AUC:       {pr_auc:.4f}")
    print(f"Best Thresh:  {best_t:.2f}")
    print(f"F1@Best:      {best_f1:.4f}")

    plot_roc(y_valid, proba, model_name)
    plot_pr(y_valid, proba, model_name)
    plot_confusion(y_valid, proba, best_t, model_name)
    plot_threshold_curve(y_valid, proba, model_name)
    save_classification_reports(y_valid, preds, preds_best, model_name)

    joblib.dump(model, model_path(f"{model_name}.pkl"))
    joblib.dump(model, model_path(f"{model_name}_xgb_shap.pkl"))

    joblib.dump(
        {
            "best_threshold": best_t,
            "metrics": {
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "auc": auc,
                "pr_auc": pr_auc
            }
        },
        model_path(f"{model_name}_metadata.pkl")
    )

    return {
        "Accuracy@0.5": acc,
        "F1@0.5": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "BestThreshold": best_t,
        "F1@Best": best_f1
    }


# ================================================================
# MAIN PIPELINE
# ================================================================
def train_models():

    print("\nLoading OULAD dataset...")
    student_info, reg, assess, vle, vle_meta, assess_meta, courses = load_oulad()

    print("\nBuilding EARLY features...")
    X_early, y_early = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta, courses, early_only=True
    )

    early_results = train_single_model(
        X_early, y_early, "at_risk_model", n_trials=90, optimize_target="pr_auc"
    )

    print("\nBuilding FINAL features...")
    X_final, y_final = build_full_features(
        student_info, reg, assess, vle, vle_meta, assess_meta, courses, early_only=False
    )

    final_results = train_single_model(
        X_final, y_final, "final_model", n_trials=70, optimize_target="auc"
    )

    summary = pd.DataFrame({
        "Model": ["Early Risk", "Final Outcome"],
        "Accuracy@0.5": [early_results["Accuracy@0.5"], final_results["Accuracy@0.5"]],
        "F1@0.5": [early_results["F1@0.5"], final_results["F1@0.5"]],
        "Precision": [early_results["Precision"], final_results["Precision"]],
        "Recall": [early_results["Recall"], final_results["Recall"]],
        "ROC_AUC": [early_results["ROC_AUC"], final_results["ROC_AUC"]],
        "PR_AUC": [early_results["PR_AUC"], final_results["PR_AUC"]],
        "BestThreshold": [early_results["BestThreshold"], final_results["BestThreshold"]],
        "F1@Best": [early_results["F1@Best"], final_results["F1@Best"]],
    })

    summary.to_csv(summary_path("train_metrics.csv"), index=False)

    print("\nFINAL RESULTS")
    print(summary)
    print("\nModels and figures saved successfully.")


if __name__ == "__main__":
    train_models()

