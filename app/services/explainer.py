import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import json

MODEL_DIR = os.environ.get("MODEL_DIR", "model")


def model_path(name: str) -> str:
    return os.path.join(MODEL_DIR, name)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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
risk_model = joblib.load(model_path("at_risk_model_xgb_shap.pkl"))
final_model = joblib.load(model_path("final_model_xgb_shap.pkl"))

# ======================================================
# LOAD FEATURE NAMES (AUTHORITATIVE)
# ======================================================
risk_feature_names = load_feature_columns(
    model_path("at_risk_model_features.pkl"),
    "Early Risk"
)

final_feature_names = load_feature_columns(
    model_path("final_model_features.pkl"),
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
    "week_0_clicks": "Clicks on learning materials during week 0",
    "week_1_clicks": "Clicks on learning materials during week 1",
    "week_2_clicks": "Clicks on learning materials during week 2",
    "week_3_clicks": "Clicks on learning materials during week 3",
    "early_click_trend": "Trend of engagement over the early weeks",
    "early_click_volatility": "Variability in early-week engagement",
    "max_gap": "Longest period of inactivity",
    "longest_streak": "Longest consecutive activity streak",
    "avg_score": "Average assessment performance",
    "timed_score_mean": "Assessment score weighted by timing in course",
    "missing_assessments": "Number of expected assessments not submitted",
    "late_submissions": "Count of late submissions",
    "activity_diversity": "Diversity of learning activity types",
    "age_band": "Age group of learner",
    "highest_education": "Educational background",
    "gender": "Gender encoded feature",
    "disability": "Disability support indicator",
    "region_freq": "Frequency-based regional representation"
}

FINAL_FEATURE_MEANINGS = {
    "total_clicks": "Total click activity across course",
    "clicks_per_week": "Average clicks per course week",
    "avg_score": "Average assessment score",
    "timed_score_mean": "Assessment score weighted by assessment timing",
    "num_assessments": "Number of completed assessments",
    "missing_assessments": "Number of missing expected assessments",
    "first_score": "First assessment score",
    "last_score": "Last assessment score",
    "score_improvement": "Change from first to last score",
    "late_submissions": "Count of late submissions",
    "active_days": "Estimated active registration duration",
    "registered_late": "Late registration indicator",
    "activity_diversity": "Diversity of LMS activity types",
    "longest_streak": "Longest consecutive activity streak",
    "max_gap": "Longest inactivity gap",
    "avg_difficulty": "Average difficulty of attempted assessments",
    "click_mean": "Mean clicks per interaction record",
    "click_std": "Std deviation of clicks",
    "click_min": "Minimum click count observed",
    "click_max": "Maximum click count observed",
    "click_skew": "Asymmetry of click distribution",
    "click_kurt": "Tail heaviness of click distribution",
    "click_iqr": "Interquartile range of click counts",
    "age_band": "Age group of learner",
    "highest_education": "Educational background",
    "gender": "Gender encoded feature",
    "disability": "Disability support indicator",
    "region_freq": "Frequency-based regional representation"
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
    abs_sum = float(np.sum(np.abs(shap_values))) + 1e-12

    # Tie local factors to cohort-level global ranking when available.
    stage = "early" if feature_names == risk_feature_names else "final"
    global_summary = get_global_shap_summary(stage=stage, top_k=max(30, top_k))
    global_rank = {
        item["feature"]: idx + 1
        for idx, item in enumerate(global_summary.get("top_features", []))
    }

    for name, value in zip(feature_names, shap_values):
        local_share = abs(float(value)) / abs_sum
        explanations.append({
            "feature": name,
            "impact": float(value),
            "direction": "increases_class_1_probability" if float(value) > 0 else "decreases_class_1_probability",
            "contribution_share": float(local_share),
            "global_rank": global_rank.get(name),
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
    total_logit = base_value + float(np.sum(shap_row))

    all_factors = build_explanations(
        shap_row,
        risk_feature_names,
        RISK_FEATURE_MEANINGS,
        top_k=len(risk_feature_names)
    )
    pos_factors = [x for x in all_factors if x["impact"] > 0][:5]
    neg_factors = [x for x in all_factors if x["impact"] < 0][:5]

    return {
        "base_value": base_value,
        "base_probability": float(sigmoid(base_value)),
        "reconstructed_logit": total_logit,
        "reconstructed_probability": float(sigmoid(total_logit)),
        "top_factors": all_factors[:5],
        "top_increasing_factors": pos_factors,
        "top_decreasing_factors": neg_factors,
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
    total_logit = base_value + float(np.sum(shap_row))

    all_factors = build_explanations(
        shap_row,
        final_feature_names,
        FINAL_FEATURE_MEANINGS,
        top_k=len(final_feature_names)
    )
    pos_factors = [x for x in all_factors if x["impact"] > 0][:7]
    neg_factors = [x for x in all_factors if x["impact"] < 0][:7]

    return {
        "base_value": base_value,
        "base_probability": float(sigmoid(base_value)),
        "reconstructed_logit": total_logit,
        "reconstructed_probability": float(sigmoid(total_logit)),
        "top_factors": all_factors[:7],
        "top_increasing_factors": pos_factors,
        "top_decreasing_factors": neg_factors,
        "raw_shap_values": shap_row.tolist()
    }


def _global_summary_path(stage):
    return os.path.join("outputs", "summaries", f"global_shap_{stage}.json")


def _build_global_summary(shap_vals, feature_names, top_k):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_signed = shap_vals.mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = []
    for idx in order[:top_k]:
        top.append({
            "feature": feature_names[int(idx)],
            "mean_abs_shap": float(mean_abs[int(idx)]),
            "mean_shap": float(mean_signed[int(idx)]),
            "dominant_direction": (
                "towards_class_1" if mean_signed[int(idx)] > 0 else "towards_class_0"
            )
        })
    return top


def get_global_shap_summary(stage: str, top_k: int = 15):
    stage = stage.lower()
    if stage not in {"early", "final"}:
        raise ValueError("stage must be 'early' or 'final'")

    out_path = _global_summary_path(stage)
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: compute from cached feature matrices produced during training.
    if stage == "early":
        features = joblib.load(model_path("at_risk_model_features_full.pkl")).values
        shap_vals = risk_explainer.shap_values(features)
        summary = {
            "stage": "early",
            "top_features": _build_global_summary(shap_vals, risk_feature_names, top_k),
            "n_features": len(risk_feature_names)
        }
    else:
        features = joblib.load(model_path("final_model_features_full.pkl")).values
        shap_vals = final_explainer.shap_values(features)
        summary = {
            "stage": "final",
            "top_features": _build_global_summary(shap_vals, final_feature_names, top_k),
            "n_features": len(final_feature_names)
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

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
    risk_summary = {
        "stage": "early",
        "top_features": _build_global_summary(shap_vals_risk, risk_feature_names, 20),
        "n_features": len(risk_feature_names)
    }
    os.makedirs("outputs/summaries", exist_ok=True)
    with open(_global_summary_path("early"), "w", encoding="utf-8") as f:
        json.dump(risk_summary, f, indent=2)

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

    # Dependence plot for top global feature (early model)
    risk_importance = np.abs(shap_vals_risk).mean(axis=0)
    risk_top_idx = int(np.argmax(risk_importance))
    shap.dependence_plot(
        risk_top_idx,
        shap_vals_risk,
        X_risk,
        feature_names=risk_feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/early_shap_dependence_top1.png", dpi=300)
    plt.close()

    # Local example waterfall (early model)
    early_sample = 0
    early_expl = shap.Explanation(
        values=shap_vals_risk[early_sample],
        base_values=float(risk_explainer.expected_value),
        data=X_risk[early_sample],
        feature_names=risk_feature_names
    )
    shap.plots.waterfall(early_expl, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig("outputs/figures/early_shap_waterfall_example.png", dpi=300)
    plt.close()

    # -----------------------
    # FINAL OUTCOME MODEL
    # -----------------------
    _validate_alignment(X_final, final_feature_names, "Final Outcome")

    shap_vals_final = final_explainer.shap_values(X_final)
    final_summary = {
        "stage": "final",
        "top_features": _build_global_summary(shap_vals_final, final_feature_names, 20),
        "n_features": len(final_feature_names)
    }
    with open(_global_summary_path("final"), "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

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

    # Dependence plot for top global feature (final model)
    final_importance = np.abs(shap_vals_final).mean(axis=0)
    final_top_idx = int(np.argmax(final_importance))
    shap.dependence_plot(
        final_top_idx,
        shap_vals_final,
        X_final,
        feature_names=final_feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/figures/final_shap_dependence_top1.png", dpi=300)
    plt.close()

    # Local example waterfall (final model)
    final_sample = 0
    final_expl = shap.Explanation(
        values=shap_vals_final[final_sample],
        base_values=float(final_explainer.expected_value),
        data=X_final[final_sample],
        feature_names=final_feature_names
    )
    shap.plots.waterfall(final_expl, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig("outputs/figures/final_shap_waterfall_example.png", dpi=300)
    plt.close()
