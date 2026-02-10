import os
import sys
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.train import train_models


EXPECTED_FIGURES = [
    "at_risk_model_roc.png",
    "at_risk_model_pr.png",
    "at_risk_model_confusion.png",
    "at_risk_model_confusion_normalized.png",
    "at_risk_model_threshold_f1.png",
    "at_risk_model_classification_report_05.png",
    "at_risk_model_classification_report_best.png",
    "final_model_roc.png",
    "final_model_pr.png",
    "final_model_confusion.png",
    "final_model_confusion_normalized.png",
    "final_model_threshold_f1.png",
    "final_model_classification_report_05.png",
    "final_model_classification_report_best.png",
    "early_shap_summary.png",
    "early_shap_bar.png",
    "early_shap_dependence_top1.png",
    "early_shap_waterfall_example.png",
    "final_shap_summary.png",
    "final_shap_bar.png",
    "final_shap_dependence_top1.png",
    "final_shap_waterfall_example.png",
]

EXPECTED_SUMMARIES = [
    "train_metrics.csv",
    "at_risk_model_classification_report_05.csv",
    "at_risk_model_classification_report_05.txt",
    "at_risk_model_classification_report_best.csv",
    "at_risk_model_classification_report_best.txt",
    "final_model_classification_report_05.csv",
    "final_model_classification_report_05.txt",
    "final_model_classification_report_best.csv",
    "final_model_classification_report_best.txt",
]

EXPECTED_REGISTRY = "app/data/students_registry.json"


def verify_artifacts():
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    summaries_dir = PROJECT_ROOT / "outputs" / "summaries"

    manifest_name = "artifact_manifest.json"
    missing_figures = [f for f in EXPECTED_FIGURES if not (figures_dir / f).exists()]
    missing_summaries = [f for f in EXPECTED_SUMMARIES if not (summaries_dir / f).exists()]
    missing_registry = not (PROJECT_ROOT / EXPECTED_REGISTRY).exists()

    manifest = {
        "counts": {
            "figures_expected": len(EXPECTED_FIGURES),
            "summaries_expected": len(EXPECTED_SUMMARIES),
            "registry_expected": 1,
            "figures_generated": len([p for p in figures_dir.glob("*.png")]) if figures_dir.exists() else 0,
            "summaries_generated": len(
                [p for p in summaries_dir.iterdir() if p.is_file() and p.name != manifest_name]
            ) if summaries_dir.exists() else 0,
        },
        "missing": {
            "figures": missing_figures,
            "summaries": missing_summaries,
            "registry": [EXPECTED_REGISTRY] if missing_registry else [],
        },
        "ok": (len(missing_figures) == 0 and len(missing_summaries) == 0 and not missing_registry),
    }

    out_path = PROJECT_ROOT / "outputs" / "summaries" / manifest_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest, out_path


def main():
    print("Step 1/4: Training models and generating evaluation artifacts...")
    train_models()

    print("Step 2/4: Generating SHAP figures...")
    # Import lazily to avoid loading model-dependent modules before training.
    from scripts.generate_shap_figures import main as generate_shap_figures_main
    generate_shap_figures_main()

    print("Step 3/4: Building students registry...")
    # Import lazily to reuse trained feature caches from Step 1.
    from scripts.build_student_registry import build_registry
    build_registry()

    print("Step 4/4: Verifying artifacts...")
    manifest, path = verify_artifacts()

    print("\nArtifact summary")
    print(f"Figures expected:   {manifest['counts']['figures_expected']}")
    print(f"Summaries expected: {manifest['counts']['summaries_expected']}")
    print(f"Figures found:      {manifest['counts']['figures_generated']}")
    print(f"Summaries found:    {manifest['counts']['summaries_generated']}")
    print(f"Manifest:           {path}")
    print(f"Status:             {'OK' if manifest['ok'] else 'MISSING ARTIFACTS'}")

    if not manifest["ok"]:
        print("\nMissing figures:")
        for item in manifest["missing"]["figures"]:
            print(f" - {item}")
        print("Missing summaries:")
        for item in manifest["missing"]["summaries"]:
            print(f" - {item}")
        print("Missing registry:")
        for item in manifest["missing"]["registry"]:
            print(f" - {item}")


if __name__ == "__main__":
    main()
