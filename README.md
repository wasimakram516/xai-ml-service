# XAI ML Service (From Black Box to Glass Box)

FastAPI-based ML service for adaptive learning that provides:

- Early at-risk prediction
- Final outcome prediction
- Local SHAP explanations per student
- Global SHAP explanations for cohort-level insight
- Auth-protected endpoints for teacher-facing use

## Project Goals

This service is designed to support trust-aware educational AI by combining predictive performance with explainability.

Current pipeline supports:

- High-performance XGBoost models for early/final tasks
- Threshold-aware prediction outputs
- Rich local explanation payloads (direction, contribution share, probability context)
- Global explanation APIs for UI dashboards

## Tech Stack

- Python, FastAPI
- XGBoost, Optuna, scikit-learn
- SHAP
- SQLite (teacher auth/profile)

## Repository Structure

- `app/main.py`: FastAPI entrypoint
- `app/routers/auth.py`: register/login endpoints
- `app/routers/teachers.py`: teacher profile/password endpoints
- `app/routers/students.py`: student list/profile, insights, global SHAP endpoints
- `app/services/predictor.py`: model inference + threshold + confidence
- `app/services/explainer.py`: local/global SHAP logic
- `app/utils/preprocess.py`: OULAD preprocessing and feature engineering
- `model/train.py`: model training + metrics + evaluation artifacts
- `scripts/run_artifact_pipeline.py`: one-command train + SHAP + registry + verification
- `scripts/build_student_registry.py`: registry generation
- `scripts/generate_shap_figures.py`: SHAP figure generation

## Data Inputs

Expected OULAD files under `data/oulad/`:

- `studentInfo.csv`
- `studentRegistration.csv`
- `studentAssessment.csv`
- `studentVle.csv`
- `vle.csv`
- `assessments.csv`
- `courses.csv`

## Setup

1. Create and activate virtual environment.
2. Install dependencies.

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Train + Generate All Artifacts

Run a single command:

```powershell
python scripts/run_artifact_pipeline.py
```

This performs:

1. Model training (`model/train.py`)
2. SHAP figure generation
3. Student registry generation
4. Artifact verification manifest

## Model Storage (Explicit)

Model files are stored under `MODEL_DIR`.

- Default: `model/`
- Override with environment variable:

```powershell
$env:MODEL_DIR="E:\path\to\model_store"
python scripts/run_artifact_pipeline.py
```

Used consistently by:

- `model/train.py`
- `app/services/predictor.py`
- `app/services/explainer.py`
- relevant scripts

## Generated Outputs

### Summaries

- `outputs/summaries/train_metrics.csv`
- classification reports (`*_classification_report_05.*`, `*_classification_report_best.*`)
- global SHAP summaries:
  - `outputs/summaries/global_shap_early.json`
  - `outputs/summaries/global_shap_final.json`
- artifact manifest:
  - `outputs/summaries/artifact_manifest.json`

### Figures

- ROC / PR / confusion / normalized confusion / threshold-vs-F1
- classification report visuals (PNG)
- SHAP figures (summary, bar, dependence, waterfall)

### Registry

- `app/data/students_registry.json`

## Run API

```powershell
uvicorn app.main:app --reload
```

Swagger UI:

- `http://127.0.0.1:8000/docs`

## Authentication Flow

1. `POST /auth/register`
2. `POST /auth/login` -> get `access_token`
3. Send header on protected routes:

```http
Authorization: Bearer <token>
```

## Main Endpoints

### Students

- `GET /students/`
- `GET /students/{student_id}`
- `GET /students/{student_id}/insights?stage=early|final&top_k=10`
- `GET /students/global/early?top_k=15`
- `GET /students/global/final?top_k=15`

### Teachers

- `GET /teachers/me`
- `PUT /teachers/me`
- `PUT /teachers/me/password`
- `DELETE /teachers/me`

## Explanation Design

### Local explanations (student-level)

Returned in `/students/{id}/insights`:

- top factors
- increasing vs decreasing factors
- contribution share
- global rank alignment
- base/reconstructed probability context

### Global explanations (cohort-level)

Returned in `/students/global/*`:

- top features by mean absolute SHAP
- mean SHAP sign/direction
- cached JSON summaries for UI

## Notes

- Early model is optimized for intervention usefulness (precision-recall tradeoff), not only raw accuracy.
- Final model is optimized for strong discrimination (high AUC/PR-AUC).
- For reproducibility, use the full pipeline script instead of running individual pieces manually.
