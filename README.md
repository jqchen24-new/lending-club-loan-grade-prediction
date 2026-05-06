# Loan Grade Prediction — Lending Club
**ML Zoomcamp Capstone Project**

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red) ![Docker](https://img.shields.io/badge/Docker-ready-green)

## Problem description

LendingClub assigned each loan a grade from A (lowest risk) to G (highest risk) based on the borrower's credit profile. This project builds a multi-class classifier that predicts a loan's grade from borrower attributes available at origination — replicating the kind of risk tiering decision used in consumer credit underwriting.

A deployed model could help a lender automatically route applications to the correct risk tier, price interest rates, or flag borderline cases for manual review.

## Dataset

- **Source**: [Kaggle — adarshsng/lending-club-loan-data-csv](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)
- **Size**: ~2M rows, 150 columns
- **Target**: `grade` (A / B / C / D / E / F / G)

## Project structure

```
Capstone_project/   # local clone folder name may differ
├── notebook.ipynb           # EDA, feature engineering, experiments
├── train.py                 # Training script (saves predictor.pkl)
├── model.py                 # LoanGradePredictor and pipeline definitions
├── predict.py               # FastAPI app (predict, explain, batch)
├── explain.py               # SHAP KernelExplainer + explanation row builder
├── streamlit_app.py         # Browser UI (HTTP client to the API)
├── background.csv           # Background sample for SHAP (must match model features)
├── predictor.pkl            # Serialized trained artifact
├── tests/
│   └── test_api.py          # API tests (pytest)
├── Dockerfile               # FastAPI + model image
├── Dockerfile.streamlit     # Streamlit UI only (calls API over HTTP)
├── requirements.txt
├── requirements-streamlit.txt  # Slim deps for the UI container (no torch)
├── sklearn/                 # Optional XGBoost experiment artifacts
└── README.md
```

## Workflow

| Step | Description |
|---|---|
| EDA | Class distribution, FICO vs grade, interest rate by grade, null analysis, correlation heatmap |
| Data preprocessing | Drop post-origination leakage columns, impute nulls, encode categoricals |
| Baseline | Random Forest (scikit-learn) |
| XGBoost | XGBoost |
| Neural net | PyTorch MLP |
| Evaluation | Accuracy, Log-loss, Weighted F1, macro F1 |
| Explanations | SHAP (KernelExplainer) with `background.csv` aligned to the predictor |
| Deployment | FastAPI on Railway; optional Streamlit UI as a second Railway service ([Dockerfile.streamlit](Dockerfile.streamlit)) |

## Results

| Metric        | Random Forest | XGBoost | PyTorch MLP (v1) | PyTorch MLP (v2) |
|---------------|---------------|---------|------------------|------------------|
| Accuracy      | 0.54          | 0.94    | 0.91             | 0.95             |
| Weighted F1   | 0.52          | 0.94    | 0.91             | 0.95             |
| Macro F1      | 0.30          | 0.82    | 0.81             | 0.86             |
| Grade A F1    | 0.71          | 0.98    | 0.97             | 0.99             |
| Grade B F1    | 0.51          | 0.92    | 0.90             | 0.94             |
| Grade C F1    | 0.52          | 0.91    | 0.89             | 0.93             |
| Grade D F1    | 0.30          | 0.94    | 0.91             | 0.96             |
| Grade E F1    | 0.05          | 0.92    | 0.88             | 0.92             |
| Grade F F1    | 0.00          | 0.81    | 0.74             | 0.82             |
| Grade G F1    | 0.00          | 0.26    | 0.38             | 0.46             |

## Limitations and trust

- **`risk_level` in the API** is a fixed mapping from the **predicted letter grade** (A → “Very Low Risk”, etc.). It is not an independent second model and should not be read as real underwriting.
- The MLP can behave **out of distribution** on extreme or unusual combinations of fields (very large amounts vs typical training rows, peaked softmax on one grade, etc.). The Streamlit app shows **reliability warnings** when inputs look high-stress but the model still predicts A or B; treat those cases as demos, not credit decisions.
- SHAP values depend on the **background sample** (`background.csv`). Regenerate it if you change the feature pipeline or retrain.

## Running locally

### Prerequisites

- Python 3.10 (recommended; match training if you retrain)
- Docker (optional, for containerized API)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API

From the **project root** (so `predictor.pkl` and `background.csv` resolve):

```bash
uvicorn predict:app --host 0.0.0.0 --port 9696
```

Equivalent:

```bash
python predict.py
```

Interactive docs: [http://127.0.0.1:9696/docs](http://127.0.0.1:9696/docs)

### 3. Run the Streamlit UI (optional)

Start the API first, then in another terminal:

```bash
streamlit run streamlit_app.py
```

Override the API base URL if needed:

```bash
export LOAN_API_BASE=http://127.0.0.1:9696
streamlit run streamlit_app.py
```

### 4. Run tests

```bash
pytest tests/test_api.py
```

### 5. Call the API with curl

**Predict only** (grade + `risk_level` + `grade_probabilities`):

```bash
curl -s -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 35000, "annual_inc": 32000, "dti": 42.0,
    "term": " 60 months", "home_ownership": "RENT",
    "revol_util": 95.0, "delinq_2yrs": 5, "pub_rec": 2,
    "pub_rec_bankruptcies": 1, "num_tl_90g_dpd_24m": 4,
    "pct_tl_nvr_dlq": 40.0, "installment": 950.0,
    "funded_amnt": 35000, "bc_util": 92.0
  }' | python -m json.tool
```

**Predict with SHAP explanation** (adds `explanation`):

```bash
curl -s -X POST "http://localhost:9696/predict?explain=true&top_n=10&max_evals=256" \
  -H "Content-Type: application/json" \
  -d '{"loan_amnt": 10000, "annual_inc": 60000, "dti": 18.0, "revol_util": 30.0}' \
  | python -m json.tool
```

**Dedicated explain endpoint** (same payload shape as `/predict`; always returns `explanation`):

```bash
curl -s -X POST "http://localhost:9696/explain?top_n=10&max_evals=256" \
  -H "Content-Type: application/json" \
  -d '{"loan_amnt": 10000, "annual_inc": 60000, "dti": 18.0}' \
  | python -m json.tool
```

## API reference (brief)

| Method | Path | Notes |
|--------|------|--------|
| `GET` | `/health` | `{ "status": "ok" }` |
| `POST` | `/predict` | Body: partial or full `LoanInput` JSON. Query: `explain` (bool), `top_n`, `max_evals` when `explain=true`. |
| `POST` | `/explain` | Same body; always includes SHAP-style `explanation`. Query: `top_n`, `max_evals`. |
| `POST` | `/batch` | Body: `{ "loans": [ { ... }, ... ] }`. Query: optional `explain`, `top_n`, `max_evals`. |

**Typical JSON fields** (non-exhaustive; see OpenAPI schema in `/docs`):

- `grade`: predicted letter `A`–`G`
- `risk_level`: human-readable label derived from `grade`
- `grade_probabilities`: map of letter → probability (one row per prediction)
- `explanation` (when requested): list of `{ "feature", "value", "shap_value", "direction" }` rows (top drivers by absolute SHAP value)

## Build and run with Docker

```bash
docker build -t loan-grade-predictor .
docker run -p 9696:9696 loan-grade-predictor
```

**SHAP in Docker:** The checked-in [Dockerfile](Dockerfile) copies `predict.py`, `model.py`, and `predictor.pkl` only. [predict.py](predict.py) imports [explain.py](explain.py) at startup, so the container **will fail to start** unless **`explain.py`** is copied into the image. **`background.csv`** is read when the SHAP explainer is first built; without it, **`POST /predict`** (without `explain`) may still work, but **`/explain`**, **`/predict?explain=true`**, and **`/batch?explain=true`** fail until `background.csv` is present (add **`COPY explain.py .`** and **`COPY background.csv .`** to match local parity).

**Streamlit image (local):**

```bash
docker build -f Dockerfile.streamlit -t loan-grade-ui .
docker run --rm -p 8501:8501 \
  -e LOAN_API_BASE=http://host.docker.internal:9696 \
  loan-grade-ui
```

Use your machine’s reachable API URL instead of `host.docker.internal` on Linux if needed (`172.17.0.1` or the host LAN IP).

## Cloud Deployment

The model API is deployed on Railway and publicly accessible.

**Live URLs:**

| | |
|---|---|
| **Streamlit UI** | [https://lending-club-loan-grade-prediction-production-4bbc.up.railway.app](https://lending-club-loan-grade-prediction-production-4bbc.up.railway.app) |
| **FastAPI** | [https://lending-club-loan-grade-prediction-production.up.railway.app](https://lending-club-loan-grade-prediction-production.up.railway.app) |

The deployed API image must include **`explain.py`** (otherwise the process exits on import). It should also include **`background.csv`** if you rely on **`/explain`** or **`explain=true`** (same as local).

### Railway: Streamlit UI (second service)

Keep your **existing FastAPI service** as-is. Add a **new** Railway service from the **same GitHub repo** for the browser UI.

Use the GitHub **branch** that contains `Dockerfile.streamlit` (for example `SHAP`); if Railway deploys **`main`** only, merge that branch or switch the service’s tracked branch.

1. In your Railway project, click **New** → **GitHub Repo** → select this repository (or **Empty service** → connect repo).
2. Open the new service → **Settings**:
   - Under **Build**, set **Dockerfile path** to `Dockerfile.streamlit` (not the root `Dockerfile`).
3. **Variables** → add:
   - **`LOAN_API_BASE`** = your API’s public origin only, e.g. `https://lending-club-loan-grade-prediction-production.up.railway.app`  
     (no `/predict` suffix; the app calls `/explain` and `/predict` internally.)
4. **Networking**: generate a **public domain** for the Streamlit service (Railway assigns **`PORT`** automatically; the image listens on `$PORT`).
5. Deploy and open the Streamlit URL. If the page loads but requests fail, check logs: wrong `LOAN_API_BASE`, HTTP vs HTTPS, or API cold-start timeouts.

The Streamlit container installs only [requirements-streamlit.txt](requirements-streamlit.txt) (no PyTorch or `predictor.pkl`)—SHAP still runs on the **API** when you use `/explain`.

### Health check

```bash
curl https://lending-club-loan-grade-prediction-production.up.railway.app/health
```

### Test the live API

```bash
curl -X POST https://lending-club-loan-grade-prediction-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 35000,
    "annual_inc": 32000,
    "dti": 42.0,
    "term": " 60 months",
    "home_ownership": "RENT",
    "revol_util": 95.0,
    "delinq_2yrs": 5,
    "pub_rec": 2,
    "pub_rec_bankruptcies": 1,
    "num_tl_90g_dpd_24m": 4,
    "pct_tl_nvr_dlq": 40.0,
    "installment": 950.0,
    "funded_amnt": 35000,
    "bc_util": 92.0
  }'
```

### Interactive API docs

[https://lending-club-loan-grade-prediction-production.up.railway.app/docs](https://lending-club-loan-grade-prediction-production.up.railway.app/docs)

## Training

> For best results, run `train.py` on Google Colab with T4 GPU runtime.

**On Colab (recommended):**

```bash
!python train.py
```

**Locally on CPU (faster, fewer rows):**

```bash
python train.py --cpu --nrows 150000
```

The script downloads the dataset automatically via `kagglehub` and saves `predictor.pkl`.

## Requirements

All versions are pinned in [requirements.txt](requirements.txt). Install with:

```bash
pip install -r requirements.txt
```

Notable libraries beyond the core stack: **FastAPI**, **uvicorn**, **torch**, **scikit-learn**, **xgboost**, **shap** (explanations), **httpx** (Streamlit client), **streamlit** (UI), **pytest** (tests).

## Acknowledgements

Dataset sourced from Kaggle. Project built as part of the ML Zoomcamp curriculum.
