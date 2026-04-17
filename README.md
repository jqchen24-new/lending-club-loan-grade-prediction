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
lending-club-grade-prediction/
├── notebook.ipynb          # EDA, feature engineering, model training
├── train.py                # training script
├── predict.py              # FastAPI prediction service  ← was Flask
├── model.py                # model and predictor class definitions  ← add this
├── predictor.pkl           # serialized model artifact
├── Dockerfile
├── requirements.txt
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
| Deployment | FastAPI containerized with Docker |

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

## Running locally

### Prerequisites
- Python 3.10
- Docker

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the prediction service
```bash
python predict.py
```

### 3. Test the API
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 35000, "annual_inc": 32000, "dti": 42.0,
    "term": " 60 months", "home_ownership": "RENT",
    "revol_util": 95.0, "delinq_2yrs": 5, "pub_rec": 2,
    "pub_rec_bankruptcies": 1, "num_tl_90g_dpd_24m": 4,
    "pct_tl_nvr_dlq": 40.0, "installment": 950.0,
    "funded_amnt": 35000, "bc_util": 92.0
  }'
```

### 4. Build and run with Docker
```bash
docker build -t loan-grade-predictor .
docker run -p 9696:9696 loan-grade-predictor
```

## Training

> ⚠️ For best results, run `train.py` on Google Colab with T4 GPU runtime.

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

```
pandas
scikit-learn==1.6.1
xgboost
torch
fastapi
uvicorn
kagglehub
joblib
numpy
```

## Acknowledgements

Dataset sourced from Kaggle. Project built as part of the ML Zoomcamp curriculum.