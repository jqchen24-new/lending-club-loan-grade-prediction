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
├── train.py                # training script (exported from notebook)
├── predict.py              # Flask prediction service
├── model.pkl               # serialized pipeline
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
| Deployment | Flask API containerized with Docker |

## Results

| Metric        | Random Forest | XGBoost | PyTorch MLP (v1) | PyTorch MLP (v2) |
|---------------|---------------|---------|------------------|------------------|
| Accuracy      | 0.48          | 0.94    | 0.87             | 0.95             |
| Weighted F1   | 0.45          | 0.94    | 0.87             | 0.95             |
| Macro F1      | 0.26          | 0.66    | 0.77             | 0.85             |
| Grade A F1    | 0.65          | 0.97    | 0.93             | 0.99             |
| Grade B F1    | 0.42          | 0.93    | 0.84             | 0.94             |
| Grade C F1    | 0.45          | 0.92    | 0.85             | 0.93             |
| Grade D F1    | 0.21          | 0.92    | 0.87             | 0.96             |
| Grade E F1    | 0.10          | 0.91    | 0.81             | 0.91             |
| Grade F F1    | 0.00          | 0.00    | 0.66             | 0.80             |
| Grade G F1    | 0.00          | 0.00    | 0.46             | 0.45             |

## Running locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Run the prediction service
```bash
python predict.py
```

### 4. Build and run with Docker
```bash
docker build -t loan-grade-predictor .
docker run -p 9696:9696 loan-grade-predictor
```

### 5. Test the endpoint
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"loan_amnt": 10000, "annual_inc": 65000, "dti": 18.5, "fico_range_low": 700, "purpose": "debt_consolidation", "home_ownership": "RENT", "emp_length": "5 years"}'
```

## Requirements

```
pandas
scikit-learn
xgboost
torch
flask
gunicorn
kagglehub
```

## Acknowledgements

Dataset sourced from Kaggle. Project built as part of the ML Zoomcamp curriculum.