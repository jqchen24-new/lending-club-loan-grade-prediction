"""Smoke tests for FastAPI routes (run from repo root; requires predictor.pkl + background.csv)."""

import os
import sys

import pytest
from starlette.testclient import TestClient

# Import app with cwd = repo root so artifact paths resolve
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != _REPO:
    os.chdir(_REPO)
sys.path.insert(0, _REPO)


@pytest.fixture(scope="module")
def client():
    import predict  # noqa: WPS433 — loads model on import

    return TestClient(predict.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_minimal(client):
    r = client.post("/predict", json={"loan_amnt": 10000, "dti": 18.5})
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"grade", "risk_level", "grade_probabilities"}
    assert isinstance(data["grade"], str)
    assert isinstance(data["risk_level"], str)
    gp = data["grade_probabilities"]
    assert isinstance(gp, dict) and len(gp) == 7
    assert abs(sum(gp.values()) - 1.0) < 1e-5
    assert data["grade"] in gp


def test_predict_explain_shape(client):
    r = client.post(
        "/predict?explain=true&top_n=3&max_evals=256",
        json={"loan_amnt": 10000, "dti": 18.5},
    )
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"grade", "risk_level", "grade_probabilities", "explanation"}
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) <= 3
    for row in data["explanation"]:
        assert set(row.keys()) == {"feature", "value", "shap_value", "direction"}
        if row["shap_value"] > 0:
            assert row["direction"] == "increases risk"
        elif row["shap_value"] < 0:
            assert row["direction"] == "decreases risk"
        else:
            assert row["direction"] == "neutral"


def test_explain_shape(client):
    r = client.post(
        "/explain?top_n=3&max_evals=256",
        json={"loan_amnt": 10000, "dti": 18.5},
    )
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"grade", "risk_level", "grade_probabilities", "explanation"}
    assert isinstance(data["explanation"], list)
    assert isinstance(data["grade_probabilities"], dict)
    assert len(data["grade_probabilities"]) == 7


def test_batch_predict(client):
    r = client.post(
        "/batch",
        json={
            "loans": [
                {"loan_amnt": 10000, "dti": 18.5},
                {"loan_amnt": 20000, "dti": 10.0},
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    for p in data["predictions"]:
        assert set(p.keys()) == {"grade", "risk_level", "grade_probabilities"}
        assert isinstance(p["grade"], str)
        assert len(p["grade_probabilities"]) == 7
        assert "explanation" not in p


def test_batch_predict_explain(client):
    r = client.post(
        "/batch?explain=true&top_n=3&max_evals=256",
        json={
            "loans": [
                {"loan_amnt": 10000, "dti": 18.5},
                {"loan_amnt": 20000, "dti": 10.0},
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    for p in data["predictions"]:
        assert set(p.keys()) == {"grade", "risk_level", "grade_probabilities", "explanation"}
        assert isinstance(p["grade"], str)
        assert isinstance(p["risk_level"], str)
        assert isinstance(p["explanation"], list)
        assert len(p["explanation"]) <= 3
        for row in p["explanation"]:
            assert set(row.keys()) == {"feature", "value", "shap_value", "direction"}
            if row["shap_value"] > 0:
                assert row["direction"] == "increases risk"
            elif row["shap_value"] < 0:
                assert row["direction"] == "decreases risk"
            else:
                assert row["direction"] == "neutral"
