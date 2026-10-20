import pytest
from fastapi.testclient import TestClient

from churn.api import create_app
from churn.data import load_raw
from churn.predict import ChurnPredictor
from churn.schemas import Customer

EXAMPLE = Customer.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client(saved_dir):
    # `with` runs the app's startup, which loads the model once.
    with TestClient(create_app(saved_dir)) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info_exposes_threshold_and_metrics(client):
    info = client.get("/model-info").json()
    assert 0.1 <= info["threshold"] <= 0.9
    assert info["test_metrics"]["roc_auc"] > 0.8


def test_predict_returns_probability_and_flag(client):
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["flagged"] == (body["churn_probability"] >= body["threshold"])


def test_api_matches_the_predictor_on_the_same_customer(client, saved_dir):
    expected = ChurnPredictor(saved_dir).predict(load_raw().head(1)).iloc[0]
    body = client.post("/predict", json=EXAMPLE).json()  # EXAMPLE is the first CSV row
    assert body["churn_probability"] == pytest.approx(expected["churn_probability"])


def test_new_customer_may_omit_total_charges(client):
    payload = {**EXAMPLE, "tenure": 0}
    del payload["TotalCharges"]
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"Contract": "Weekly"},        # not an allowed category
        {"tenure": -3},                # below the allowed range
        {"MonthlyCharges": "cheap"},   # wrong type
        {"SeniorCitizen": 2},          # only 0 or 1
    ],
)
def test_invalid_input_is_rejected_with_422(client, change):
    response = client.post("/predict", json={**EXAMPLE, **change})
    assert response.status_code == 422
    assert next(iter(change)) in str(response.json()["detail"])


def test_missing_field_is_rejected_with_422(client):
    payload = {k: v for k, v in EXAMPLE.items() if k != "Contract"}
    assert client.post("/predict", json=payload).status_code == 422


def test_startup_fails_clearly_when_no_model_is_saved(tmp_path):
    with pytest.raises(RuntimeError, match="train.py"):
        with TestClient(create_app(tmp_path / "nothing-here")):
            pass
