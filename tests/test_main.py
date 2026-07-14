"""
Pytest suite for AgriRisk FastAPI backend.
Covers the claims made in Section 4.3 of the AI4I proposal:
  - /predict response schema and probability normalization
  - HTTP 422 rejection of out-of-range / missing inputs
  - Graceful handling of an unseen crop value
  - /districts row count and province filtering
  - /health dataset disclosure statement
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID_PAYLOAD = {
    "rainfall_mm": 45,
    "ndvi_proxy_0_1": 0.6,
    "pest_incidents_reported": 3,
    "irrigation_coverage_pct": 40,
    "input_availability_score_0_100": 60,
    "avg_farmgate_price_usd_per_tonne": 300,
    "month": 7,
    "crop": "maize",
    "province": "Mashonaland East",
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------- /predict ----------

def test_predict_returns_valid_schema(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "risk_level" in body
    assert "risk_probabilities" in body
    assert "action_recommendation" in body
    assert body["risk_level"] in ("Low", "Elevated")


def test_predict_probabilities_normalize_to_one(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    probs = resp.json()["risk_probabilities"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01


def test_predict_rejects_out_of_range_ndvi(client):
    bad_payload = {**VALID_PAYLOAD, "ndvi_proxy_0_1": 1.5}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_required_field(client):
    bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "rainfall_mm"}
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_handles_unseen_crop_gracefully(client):
    odd_payload = {**VALID_PAYLOAD, "crop": "dragonfruit_experimental"}
    resp = client.post("/predict", json=odd_payload)
    # Should not 500 -- either a clean prediction or a handled 422/400,
    # never an unhandled server crash.
    assert resp.status_code in (200, 400, 422)


# ---------- /districts ----------

def test_districts_returns_rows(client):
    resp = client.get("/districts")
    assert resp.status_code == 200
    assert len(resp.json()) > 0


def test_districts_filters_by_province(client):
    resp = client.get("/districts", params={"province": "Mashonaland East"})
    assert resp.status_code == 200
    body = resp.json()
    for row in body:
        assert row.get("province") == "Mashonaland East"
        assert "district" in row
        assert "avg_yield_t_per_ha" in row


# ---------- /health ----------

def test_health_discloses_dataset_and_dropped_module(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    statement = resp.json()["dataset_statement"]
    assert "synthetic" in statement.lower()
    assert "yield" in statement.lower()
