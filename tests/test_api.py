"""Tests for app/main.py via FastAPI's TestClient.

TestClient runs the app's lifespan (loading the model bundle) and dispatches
requests in-process — no real HTTP server is started.
"""


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_model_info_includes_expected_fields(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "xgboost_v1"
    assert 0 < body["threshold"] < 1
    assert len(body["feature_columns"]) == 30
    assert "test_metrics" in body
    assert body["test_metrics"]["pr_auc"] > 0.8  # we know test PR-AUC is ~0.877


def test_predict_classifies_genuine_correctly(client, genuine_tx):
    r = client.post("/predict", json=genuine_tx)
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is False
    assert body["fraud_probability"] < 0.01


def test_predict_classifies_fraud_correctly(client, fraud_tx):
    r = client.post("/predict", json=fraud_tx)
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is True
    assert body["fraud_probability"] > 0.95


def test_predict_rejects_missing_fields(client):
    r = client.post("/predict", json={"Time": 0, "Amount": 100})
    assert r.status_code == 422
    detail = r.json()["detail"]
    # All 28 V-features are missing; Pydantic should report each.
    assert len(detail) == 28


def test_predict_rejects_negative_amount(client, genuine_tx):
    bad = dict(genuine_tx)
    bad["Amount"] = -50
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_batch_returns_predictions_in_order(client, genuine_tx, fraud_tx):
    r = client.post("/predict/batch", json={"transactions": [genuine_tx, fraud_tx]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[0]["is_fraud"] is False
    assert preds[1]["is_fraud"] is True


def test_predict_batch_handles_empty_list(client):
    r = client.post("/predict/batch", json={"transactions": []})
    assert r.status_code == 200
    assert r.json()["predictions"] == []
