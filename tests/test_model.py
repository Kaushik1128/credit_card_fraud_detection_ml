"""Tests for src/model.py — bundle structure and inference helpers."""
import pandas as pd

from src.model import predict, predict_proba


def test_bundle_has_expected_shape(bundle):
    assert set(bundle.keys()) == {"pipeline", "threshold", "feature_columns", "metadata"}
    assert isinstance(bundle["threshold"], float)
    assert 0.0 < bundle["threshold"] < 1.0
    assert len(bundle["feature_columns"]) == 30
    assert "model_name" in bundle["metadata"]


def test_predict_proba_returns_probability_in_unit_interval(bundle, genuine_tx):
    probs = predict_proba(bundle, pd.DataFrame([genuine_tx]))
    assert probs.shape == (1,)
    assert 0.0 <= float(probs[0]) <= 1.0


def test_predict_returns_zero_for_known_genuine(bundle, genuine_tx):
    pred = predict(bundle, pd.DataFrame([genuine_tx]))
    assert pred.tolist() == [0]


def test_predict_returns_one_for_known_fraud(bundle, fraud_tx):
    pred = predict(bundle, pd.DataFrame([fraud_tx]))
    assert pred.tolist() == [1]


def test_predict_proba_is_low_for_genuine(bundle, genuine_tx):
    p = float(predict_proba(bundle, pd.DataFrame([genuine_tx]))[0])
    assert p < 0.01


def test_predict_proba_is_high_for_fraud(bundle, fraud_tx):
    p = float(predict_proba(bundle, pd.DataFrame([fraud_tx]))[0])
    assert p > 0.95


def test_predict_handles_batch_input(bundle, genuine_tx, fraud_tx):
    X = pd.DataFrame([genuine_tx, fraud_tx])
    probs = predict_proba(bundle, X)
    preds = predict(bundle, X)
    assert probs.shape == (2,)
    assert preds.tolist() == [0, 1]
