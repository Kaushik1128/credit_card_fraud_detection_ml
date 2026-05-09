"""Shared pytest fixtures.

Tests skip cleanly with a clear message if the model bundle or test
parquet hasn't been built yet — see README.md "Quickstart" for setup.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

BUNDLE_PATH = Path("models/fraud_model.joblib")
TEST_PARQUET_PATH = Path("data/processed/test.parquet")


@pytest.fixture(scope="session")
def bundle():
    if not BUNDLE_PATH.exists():
        pytest.skip(
            f"{BUNDLE_PATH} not found — run `python -m src.finalize_model` first."
        )
    return joblib.load(BUNDLE_PATH)


@pytest.fixture(scope="session")
def client():
    """A TestClient that runs the FastAPI lifespan (so the model loads)."""
    if not BUNDLE_PATH.exists():
        pytest.skip(
            f"{BUNDLE_PATH} not found — run `python -m src.finalize_model` first."
        )
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def _scored_test_df(bundle):
    """Test split with predicted scores attached, for picking confident examples."""
    if not TEST_PARQUET_PATH.exists():
        pytest.skip(
            f"{TEST_PARQUET_PATH} not found — run `python -m src.split_data` first."
        )
    df = pd.read_parquet(TEST_PARQUET_PATH)
    X = df[bundle["feature_columns"]]
    scores = bundle["pipeline"].predict_proba(X)[:, 1]
    return df.assign(score=scores)


@pytest.fixture
def genuine_tx(_scored_test_df, bundle) -> dict:
    """A test-set transaction the model classifies as genuine with very high confidence."""
    row = _scored_test_df[
        (_scored_test_df["Class"] == 0) & (_scored_test_df["score"] < 1e-4)
    ].iloc[0]
    return {c: float(row[c]) for c in bundle["feature_columns"]}


@pytest.fixture
def fraud_tx(_scored_test_df, bundle) -> dict:
    """A test-set transaction the model classifies as fraud with very high confidence."""
    row = _scored_test_df[
        (_scored_test_df["Class"] == 1) & (_scored_test_df["score"] > 0.99)
    ].iloc[0]
    return {c: float(row[c]) for c in bundle["feature_columns"]}
