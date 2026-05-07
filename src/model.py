"""Final fraud-detection bundle and inference helpers.

The serialized bundle is a plain dict so consumers can load it with joblib
without importing any project-specific class. The TypedDict gives editors
and type-checkers awareness of the bundle's shape without affecting runtime.
"""
from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from sklearn.pipeline import Pipeline


class ModelBundle(TypedDict):
    pipeline: Pipeline
    threshold: float
    feature_columns: list[str]
    metadata: dict[str, Any]


def predict_proba(bundle: ModelBundle, X) -> np.ndarray:
    """Return P(fraud) for each row of X."""
    return bundle["pipeline"].predict_proba(X)[:, 1]


def predict(bundle: ModelBundle, X) -> np.ndarray:
    """Return 0/1 labels using the bundle's chosen decision threshold."""
    return (predict_proba(bundle, X) >= bundle["threshold"]).astype(int)
