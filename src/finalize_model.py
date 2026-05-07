"""Finalize the fraud model: score on test, bundle artifact, save.

Bundles the trained XGBoost pipeline + chosen threshold + feature spec +
metadata into models/fraud_model.joblib, ready for the FastAPI service
in Phase 3.

Run from project root:
    .venv/Scripts/python.exe -m src.finalize_model
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.data import FEATURE_COLUMNS, load_splits, split_xy
from src.metrics import (
    evaluate,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    print_report,
    save_metrics,
)
from src.model import predict, predict_proba

XGBOOST_PATH = Path("models/xgboost_baseline.joblib")
THRESHOLD_PATH = Path("models/xgboost_threshold_analysis.json")
VAL_METRICS_PATH = Path("models/xgboost_baseline_metrics.json")
BUNDLE_PATH = Path("models/fraud_model.joblib")
TEST_METRICS_PATH = Path("models/fraud_model_test_metrics.json")
FIG_DIR = Path("notebooks/figures")


def main() -> None:
    print("Assembling final model bundle...")
    pipeline = joblib.load(XGBOOST_PATH)

    with open(THRESHOLD_PATH) as f:
        threshold_analysis = json.load(f)
    threshold = threshold_analysis["candidates"]["cost_min"]["threshold"]
    print(f"  pipeline   <- {XGBOOST_PATH}")
    print(f"  threshold  <- {threshold:.4f} (cost-min from Phase 2.6)")

    with open(VAL_METRICS_PATH) as f:
        val_metrics = json.load(f)

    print("\nLoading splits — first and only touch of the test set...")
    splits = load_splits()
    X_train, y_train = split_xy(splits["train"])
    X_test, y_test = split_xy(splits["test"])
    print(f"  test = {len(X_test):,}  fraud = {int(y_test.sum())}")

    print(f"\n=== Test set: final, honest score @ threshold {threshold:.4f} ===")
    test_metrics = evaluate(pipeline, X_test, y_test, threshold=threshold)
    print_report(test_metrics, title="Final XGBoost (test set)")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    metadata = {
        "model_name": "xgboost_v1",
        "model_type": "sklearn.Pipeline(StandardScaler + XGBClassifier)",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training": {
            "n_train_samples": int(len(X_train)),
            "n_train_fraud": n_pos,
            "scale_pos_weight": n_neg / n_pos,
            "imbalance_strategy": "scale_pos_weight (validated in Phase 2.5 bake-off)",
            "threshold_strategy": "cost-min @ FN=$100 / FP=$10 (Phase 2.6 sweep)",
        },
        "validation_metrics_at_t0.5": val_metrics,
        "test_metrics_at_chosen_threshold": test_metrics,
    }

    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "feature_columns": FEATURE_COLUMNS,
        "metadata": metadata,
    }
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)
    print(f"\nSaved bundle  -> {BUNDLE_PATH}")

    save_metrics(test_metrics, TEST_METRICS_PATH)
    print(f"Saved metrics -> {TEST_METRICS_PATH}")

    print("Saving test-set plots...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    plot_confusion_matrix(
        y_test, y_pred, FIG_DIR / "14_final_test_confusion.png",
        title=f"Final XGBoost — confusion matrix (test, t={threshold:.3f})",
    )
    plot_pr_curve(y_test, y_score, FIG_DIR / "15_final_test_pr_curve.png", label="Final XGBoost")
    plot_roc_curve(y_test, y_score, FIG_DIR / "16_final_test_roc_curve.png", label="Final XGBoost")

    print("\nSmoke test: load saved bundle, predict on first 5 test rows...")
    loaded = joblib.load(BUNDLE_PATH)
    sample = X_test.head(5)
    probs = predict_proba(loaded, sample)
    preds = predict(loaded, sample)
    print(f"  Bundle keys:        {list(loaded.keys())}")
    print(f"  Threshold:          {loaded['threshold']}")
    print(f"  Feature columns:    {len(loaded['feature_columns'])} fields, first = {loaded['feature_columns'][:3]}")
    print(f"  P(fraud) per row:   {probs.round(4).tolist()}")
    print(f"  Predictions (0/1):  {preds.tolist()}")
    print(f"  Actual labels:      {y_test.head(5).tolist()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
