"""Train XGBoost contender on the fraud dataset.

Run from the project root:
    .venv/Scripts/python.exe -m src.train_xgboost
"""
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import RANDOM_STATE, load_splits, make_preprocessor, split_xy
from src.metrics import (
    evaluate,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    print_report,
    save_metrics,
)

MODEL_PATH = Path("models/xgboost_baseline.joblib")
METRICS_PATH = Path("models/xgboost_baseline_metrics.json")
FIG_DIR = Path("notebooks/figures")
THRESHOLD = 0.5  # tuned in Phase 2.6


def main() -> None:
    print("Loading splits...")
    splits = load_splits()
    X_train, y_train = split_xy(splits["train"])
    X_val, y_val = split_xy(splits["val"])
    print(f"  train = {len(X_train):>7,}   fraud = {int(y_train.sum())}")
    print(f"  val   = {len(X_val):>7,}   fraud = {int(y_val.sum())}")

    # Compute scale_pos_weight from TRAIN only — the val/test rates would leak.
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"\nscale_pos_weight = {n_neg:,} / {n_pos} = {scale_pos_weight:.1f}")

    print("\nBuilding pipeline (preprocessor + XGBClassifier)...")
    pipeline = Pipeline([
        ("pre", make_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("Fitting on train (this takes ~30-60s)...")
    pipeline.fit(X_train, y_train)

    print("\nScoring on validation set...")
    metrics = evaluate(pipeline, X_val, y_val, threshold=THRESHOLD)
    print_report(metrics, title="XGBoost (validation)")

    print(f"\nSaving model      -> {MODEL_PATH}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Saving metrics    -> {METRICS_PATH}")
    save_metrics(metrics, METRICS_PATH)

    print("Saving plots      -> notebooks/figures/")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y_score = pipeline.predict_proba(X_val)[:, 1]
    y_pred = (y_score >= THRESHOLD).astype(int)
    plot_confusion_matrix(
        y_val, y_pred, FIG_DIR / "08_xgb_confusion.png",
        title="XGBoost — confusion matrix (val)",
    )
    plot_pr_curve(y_val, y_score, FIG_DIR / "09_xgb_pr_curve.png", label="XGBoost")
    plot_roc_curve(y_val, y_score, FIG_DIR / "10_xgb_roc_curve.png", label="XGBoost")

    print("\nDone.")


if __name__ == "__main__":
    main()
