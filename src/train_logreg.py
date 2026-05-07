"""Train baseline logistic regression on the fraud dataset.

Run from the project root:
    .venv/Scripts/python.exe -m src.train_logreg
"""
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import RANDOM_STATE, load_splits, make_preprocessor, split_xy
from src.metrics import (
    evaluate,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    print_report,
    save_metrics,
)

MODEL_PATH = Path("models/logreg_baseline.joblib")
METRICS_PATH = Path("models/logreg_baseline_metrics.json")
FIG_DIR = Path("notebooks/figures")
THRESHOLD = 0.5  # default; we'll tune deliberately in Phase 2.6


def main() -> None:
    print("Loading splits...")
    splits = load_splits()
    X_train, y_train = split_xy(splits["train"])
    X_val, y_val = split_xy(splits["val"])
    print(f"  train = {len(X_train):>7,}   fraud = {int(y_train.sum())}")
    print(f"  val   = {len(X_val):>7,}   fraud = {int(y_val.sum())}")

    print("\nBuilding pipeline (preprocessor + LogisticRegression)...")
    pipeline = Pipeline([
        ("pre", make_preprocessor()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])

    print("Fitting on train (under a minute)...")
    pipeline.fit(X_train, y_train)

    print("\nScoring on validation set...")
    metrics = evaluate(pipeline, X_val, y_val, threshold=THRESHOLD)
    print_report(metrics, title="Logistic regression baseline (validation)")

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
        y_val, y_pred, FIG_DIR / "05_logreg_confusion.png",
        title="LogReg baseline — confusion matrix (val)",
    )
    plot_pr_curve(y_val, y_score, FIG_DIR / "06_logreg_pr_curve.png", label="LogReg")
    plot_roc_curve(y_val, y_score, FIG_DIR / "07_logreg_roc_curve.png", label="LogReg")

    print("\nDone.")


if __name__ == "__main__":
    main()
