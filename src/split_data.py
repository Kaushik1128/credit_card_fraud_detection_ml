"""Materialize train/val/test splits + the fitted preprocessor.

Run from the project root:
    .venv/Scripts/python.exe -m src.split_data
"""
from pathlib import Path

from src.data import (
    TARGET_COLUMN,
    load_raw,
    make_preprocessor,
    make_splits,
    save_preprocessor,
    save_splits,
    split_xy,
)

PROCESSED_DIR = Path("data/processed")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


def main() -> None:
    print("Loading raw...")
    df = load_raw()
    print(f"  shape: {df.shape}")

    print("\nSplitting (stratified 60/20/20)...")
    splits = make_splits(df)
    for name, frame in splits.items():
        n = len(frame)
        n_fraud = int(frame[TARGET_COLUMN].sum())
        print(f"  {name:>5}: {n:>7,} rows, {n_fraud:>3} fraud ({n_fraud / n:.4%})")

    print(f"\nSaving splits to {PROCESSED_DIR}/ ...")
    save_splits(splits, PROCESSED_DIR)

    # Fit on TRAIN only — fitting on val/test would leak target distribution.
    print("\nFitting preprocessor on TRAIN only (no leakage)...")
    pre = make_preprocessor()
    X_train, _ = split_xy(splits["train"])
    pre.fit(X_train)

    print(f"Saving preprocessor to {PREPROCESSOR_PATH}...")
    save_preprocessor(pre, PREPROCESSOR_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
