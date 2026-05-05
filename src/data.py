"""Data loading, splitting, preprocessing primitives.

Pure functions — no side effects beyond what the caller orchestrates.
Imported by training scripts, the FastAPI service, and tests.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Single source of truth for everything stochastic in the pipeline.
RANDOM_STATE = 42

TARGET_COLUMN = "Class"
SCALED_COLUMNS = ["Time", "Amount"]
PASSTHROUGH_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = SCALED_COLUMNS + PASSTHROUGH_COLUMNS  # 30 features


def load_raw(path: str | Path = "data/raw/creditcard.csv") -> pd.DataFrame:
    """Load the raw Kaggle creditcard.csv and verify expected columns."""
    df = pd.read_csv(path)
    expected = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df


def make_splits(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Stratified split into train/val/test (default 60/20/20).

    Stratification preserves the 0.17% fraud rate in every split — without it,
    a random split could land an unlucky test set with zero positives.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET_COLUMN],
        random_state=random_state,
    )
    # val_size is a fraction of the *original*; rescale for the second split.
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val[TARGET_COLUMN],
        random_state=random_state,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (30 cols) from target."""
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def make_preprocessor() -> ColumnTransformer:
    """Build an UNFITTED preprocessor: StandardScaler on Time/Amount, passthrough V1-V28.

    V1-V28 are PCA components from the dataset authors and already on roughly
    unit scale, so re-scaling them adds noise without benefit.
    """
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), SCALED_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)


def load_splits(in_dir: str | Path = "data/processed") -> dict[str, pd.DataFrame]:
    in_p = Path(in_dir)
    return {name: pd.read_parquet(in_p / f"{name}.parquet") for name in ("train", "val", "test")}


def save_preprocessor(preprocessor, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str | Path):
    return joblib.load(path)
