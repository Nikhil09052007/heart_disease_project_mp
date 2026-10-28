"""Dataset loading with three fallbacks, plus schema validation and target encoding.

Order tried:
  1. Local cached CSV  (data/heart_disease_raw.csv)
  2. ucimlrepo package (UCI dataset id 45)
  3. Direct download of processed.cleveland.data from the UCI archive
If all fail, a clear error explains how to place the CSV manually.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402


def _from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=cfg.UCI_DATASET_ID)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    # The targets frame has a single column ('num'); be defensive about its name.
    if cfg.SOURCE_TARGET_COL not in y.columns:
        y = y.rename(columns={y.columns[0]: cfg.SOURCE_TARGET_COL})
    return pd.concat([X, y[[cfg.SOURCE_TARGET_COL]]], axis=1)


def _from_url() -> pd.DataFrame:
    return pd.read_csv(
        cfg.UCI_FALLBACK_URL, header=None, names=cfg.RAW_COLUMNS, na_values="?"
    )


def validate_schema(df: pd.DataFrame) -> None:
    """Raise a helpful error if expected columns are missing."""
    needed = cfg.FEATURE_COLUMNS + [cfg.SOURCE_TARGET_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}. Expected: {needed}"
        )


def load_raw_data(refresh: bool = False) -> pd.DataFrame:
    """Return the raw dataset (13 features + 'num'), caching it to data/."""
    if cfg.RAW_DATA_PATH.exists() and not refresh:
        df = pd.read_csv(cfg.RAW_DATA_PATH, na_values=["?", ""])
        validate_schema(df)
        print(f"[data] Loaded cached dataset: {cfg.RAW_DATA_PATH} {df.shape}")
        return df

    errors = []
    df = None
    for name, fn in (("ucimlrepo", _from_ucimlrepo), ("UCI URL", _from_url)):
        try:
            print(f"[data] Trying {name} ...")
            df = fn()
            validate_schema(df)
            print(f"[data] Downloaded via {name}: {df.shape}")
            break
        except Exception as exc:  # network, package or schema problems
            errors.append(f"{name}: {exc}")
            df = None

    if df is None:
        raise RuntimeError(
            "Could not load the dataset.\n  " + "\n  ".join(errors) + "\n"
            f"Manual fix: download processed.cleveland.data from\n  {cfg.UCI_FALLBACK_URL}\n"
            f"and save it as CSV with header {cfg.RAW_COLUMNS} to {cfg.RAW_DATA_PATH}"
        )

    df = df[cfg.FEATURE_COLUMNS + [cfg.SOURCE_TARGET_COL]].apply(pd.to_numeric, errors="coerce")
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.RAW_DATA_PATH, index=False)
    print(f"[data] Cached to {cfg.RAW_DATA_PATH}")
    return df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary 'target': 0 if num == 0, 1 if num > 0."""
    out = df.copy()
    out[cfg.TARGET_COL] = (out[cfg.SOURCE_TARGET_COL] > 0).astype(int)
    return out


def load_dataset(refresh: bool = False) -> pd.DataFrame:
    """Raw data + binary target, ready for EDA/training."""
    df = load_raw_data(refresh=refresh)
    df = df.dropna(subset=[cfg.SOURCE_TARGET_COL])  # target must exist; features may be NaN
    return add_binary_target(df)


if __name__ == "__main__":
    data = load_dataset()
    print(data.head())
    print(data[cfg.TARGET_COL].value_counts())
