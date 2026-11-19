"""Metadata cleaning, regression-target construction and leakage-free splits.

The dataset photographs each avocado ("sample") once per day, from two
sides (`_a_` / `_b_`), under one of three storage conditions
(T10 = 10 °C, T20 = 20 °C, Tam = ambient). Each photo is labeled with a
ripening index from 1 (underripe) to 5 (overripe).

Regression target: days_until_overripe
    For every sample, find the first day it reached stage 5. Each photo's
    target is that day minus the photo's day, clipped at -2. Two
    alternatives (days until ripe, until stage 4) were evaluated during
    research and rejected because most values collapse to 0, which teaches
    the model to always predict 0. Samples that never reach stage 5 have no
    target and are dropped from training.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

# Original spreadsheet column -> normalized column
RAW_COLUMNS = {
    "File Name": "file_name",
    "Time Stamp": "timestamp",
    "Storage Group": "storage_group",
    "Sample": "sample",
    "Day of Experiment": "day",
    "Ripening Index Classification": "stage",
}

TARGET_COLUMN = "days_until_overripe"
TARGET_CLIP_MIN = -2


def load_raw_metadata(source: str | Path | IO[bytes]) -> pd.DataFrame:
    """Read the dataset spreadsheet and return a clean, typed DataFrame."""
    df = pd.read_excel(source)
    df.columns = df.columns.str.strip()
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Spreadsheet is missing columns: {sorted(missing)}")

    df = df.rename(columns=RAW_COLUMNS)[list(RAW_COLUMNS.values())]
    for column in ("file_name", "storage_group", "sample"):
        df[column] = df[column].astype(str).str.strip()
    df["day"] = pd.to_numeric(df["day"], errors="coerce")
    df["stage"] = pd.to_numeric(df["stage"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.dropna(subset=["file_name", "storage_group", "sample", "day", "stage"])
    df = df[df["stage"].between(1, 5)].copy()
    df["day"] = df["day"].astype(int)
    df["stage"] = df["stage"].astype(int)
    df["group_id"] = df["storage_group"] + "_" + df["sample"]
    return df.reset_index(drop=True)


def add_regression_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the `days_until_overripe` column (NaN for samples never reaching stage 5)."""
    first_overripe_day = df[df["stage"] >= 5].groupby("group_id")["day"].min()
    result = df.copy()
    result[TARGET_COLUMN] = (result["group_id"].map(first_overripe_day) - result["day"]).clip(
        lower=TARGET_CLIP_MIN
    )
    return result


def split_by_group(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/val/test split at the *fruit* level.

    Every photo of the same avocado ends up in the same split, so the model
    is never evaluated on a fruit it has already seen (no data leakage).
    `test_size` and `val_size` are fractions of the whole dataset.
    """
    from sklearn.model_selection import GroupShuffleSplit  # training-only dependency

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=df["group_id"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval["group_id"]))
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]

    groups = [set(part["group_id"]) for part in (train, val, test)]
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
