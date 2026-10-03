"""Naive, no-ML baselines — what TabPFN-3.5 is judged against.

Both are a historical lookup table (station x weekend-flag x hour), fit on the
train pool only, zero "training" cost beyond a groupby. They are deliberately
not a tuned classical model (XGBoost/CatBoost) — the point of this repo's
benchmark (`tabpfn_lab/evaluate.py`) is "a historical-average dispatcher
heuristic vs. a foundation model with one `.fit()` call", which is the actual
choice an operator-tooling team faces under a hackathon deadline: hand-tune a
gradient-boosted model per task, or hand TabPFN-3.5 the feature table and get
a calibrated prediction back in seconds.
"""
from __future__ import annotations

import pandas as pd


def fit_classification_baseline(train_pool: pd.DataFrame, label_col: str = "overcrowded") -> pd.DataFrame:
    """Historical overcrowding rate per (station, weekend-flag, hour)."""
    return (train_pool.groupby(["station_name", "is_weekend", "hour"])[label_col].mean()
            .rename("baseline_probability").reset_index())


def predict_classification_baseline(rate_table: pd.DataFrame, rows: pd.DataFrame, fallback: float) -> pd.Series:
    merged = rows[["station_name", "is_weekend", "hour"]].merge(
        rate_table, on=["station_name", "is_weekend", "hour"], how="left")
    return merged["baseline_probability"].fillna(fallback)


def fit_regression_baseline(train_pool: pd.DataFrame, target_col: str = "passengers") -> pd.DataFrame:
    """Historical mean passenger count per (station, weekend-flag, hour)."""
    return (train_pool.groupby(["station_name", "is_weekend", "hour"])[target_col].mean()
            .rename("baseline_passengers").reset_index())


def predict_regression_baseline(mean_table: pd.DataFrame, rows: pd.DataFrame, fallback: float) -> pd.Series:
    merged = rows[["station_name", "is_weekend", "hour"]].merge(
        mean_table, on=["station_name", "is_weekend", "hour"], how="left")
    return merged["baseline_passengers"].fillna(fallback)
