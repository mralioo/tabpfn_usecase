"""A large, real, same-topic railway dataset: Deutsche Bahn (German national rail) per-stop
delay records, for the bigger/harder benchmark this repo's README explains the need for (the
Berlin U-Bahn data alone is too periodic to prove anything against a baseline — see
`tabpfn_lab/baselines.py`).

Source: **piebro/deutsche-bahn-data** on Hugging Face
(https://huggingface.co/datasets/piebro/deutsche-bahn-data, project page
https://github.com/piebro/deutsche-bahn-data), CC BY 4.0, built from Deutsche Bahn's own public
timetable/delay feeds. One month of real German train stops (~2M rows, 107 major stations, every
train type from S-Bahn to ICE) — genuinely railway-topic, same country as the Berlin U-Bahn data,
and big enough (unlike Berlin) for a classification+regression accuracy contest against XGBoost
to mean something.

Fetched via `huggingface_hub.hf_hub_download` — no auth for a public dataset. Cached twice: once
by `huggingface_hub` itself (`~/.cache/huggingface`), and a copy mirrored into `data/DB_data/`
(gitignored) the first time a month is loaded, so this dataset sits next to the other two under
`data/` like a normal local dataset after the first run — still a single reproducible function
call for a third party on a clean machine.

Two targets, same `delay_in_min` column, mirroring the Berlin demo's classification+regression
pair and the FI-TW paper's own convention (a 5-minute threshold for "delayed"):
  - classification: `delayed` = 1 if delay_in_min >= 5
  - regression: `delay_in_min` itself (outlier-clipped — see `load_regression_table`)
"""
from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import DB_DATA_DIR

REPO_ID = "piebro/deutsche-bahn-data"
DEFAULT_MONTH = "2025-06"  # one month ≈ 1.95M rows; change to pull a different / additional month

FEATURES = ["hour", "dow", "is_weekend", "station_name", "train_type",
            "train_line_station_num", "is_additional_stop", "is_replacement_train"]
CATEGORICAL = ["station_name", "train_type"]
DELAY_THRESHOLD_MIN = 5

# delay_in_min has a handful of data-glitch outliers (a few rows past -1000 or +600 minutes);
# clipped to this range for the regression task so a handful of bad rows don't dominate MAE/RMSE.
DELAY_CLIP = (-15, 120)


def _month_path(month: str = DEFAULT_MONTH) -> str:
    local = DB_DATA_DIR / f"data-{month}.parquet"
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                  filename=f"monthly_processed_data/data-{month}.parquet")
    DB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(downloaded, local)
    return str(local)


@lru_cache(maxsize=4)
def load_raw(month: str = DEFAULT_MONTH) -> pd.DataFrame:
    df = pd.read_parquet(_month_path(month))
    df = df.dropna(subset=["delay_in_min", "arrival_planned_time"]).reset_index(drop=True)
    df["hour"] = df["arrival_planned_time"].dt.hour
    df["dow"] = df["arrival_planned_time"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_additional_stop"] = df["is_additional_stop"].astype(int)
    df["is_replacement_train"] = df["is_replacement_train"].astype(int)
    return df


def load_classification_table(month: str = DEFAULT_MONTH) -> pd.DataFrame:
    df = load_raw(month)
    df = df.copy()
    df["delayed"] = (df["delay_in_min"] >= DELAY_THRESHOLD_MIN).astype(int)
    return df[["arrival_planned_time"] + FEATURES + ["delayed"]]


def load_regression_table(month: str = DEFAULT_MONTH) -> pd.DataFrame:
    df = load_raw(month)
    lo, hi = DELAY_CLIP
    df = df[(df["delay_in_min"] >= lo) & (df["delay_in_min"] <= hi)].copy()
    return df[["arrival_planned_time"] + FEATURES + ["delay_in_min"]]


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str], reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same label-encoding approach as tabpfn_lab.datasets.berlin — TabPFN's
    categorical_features_indices and XGBoost are fed the SAME encoded matrix, for a fair,
    apples-to-apples comparison."""
    out = df.copy()
    basis = reference if reference is not None else df
    for col in cat_cols:
        categories = pd.Categorical(basis[col]).categories
        out[col] = pd.Categorical(out[col], categories=categories).codes
    return out


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Real timestamps this time (unlike the OpenML flight snapshots this replaced) — split by
    date, last `test_fraction` of days held out, no shuffling."""
    dates = sorted(df["arrival_planned_time"].dt.date.unique())
    cutoff = dates[int(len(dates) * (1 - test_fraction))]
    train = df[df["arrival_planned_time"].dt.date < cutoff]
    test = df[df["arrival_planned_time"].dt.date >= cutoff]
    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff
