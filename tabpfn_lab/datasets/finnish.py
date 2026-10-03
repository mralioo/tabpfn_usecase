"""Finnish railway operations dataset — VR (Finnish state railway) long-distance train stops
matched against FMI (Finnish Meteorological Institute) weather observations at the nearest
station. Data: `data/Finnish_Railway_Operations_data/` — 96 monthly parquet files, January 2018
through December 2025, ~2.75 GB, one row per (train run, station along the route). This is the
dataset behind the FI-TW paper (arXiv:2601.16592) — "An Open Train-Weather Dataset for Railway
Delay Analysis in Finland" — same construction (per-stop delay + matched weather), not fetched
from Kaggle here since the files were already available locally; see `data/SOURCE.md`.

Biggest of the three datasets by a wide margin (~500k rows/month x 96 months ≈ tens of millions
of rows total) and the only one with genuine multi-year span, so it's the one to reach for when
the question is "does this hold up over years / schema drift / a changing punctuality regime"
rather than one snapshot month. First exploration only for now — see `docs/MODULES.md` for the
planned classification+regression pairing (mirroring Berlin and Deutsche Bahn) once modeling
starts here.

Columns actually exist under original names with spaces, e.g. "Air temperature" — exploration
code accesses them as plain string keys. Build_feature_table below selects a small subset (not
the full 90-odd rolling-window weather columns) as a first-pass tabular design.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import FINNISH_DATA_DIR

DELAY_THRESHOLD_MIN = 5  # same convention as the FI-TW paper and the Deutsche Bahn loader
# differenceInMinutes is a real (not data-glitch) long tail — clipped for the regression task
# so a handful of multi-hour disruptions don't dominate MAE/RMSE. Chosen from the observed
# distribution (p99 is well under 150 for a typical month), not an arbitrary round number.
DELAY_CLIP = (-20, 180)

FEATURES = ["hour", "dow", "is_weekend", "month", "stationName", "trainType",
            "trainStopping", "Air temperature", "Wind speed", "Precipitation amount",
            "Snow depth", "Relative humidity", "Pressure (msl)", "Horizontal visibility"]
CATEGORICAL = ["stationName", "trainType"]


_MONTH_RE = re.compile(r"matched_data_flat_(\d{4})_(\d{2})\.parquet$")


def list_available_months() -> list[str]:
    """Every month present locally, as 'YYYY_MM' strings, sorted."""
    months = []
    for p in FINNISH_DATA_DIR.glob("matched_data_flat_*.parquet"):
        m = _MONTH_RE.search(p.name)
        if m:
            months.append(f"{m.group(1)}_{m.group(2)}")
    return sorted(months)


def _month_path(month: str) -> Path:
    path = FINNISH_DATA_DIR / f"matched_data_flat_{month}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no local file for month {month!r} — available: {list_available_months()}")
    return path


@lru_cache(maxsize=8)
def load_month_raw(month: str) -> pd.DataFrame:
    """One month, as stored (122 columns, every train-stop record including non-commercial
    pass-through points — trainCategory is 'Long-distance' only in every month checked so far,
    this release does not include commuter/S-Bahn-style trains despite 'S' appearing as a
    trainType code within long-distance services)."""
    df = pd.read_parquet(_month_path(month))
    df["scheduledTime"] = pd.to_datetime(df["scheduledTime"], utc=True).dt.tz_localize(None)
    df["actualTime"] = pd.to_datetime(df["actualTime"], utc=True, errors="coerce").dt.tz_localize(None)
    return df


@lru_cache(maxsize=1)
def load_stations() -> pd.DataFrame:
    """563 Finnish train stations/stopping points — longitude, latitude, passengerTraffic flag."""
    return pd.read_csv(FINNISH_DATA_DIR / "metadata_train_stations.csv")


@lru_cache(maxsize=1)
def load_weather_stations() -> pd.DataFrame:
    """204 FMI (Finnish Meteorological Institute) observation stations used as the weather source."""
    return pd.read_csv(FINNISH_DATA_DIR / "metadata_fmi_ems_stations.csv")


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["scheduledTime"].dt.hour
    df["dow"] = df["scheduledTime"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["scheduledTime"].dt.month
    df["trainStopping"] = df["trainStopping"].astype(int)
    return df


def build_feature_table(months: list[str] | None = None) -> pd.DataFrame:
    """Concatenate `months` (default: just the most recent available month — pass more for a
    bigger/longer-span table, but mind memory: ~500k rows and ~30MB per month). Filters to real
    commercial stops only (trainStopping, not cancelled) — pass-through waypoints and cancelled
    stops don't have a meaningful "how late was the passenger-facing arrival" answer.

    Two targets from the same `differenceInMinutes` column:
      - classification: `delayed` = 1 if differenceInMinutes >= 5
      - regression: `differenceInMinutes` itself, clipped to DELAY_CLIP
    """
    months = months or [list_available_months()[-1]]
    frames = [load_month_raw(m) for m in months]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    df = df[(df["trainStopping"]) & (~df["cancelled"]) & (~df["stop_cancelled"])]
    df = df.dropna(subset=["differenceInMinutes"] + [c for c in FEATURES if c not in ("hour", "dow", "is_weekend", "month")])
    df = _calendar_features(df)
    df["delayed"] = (df["differenceInMinutes"] >= DELAY_THRESHOLD_MIN).astype(int)

    keep = ["scheduledTime", "trainNumber"] + FEATURES + ["differenceInMinutes", "delayed"]
    return df[keep].reset_index(drop=True)


def load_classification_table(months: list[str] | None = None) -> pd.DataFrame:
    table = build_feature_table(months)
    return table[["scheduledTime"] + FEATURES + ["delayed"]]


def load_regression_table(months: list[str] | None = None) -> pd.DataFrame:
    table = build_feature_table(months)
    lo, hi = DELAY_CLIP
    table = table[(table["differenceInMinutes"] >= lo) & (table["differenceInMinutes"] <= hi)]
    return table[["scheduledTime"] + FEATURES + ["differenceInMinutes"]]


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str], reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Same label-encoding approach as the other two datasets' loaders."""
    out = df.copy()
    basis = reference if reference is not None else df
    for col in cat_cols:
        categories = pd.Categorical(basis[col]).categories
        out[col] = pd.Categorical(out[col], categories=categories).codes
    return out


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    dates = sorted(df["scheduledTime"].dt.date.unique())
    cutoff = dates[int(len(dates) * (1 - test_fraction))]
    train = df[df["scheduledTime"].dt.date < cutoff]
    test = df[df["scheduledTime"].dt.date >= cutoff]
    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff
