"""FastMCP server exposing the Berlin U-Bahn feature table and the two
TabPFN-3.5 inference models as MCP tools.

This is the hand-off point the agent harness (`agent/harness.py`) calls into:
a question about crowd risk routes to `predict_overcrowding_risk`
(classification), a question about expected demand routes to
`predict_expected_flow` (regression) — same feature row, two specialised
TabPFN-3.5 calls, picked by use case, same pattern an agent would use to hand
off to a weather tool or a calculator.

Run standalone (stdio transport):
    python mcp_server/server.py
"""
from __future__ import annotations

import difflib
import sys
import threading
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tabpfn_lab.config import BERLIN_DATA_DIR, load_dotenv  # noqa: E402
from tabpfn_lab.datasets.berlin import build_feature_table, chronological_split, discover_dataset_dir, load_stations  # noqa: E402
from tabpfn_lab.models import FittedModels, load_or_fit, predict_row  # noqa: E402

load_dotenv()

mcp = FastMCP(
    "tabpfn-ubahn-predict",
    instructions=(
        "TabPFN-3.5 prediction tools over the Berlin U-Bahn hackathon dataset. "
        "Call resolve_station first if you're not sure of the exact station_name. "
        "Call describe_dataset to see the covered date range before predicting — "
        "timestamps must be exact 15-minute marks inside that window. Pick the "
        "tool by use case: predict_overcrowding_risk for a crowd-safety / "
        "early-warning question ('will this station be overcrowded'), "
        "predict_expected_flow for an operational-planning question "
        "('how many passengers are expected')."
    ),
)

_FOLDER = discover_dataset_dir(BERLIN_DATA_DIR)
_table_lock, _model_lock = threading.Lock(), threading.Lock()
_feature_table_cache: pd.DataFrame | None = None
_models_cache: FittedModels | None = None


def _get_feature_table() -> pd.DataFrame:
    global _feature_table_cache
    with _table_lock:
        if _feature_table_cache is None:
            _feature_table_cache = build_feature_table(_FOLDER)
        return _feature_table_cache


def _get_models() -> FittedModels:
    global _models_cache
    with _model_lock:
        if _models_cache is None:
            table = _get_feature_table()
            train_pool, _test_pool, cutoff = chronological_split(table)
            _models_cache = load_or_fit(train_pool, cutoff)
        return _models_cache


def _all_station_names() -> list[str]:
    return sorted(load_stations(_FOLDER)["station_name"].unique())


def _lookup_row(station_name: str, timestamp: str) -> tuple[pd.Series | None, str | None]:
    table = _get_feature_table()
    ts = pd.Timestamp(timestamp)
    match = table[(table["station_name"] == station_name) & (table["timestamp"] == ts)]
    if match.empty:
        return None, (
            f"No row for station_name='{station_name}' at timestamp='{timestamp}'. "
            "Timestamps must be an exact 15-minute mark inside the dataset's coverage "
            "window (call describe_dataset for the range); this only replays known "
            "timestamps, it does not forecast beyond the loaded data."
        )
    return match.iloc[0], None


@mcp.tool
def describe_dataset() -> dict:
    """Dataset coverage (date range, station/line counts) and what the two predict_*
    tools need as input. Call this first if unsure whether a question is in scope."""
    stations = load_stations(_FOLDER)
    table = _get_feature_table()
    return {
        "dataset_folder": _FOLDER,
        "coverage_start": str(table["timestamp"].min()),
        "coverage_end": str(table["timestamp"].max()),
        "n_stations": int(stations["station_name"].nunique()),
        "n_lines": int(stations["u_bahn_lines"].str.split(",").explode().str.strip().nunique()),
        "tools": ["resolve_station", "predict_overcrowding_risk (classification)",
                  "predict_expected_flow (regression)"],
    }


@mcp.tool
def resolve_station(query: str, max_results: int = 5) -> list[dict]:
    """Fuzzy-resolve a user-typed station reference (e.g. 'Rudow', 'Alexanderplatz') to the
    exact station_name string the predict_* tools require."""
    names = _all_station_names()
    query_lower = query.strip().lower()
    substring_hits = [n for n in names if query_lower in n.lower()]
    fuzzy_hits = difflib.get_close_matches(query, names, n=max_results, cutoff=0.4)
    ordered = list(dict.fromkeys(substring_hits + fuzzy_hits))[:max_results]
    if not ordered:
        return [{"station_name": None, "note": f"No station resembling '{query}' found."}]
    return [{"station_name": n} for n in ordered]


@mcp.tool
def predict_overcrowding_risk(station_name: str, timestamp: str) -> dict:
    """TabPFN-3.5 classification: probability this station's flow at this 15-minute
    timestamp is at/above its own historical 90th percentile — a crowd-pressure
    early-warning signal an operator would act on (add staff, hold a train) before
    the platform fills up. `timestamp` example: '2026-07-15 08:00:00'. Call
    resolve_station first if unsure of the exact station_name."""
    row, err = _lookup_row(station_name, timestamp)
    if err:
        return {"error": err}
    models = _get_models()
    result = predict_row(models, row)
    return {
        "station_name": station_name, "timestamp": timestamp,
        "overcrowding_probability": result["overcrowding_probability"],
        "predicted_label": result["predicted_label"],
        "actual_label": "overcrowded" if int(row["overcrowded"]) == 1 else "normal",
        "actual_passengers": int(row["passengers"]),
        "latency_ms": result["classify_latency_ms"],
        "model": result["model"],
    }


@mcp.tool
def predict_expected_flow(station_name: str, timestamp: str) -> dict:
    """TabPFN-3.5 regression: expected passenger count at this station and timestamp,
    given time-of-day/weekday, weather, event and closure context — for operational
    planning (staffing, train frequency). Same input contract as
    predict_overcrowding_risk."""
    row, err = _lookup_row(station_name, timestamp)
    if err:
        return {"error": err}
    models = _get_models()
    result = predict_row(models, row)
    actual = int(row["passengers"])
    return {
        "station_name": station_name, "timestamp": timestamp,
        "predicted_passengers": result["predicted_passengers"],
        "actual_passengers": actual,
        "absolute_error": round(abs(result["predicted_passengers"] - actual), 1),
        "latency_ms": result["regress_latency_ms"],
        "model": result["model"],
    }


if __name__ == "__main__":
    mcp.run()
