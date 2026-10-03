"""Smoke tests for the data/feature pipelines — no TabPFN/LLM API calls (no token needed), so
these run in CI / before you've set up `.env`. They do read local data files, so they need
`data/Berlin_Ubahn_Alstom_data/` and `data/Finnish_Railway_Operations_data/` present."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabpfn_lab.config import BERLIN_DATA_DIR
from tabpfn_lab.datasets.berlin import (
    FEATURE_COLUMNS,
    build_feature_table,
    chronological_split,
    discover_dataset_dir,
    load_stations,
)


def test_data_dir_discovered():
    folder = discover_dataset_dir(BERLIN_DATA_DIR)
    assert Path(folder).exists()


def test_stations_load():
    folder = discover_dataset_dir(BERLIN_DATA_DIR)
    stations = load_stations(folder)
    assert len(stations) > 0
    assert "station_name" in stations.columns


def test_feature_table_has_expected_columns():
    folder = discover_dataset_dir(BERLIN_DATA_DIR)
    table = build_feature_table(folder)
    assert len(table) > 0
    for col in FEATURE_COLUMNS + ["overcrowded", "passengers"]:
        assert col in table.columns
    assert table[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_chronological_split_no_leakage():
    folder = discover_dataset_dir(BERLIN_DATA_DIR)
    table = build_feature_table(folder)
    train, test, cutoff = chronological_split(table)
    assert train["timestamp"].dt.date.max() < cutoff
    assert test["timestamp"].dt.date.min() >= cutoff


def test_finnish_months_discovered():
    from tabpfn_lab.datasets.finnish import list_available_months

    months = list_available_months()
    assert len(months) > 0
    assert all(len(m) == 7 and m[4] == "_" for m in months)  # 'YYYY_MM'


def test_finnish_feature_table_has_expected_columns():
    from tabpfn_lab.datasets.finnish import FEATURES, build_feature_table, list_available_months

    table = build_feature_table([list_available_months()[-1]])
    assert len(table) > 0
    for col in FEATURES + ["differenceInMinutes", "delayed"]:
        assert col in table.columns
