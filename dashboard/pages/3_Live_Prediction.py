"""Ask the Berlin TabPFN-3.5 models directly — the same two models the MCP server / agent use."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tabpfn_lab.config import BERLIN_DATA_DIR, load_dotenv  # noqa: E402
from tabpfn_lab.datasets.berlin import build_feature_table, chronological_split, discover_dataset_dir  # noqa: E402
from tabpfn_lab.models import load_or_fit, predict_row  # noqa: E402

load_dotenv()
st.title("🔮 Live Prediction — Berlin U-Bahn")
st.caption("Pick a station and timestamp; both TabPFN-3.5 models (classifier + regressor) answer live.")


@st.cache_data(show_spinner="Loading feature table...")
def _load_table():
    folder = discover_dataset_dir(BERLIN_DATA_DIR)
    return build_feature_table(folder)


@st.cache_resource(show_spinner="Fitting TabPFN-3.5 (or loading cached fit)...")
def _load_models():
    table = _load_table()
    train_pool, _, cutoff = chronological_split(table)  # only train_pool + cutoff needed here
    return load_or_fit(train_pool, cutoff)


table = _load_table()
stations = sorted(table["station_name"].unique())
col1, col2 = st.columns(2)
default_station = "S+U Alexanderplatz Bhf (Berlin)"
station = col1.selectbox("Station", stations, index=stations.index(default_station) if default_station in stations else 0)
timestamps = sorted(table.loc[table["station_name"] == station, "timestamp"].unique())
ts = col2.select_slider("Timestamp", options=timestamps, value=timestamps[len(timestamps) // 2])

if st.button("Predict", type="primary"):
    row = table[(table["station_name"] == station) & (table["timestamp"] == ts)].iloc[0]
    models = _load_models()
    result = predict_row(models, row)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overcrowding risk", result["predicted_label"], f"{result['overcrowding_probability']:.0%} probability")
    c2.metric("Expected passengers", f"{result['predicted_passengers']:.0f}", f"actual: {int(row['passengers'])}")
    c3.metric("Latency (classify + regress)", f"{result['classify_latency_ms'] + result['regress_latency_ms']:.0f} ms")
    st.caption(f"Model: {result['model']}  ·  actual label: "
               f"{'overcrowded' if row['overcrowded'] == 1 else 'normal'}")
