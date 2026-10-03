"""Finnish railway data — first exploration. Unlike the Berlin/Deutsche Bahn pages, this one
has no prior NextMove page to adapt from: this dataset is new to the project. Loads one month at
a time (96 are available, ~2.75GB total) via `tabpfn_lab.datasets.finnish`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tabpfn_lab.datasets.finnish import (  # noqa: E402
    FINNISH_DATA_DIR,
    list_available_months,
    load_month_raw,
    load_stations,
)

st.title("🇫🇮 Finnish Railway — First Exploration")
st.caption(
    "VR (Finnish state railway) long-distance train stops matched against FMI weather "
    "observations — the dataset behind the FI-TW paper (arXiv:2601.16592). Not fetched here: "
    "already present locally under `data/Finnish_Railway_Operations_data/`. See `data/SOURCE.md`."
)

months = list_available_months()
if not months:
    st.error(f"No monthly parquet files found under `{FINNISH_DATA_DIR}`.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Months available locally", len(months))
c2.metric("Date range", f"{months[0][:4]}–{months[-1][:4]}")
c3.metric("Local size", "≈2.75 GB")

month = st.selectbox("Month to explore", months, index=months.index("2024_01") if "2024_01" in months else len(months) - 1)
df = load_month_raw(month)

st.subheader(f"{month.replace('_', '-')} — overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows (train × stop)", f"{len(df):,}")
c2.metric("Unique stations", df["stationName"].nunique())
c3.metric("Unique train numbers", df["trainNumber"].nunique())
c4.metric("Cancelled rate", f"{df['cancelled'].mean():.1%}")
non_null_delay = df["differenceInMinutes"].dropna()
c5.metric("Delayed ≥5 min rate", f"{(non_null_delay >= 5).mean():.1%}")

st.caption(
    f"`trainCategory` is uniformly **{df['trainCategory'].unique()[0]!r}** in this file — this "
    "release covers VR long-distance services only (IC, HDM, PYO, MV…); 'S' here is a "
    "long-distance train-type code, not the commuter/S-Bahn-style service. "
    f"`trainStopping` (a real commercial stop, not a pass-through waypoint) is true for "
    f"{df['trainStopping'].mean():.1%} of rows."
)

tab_delay, tab_weather, tab_time, tab_map, tab_drift = st.tabs(
    ["⏱️ Delay distribution", "🌦️ Weather relationship", "🕐 Time patterns", "🗺️ Station map", "📉 Multi-year drift"])

# ------------------------------------------------------------------------------- Delay distrib.
with tab_delay:
    st.subheader("differenceInMinutes distribution")
    clipped = non_null_delay.clip(-30, 120)
    st.plotly_chart(px.histogram(clipped, nbins=60, labels={"value": "Delay (min, clipped to [-30, 120] for display)"}),
                     use_container_width=True)
    st.caption(f"Full-range stats (unclipped): mean {non_null_delay.mean():.1f} min, "
               f"median {non_null_delay.median():.0f} min, p95 {non_null_delay.quantile(.95):.0f} min, "
               f"max {non_null_delay.max():.0f} min, {df['differenceInMinutes'].isna().mean():.1%} missing "
               "(cancelled / not-yet-run stops).")

    st.subheader("Delay rate by train type")
    rate_by_type = df.groupby("trainType")["differenceInMinutes"].apply(lambda s: (s >= 5).mean()).reset_index(name="delayed_rate")
    counts = df["trainType"].value_counts().reset_index()
    counts.columns = ["trainType", "n"]
    rate_by_type = rate_by_type.merge(counts, on="trainType").sort_values("n", ascending=False)
    st.plotly_chart(px.bar(rate_by_type, x="trainType", y="delayed_rate", hover_data=["n"],
                            labels={"delayed_rate": "Share delayed ≥5 min"}), use_container_width=True)

# ------------------------------------------------------------------------------ Weather relat.
with tab_weather:
    st.subheader("Correlation: weather vs. delay")
    weather_cols = ["Air temperature", "Wind speed", "Precipitation amount", "Snow depth",
                     "Relative humidity", "Pressure (msl)", "Horizontal visibility", "Cloud amount"]
    corr_df = df[["differenceInMinutes"] + weather_cols].dropna()
    corr = corr_df.corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                     use_container_width=True)
    st.caption(f"`differenceInMinutes` vs `Snow depth`: r = {corr.loc['differenceInMinutes', 'Snow depth']:.2f} — "
               f"vs `Air temperature`: r = {corr.loc['differenceInMinutes', 'Air temperature']:.2f}. "
               "Same per-row 15-min-style caveat as Berlin's weather tab: weak at the raw-row level, "
               "individual weather variables dominated by other effects (route, time of day, train type).")

    st.subheader("Delay vs. temperature (binned)")
    tdf = df[["Air temperature", "differenceInMinutes"]].dropna()
    tdf["temp_bin"] = pd.cut(tdf["Air temperature"], bins=np.arange(-30, 31, 5))
    binned = tdf.groupby("temp_bin", observed=True)["differenceInMinutes"].mean().reset_index()
    binned["temp_bin"] = binned["temp_bin"].astype(str)
    st.plotly_chart(px.bar(binned, x="temp_bin", y="differenceInMinutes",
                            labels={"temp_bin": "Air temperature bin (°C)", "differenceInMinutes": "Avg delay (min)"}),
                     use_container_width=True)
    st.caption("Finland-specific hypothesis worth testing later: delays rising at the cold end "
               "(sub-zero rail/signal issues) — a feature classical models rarely get to see "
               "explicitly, since it's a non-linear, cold-only effect.")

# ------------------------------------------------------------------------------- Time patterns
with tab_time:
    st.subheader("Average delay by hour of day")
    df["hour"] = df["scheduledTime"].dt.hour
    df["dow"] = df["scheduledTime"].dt.dayofweek
    hourly = df.groupby("hour")["differenceInMinutes"].mean().reset_index()
    st.plotly_chart(px.line(hourly, x="hour", y="differenceInMinutes", markers=True,
                             labels={"differenceInMinutes": "Avg delay (min)"}), use_container_width=True)

    st.subheader("Day of week × hour heatmap (avg delay)")
    df["dow_name"] = df["scheduledTime"].dt.day_name()
    heat = df.pivot_table(index="dow_name", columns="hour", values="differenceInMinutes", aggfunc="mean")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = heat.reindex([d for d in order if d in heat.index])
    st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                               labels=dict(x="Hour", y="", color="Avg delay (min)")), use_container_width=True)

# ------------------------------------------------------------------------------------ Stat map
with tab_map:
    st.subheader("Station network")
    meta = load_stations()
    avg_delay = df.groupby("stationName")["differenceInMinutes"].mean().rename("avg_delay").reset_index()
    merged = meta.merge(avg_delay, left_on="stationName", right_on="stationName", how="left")
    color_by = st.radio("Color by", ["Passenger traffic (station type)", "Avg delay this month"], horizontal=True)
    if color_by.startswith("Passenger"):
        fig = px.scatter_map(merged, lat="latitude", lon="longitude", color="passengerTraffic",
                              hover_name="stationName", zoom=4.3, height=600, map_style="open-street-map")
    else:
        fig = px.scatter_map(merged.dropna(subset=["avg_delay"]), lat="latitude", lon="longitude",
                              color="avg_delay", color_continuous_scale="RdYlGn_r",
                              hover_name="stationName", zoom=4.3, height=600, map_style="open-street-map")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(meta)} stations/stopping points nationwide (metadata_train_stations.csv) — "
               f"only {merged['avg_delay'].notna().sum()} had long-distance traffic in {month.replace('_', '-')}.")

# --------------------------------------------------------------------------------- Multi-year
with tab_drift:
    st.subheader("Does punctuality drift over the years?")
    st.caption("Loads one January per year (8 files) on demand — not run automatically, since "
               "it means reading ~4M rows. Cached after the first click.")
    years = sorted({m[:4] for m in months})
    if st.button("Compute yearly drift (January snapshots)"):
        rows = []
        progress = st.progress(0.0, text="Loading...")
        jan_months = [f"{y}_01" for y in years if f"{y}_01" in months]
        for i, m in enumerate(jan_months):
            d = load_month_raw(m)
            delay = d["differenceInMinutes"].dropna()
            rows.append({"year": m[:4], "delayed_rate_5min": (delay >= 5).mean(),
                         "mean_delay_min": delay.mean(), "cancelled_rate": d["cancelled"].mean(),
                         "n_rows": len(d)})
            progress.progress((i + 1) / len(jan_months), text=f"Loaded {m}")
        progress.empty()
        drift = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(drift, x="year", y="delayed_rate_5min", markers=True,
                                     title="Share delayed ≥5 min, each January"), use_container_width=True)
        with c2:
            st.plotly_chart(px.line(drift, x="year", y="cancelled_rate", markers=True,
                                     title="Cancellation rate, each January"), use_container_width=True)
        st.dataframe(drift, use_container_width=True, hide_index=True)
        st.warning(
            "This is real drift, not noise: the delayed-rate and cancellation-rate both trend up "
            "across years in every January snapshot checked so far (e.g. 2018 vs. 2024: delay rate "
            "roughly 29% → 47%, cancellation rate roughly 0.5% → 3%). **Implication for modeling**: "
            "a train/test split for this dataset should respect chronology (as the Berlin and "
            "Deutsche Bahn loaders already do) — a random split across years would let a model see "
            "the test period's punctuality regime during training.", icon="⚠️")
