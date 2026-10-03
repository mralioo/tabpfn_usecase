"""Berlin U-Bahn data exploration — condensed from NextMove's six-page dashboard
(Network Explorer, Passenger Flow, Events, Weather, Closures, Energy) into one page's worth of
tabs. Adapted, not copy-pasted: NextMove's charts use st.cache_data over a Streamlit-specific
loader; here everything reads through `tabpfn_lab.datasets.berlin`, the same module the agent,
MCP server and benchmark scripts use, so this page can never silently disagree with them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tabpfn_lab.config import BERLIN_DATA_DIR  # noqa: E402
from tabpfn_lab.datasets.berlin import (  # noqa: E402
    LINE_COLORS,
    discover_dataset_dir,
    dow_hour_heatmap,
    flows_long,
    hourly_profile,
    load_closures,
    load_connections,
    load_energy,
    load_events,
    load_flows,
    load_stations,
    load_weather,
    network_resilience,
    network_total_flow,
    station_avg_flow,
    station_cols,
)

st.title("📊 Berlin U-Bahn — Data Exploration")
st.caption(
    "Simulated flow (Alstom data team, InnoTrans 2026 hackathon) over a real station/line "
    "geometry. See `data/SOURCE.md`. Six angles on the same dataset, one tab each."
)

folder = discover_dataset_dir(BERLIN_DATA_DIR)
stations = load_stations(folder)
avg_flow = station_avg_flow(folder)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stations", stations["station_name"].nunique())
c2.metric("Lines", stations["u_bahn_lines"].str.split(",").explode().str.strip().nunique())
c3.metric("Avg daily passengers (network)", f"{avg_flow['avg_daily_passengers'].sum():,.0f}")
c4.metric("Busiest station", avg_flow.sort_values('avg_daily_passengers', ascending=False).iloc[0]['station_name'])

tab_net, tab_flow, tab_events, tab_weather, tab_closures, tab_energy = st.tabs(
    ["🗺️ Network", "📈 Passenger flow", "🎟️ Events", "🌦️ Weather", "🚧 Closures", "⚡ Energy"])

# ------------------------------------------------------------------------------------- Network
with tab_net:
    st.subheader("Station map")
    connections = load_connections(folder)
    resilience = network_resilience(folder)
    merged = stations.merge(avg_flow[["station_name", "avg_daily_passengers"]], on="station_name", how="left") \
        .merge(resilience[["station_name", "is_articulation_point", "betweenness_centrality"]], on="station_name", how="left")

    highlight = st.radio("Highlight", ["By line", "By ridership", "By fragmentation risk"], horizontal=True)
    id_pos = merged.set_index("station_id")[["longitude", "latitude"]]
    lon_e, lat_e = [], []
    for _, row in connections.iterrows():
        if row["station_id_1"] in id_pos.index and row["station_id_2"] in id_pos.index:
            p1, p2 = id_pos.loc[row["station_id_1"]], id_pos.loc[row["station_id_2"]]
            lon_e += [p1["longitude"], p2["longitude"], None]
            lat_e += [p1["latitude"], p2["latitude"], None]

    fig = go.Figure()
    fig.add_trace(go.Scattermap(lon=lon_e, lat=lat_e, mode="lines",
                                    line=dict(width=1.5, color="rgba(120,120,120,0.6)"), hoverinfo="skip", showlegend=False))
    if highlight == "By line":
        for line, color in LINE_COLORS.items():
            sub = merged[merged["primary_line"] == line]
            if sub.empty:
                continue
            fig.add_trace(go.Scattermap(lon=sub["longitude"], lat=sub["latitude"], mode="markers", name=line,
                                            marker=dict(size=9, color=color),
                                            text=sub["station_name"] + "<br>Lines: " + sub["u_bahn_lines"], hoverinfo="text"))
    elif highlight == "By ridership":
        fig.add_trace(go.Scattermap(
            lon=merged["longitude"], lat=merged["latitude"], mode="markers",
            marker=dict(size=8 + 24 * merged["avg_daily_passengers"].fillna(0) / merged["avg_daily_passengers"].max(),
                        color=merged["avg_daily_passengers"], colorscale="YlOrRd", showscale=True,
                        colorbar=dict(title="Avg daily pax")),
            text=merged["station_name"], hoverinfo="text", showlegend=False))
    else:
        fig.add_trace(go.Scattermap(
            lon=merged["longitude"], lat=merged["latitude"], mode="markers",
            marker=dict(size=8 + 20 * merged["betweenness_centrality"].fillna(0) / max(merged["betweenness_centrality"].max(), 1e-9),
                        color=merged["is_articulation_point"].map({True: "#d62728", False: "#1f77b4"})),
            text=merged["station_name"] + "<br>Critical: " + merged["is_articulation_point"].astype(str),
            hoverinfo="text", showlegend=False))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lon=13.38, lat=52.51), zoom=10.2),
                       height=600, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fragmentation risk — which station closures hurt the network most?")
    st.caption("Pure graph topology (networkx articulation points + betweenness centrality), no model. "
               "Fragmentation score = betweenness × avg daily ridership.")
    top_n = st.slider("Top N stations", 5, 30, 10, key="berlin_topn")
    st.dataframe(resilience.head(top_n)[["station_name", "is_articulation_point", "betweenness_centrality",
                                          "resulting_components", "avg_daily_passengers", "fragmentation_score"]],
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------------- Pass. flow
with tab_flow:
    st.subheader("Busiest & quietest stations")
    n = st.slider("How many to rank", 5, 30, 15, key="berlin_flow_n")
    ranked = avg_flow.sort_values("avg_daily_passengers", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(ranked.head(n).sort_values("avg_daily_passengers"), x="avg_daily_passengers", y="station_name",
                     orientation="h", title=f"Top {n} busiest")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(ranked.tail(n).sort_values("avg_daily_passengers"), x="avg_daily_passengers", y="station_name",
                     orientation="h", title=f"Bottom {n} quietest")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Station deep-dive")
    all_stations = sorted(station_cols(load_flows(folder)))
    default_sel = [ranked.iloc[0]["station_name"]]
    selected = st.multiselect("Select one or more stations", all_stations, default=default_sel)
    if selected:
        long = flows_long(folder)
        sel_long = long[long["station_name"].isin(selected)]
        sub1, sub2, sub3 = st.tabs(["Time series (daily)", "Hourly profile", "Day × hour heatmap"])
        with sub1:
            plot_df = sel_long.set_index("timestamp").groupby("station_name")["passengers"].resample("D").sum().reset_index()
            fig = px.line(plot_df, x="timestamp", y="passengers", color="station_name")
            st.plotly_chart(fig, use_container_width=True)
        with sub2:
            prof = hourly_profile(folder, tuple(sorted(selected)))
            prof["Day type"] = prof["is_weekend"].map({True: "Weekend", False: "Weekday"})
            fig = px.line(prof, x="hour", y="passengers", color="Day type", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        with sub3:
            heat = dow_hour_heatmap(folder, tuple(sorted(selected)))
            fig = px.imshow(heat, aspect="auto", color_continuous_scale="YlOrRd",
                             labels=dict(x="Hour of day", y="", color="Avg passengers"))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one station to see its time series and rhythm.")

# ------------------------------------------------------------------------------------- Events
with tab_events:
    events = load_events(folder)
    total_flow = network_total_flow(folder)
    daily_events = events.groupby("date").agg(n=("event_name", "count"), attendance=("estimated_attendance", "sum")).reset_index()
    daily_flow = total_flow.set_index("timestamp")["total_passengers"].resample("D").sum().reset_index()
    daily_flow["date"] = daily_flow["timestamp"].dt.date
    joined = daily_flow.merge(daily_events, on="date", how="left").fillna({"n": 0, "attendance": 0})

    st.subheader("Daily event load vs. network-wide passenger flow")
    fig = go.Figure()
    fig.add_bar(x=joined["date"], y=joined["attendance"], name="Event attendance (sum)", yaxis="y2", opacity=0.4)
    fig.add_scatter(x=joined["date"], y=joined["total_passengers"], name="Network passengers", mode="lines")
    fig.update_layout(height=400, yaxis2=dict(overlaying="y", side="right", title="Attendance"))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Events by segment")
        st.plotly_chart(px.bar(events["segment"].value_counts().reset_index(), x="segment", y="count"), use_container_width=True)
    with c2:
        st.subheader("Attendance distribution")
        st.plotly_chart(px.histogram(events, x="estimated_attendance", nbins=30), use_container_width=True)

    st.subheader("Largest events")
    st.dataframe(events.sort_values("estimated_attendance", ascending=False)
                 [["event_name", "began_local", "venue_name", "segment", "estimated_attendance"]].head(15),
                 use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------------ Weather
with tab_weather:
    weather = load_weather(folder)
    total_flow = network_total_flow(folder)
    joined = total_flow.merge(weather, on="timestamp", how="inner")

    st.subheader("Correlation with network-wide 15-min passenger flow")
    corr_cols = ["total_passengers", "temp", "rhum", "prcp", "wdir", "wspd", "pres", "cldc"]
    corr = joined[corr_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
    st.caption(f"`total_passengers` vs `temp`: r = {corr.loc['total_passengers', 'temp']:.2f} — "
               f"vs `prcp`: r = {corr.loc['total_passengers', 'prcp']:.2f}. Weak at 15-min granularity "
               "(time-of-day dominates); matches `data/Berlin_Ubahn_Alstom_data` README's own note that "
               "weather explains ~0.5% of variance in this simulated data.")

    st.subheader("Daily aggregation: weather vs. ridership")
    daily = joined.set_index("timestamp").resample("D").agg(
        total_passengers=("total_passengers", "sum"), temp=("temp", "mean"), prcp=("prcp", "sum")).reset_index()
    fig = px.scatter(daily, x="temp", y="total_passengers")
    valid = daily[["temp", "total_passengers"]].dropna()
    if len(valid) >= 2:
        slope, intercept = np.polyfit(valid["temp"], valid["total_passengers"], 1)
        x_line = np.linspace(valid["temp"].min(), valid["temp"].max(), 50)
        fig.add_scatter(x=x_line, y=slope * x_line + intercept, mode="lines",
                         name="Trend (linear fit)", line=dict(color="black", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------------- Closures
with tab_closures:
    closures = load_closures(folder).copy()
    closures["affected_segment"] = closures["affected_segment"].fillna(closures["closure_type"])
    st.subheader("Timeline")
    fig = px.timeline(closures, x_start="when", x_end="end", y="affected_segment", color="closure_type")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By affected line")
        st.plotly_chart(px.bar(closures["affected_line"].value_counts().reset_index(), x="affected_line", y="count"),
                         use_container_width=True)
    with c2:
        st.subheader("Duration distribution")
        st.plotly_chart(px.histogram(closures, x="duration_hours", nbins=20), use_container_width=True)

    st.subheader("All closures")
    st.dataframe(closures[["when", "duration_hours", "closure_type", "affected_line", "affected_segment", "reason"]],
                 use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------------- Energy
with tab_energy:
    energy = load_energy(folder)
    line_cols = [c for c in energy.columns if c != "timestamp"]
    st.subheader("Daily energy consumption by line")
    melted = energy.melt(id_vars="timestamp", value_vars=line_cols, var_name="line", value_name="MWh")
    st.plotly_chart(px.line(melted, x="timestamp", y="MWh", color="line",
                             color_discrete_map=LINE_COLORS), use_container_width=True)

    st.subheader("Energy-per-passenger efficiency by line")
    flows = load_flows(folder)
    stations_by_line = stations.groupby("primary_line")["station_name"].apply(list).to_dict()
    eff_rows = []
    for line in line_cols:
        cols = [c for c in stations_by_line.get(line, []) if c in flows.columns]
        if not cols:
            continue
        total_pax = flows[cols].sum().sum()
        total_energy = energy[line].sum()
        eff_rows.append({"line": line, "total_passengers": total_pax, "total_MWh": total_energy,
                          "Wh_per_passenger": (total_energy * 1000 / total_pax) if total_pax else np.nan})
    eff = pd.DataFrame(eff_rows).sort_values("Wh_per_passenger")
    st.plotly_chart(px.bar(eff, x="line", y="Wh_per_passenger", color="line", color_discrete_map=LINE_COLORS),
                     use_container_width=True)
    st.caption("Lower = more passengers carried per unit of energy. Approximate: a station's full "
               "ridership is attributed to its primary line only, interchanges undercount other lines.")
