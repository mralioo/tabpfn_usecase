import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    return go, mo, np, pd, px


@app.cell
def _(mo):
    mo.md(r"""
    # 00 — Data quality & EDA: Berlin U-Bahn (Alstom / InnoTrans 2026)

    Before asking *any* model a question, this notebook looks at what the raw files
    actually contain: shapes, missing values, duplicates, timestamp coverage, and the
    basic distributions that later notebooks assume. Every other notebook in this folder
    (`01_demand_forecasting_and_overcrowding.py`, `02_closure_impact_and_network_resilience.py`,
    `03_energy_consumption_forecasting.py`) builds on the same loaders used here —
    `tabpfn_lab.datasets.berlin` — so a data problem caught here is a problem avoided
    everywhere downstream.

    **Provenance, stated plainly** (see `data/SOURCE.md`): station/line geometry and
    weather are real; passenger flow, closures, and energy consumption are Alstom-simulated
    for the hackathon. Events are real public listings. That split matters for how much
    weight to put on any model's accuracy number later.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return


@app.cell
def _():
    from tabpfn_lab.config import BERLIN_DATA_DIR, load_dotenv
    from tabpfn_lab.datasets import berlin as bl

    _ = load_dotenv()
    folder = bl.discover_dataset_dir(BERLIN_DATA_DIR)
    print(folder)
    return bl, folder


@app.cell
def _(bl, folder):
    stations = bl.load_stations(folder)
    connections = bl.load_connections(folder)
    lines = bl.load_lines(folder)
    flows = bl.load_flows(folder)
    weather = bl.load_weather(folder)
    events = bl.load_events(folder)
    closures = bl.load_closures(folder)
    energy = bl.load_energy(folder)
    return (
        closures,
        connections,
        energy,
        events,
        flows,
        lines,
        stations,
        weather,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Raw file shapes & dtypes
    """)
    return


@app.cell
def _(
    closures,
    connections,
    energy,
    events,
    flows,
    lines,
    mo,
    pd,
    stations,
    weather,
):
    frames = {
        "stations_with_ubahn": stations, "berlin_ubahn_connections": connections,
        "berlin_ubahn_lines_used": lines, "flows (melted wide)": flows,
        "weather_data": weather, "berlin_events": events,
        "closures": closures, "energy_consumption": energy,
    }
    shape_table = pd.DataFrame(
        [{"file": name, "rows": len(df), "cols": df.shape[1], "dtypes": df.dtypes.nunique()}
         for name, df in frames.items()]
    )
    mo.ui.table(shape_table, selection=None)
    return (frames,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Data-quality checks
    """)
    return


@app.cell
def _(frames, mo, pd):
    missing_rows = []
    for name, df in frames.items():
        na = df.isna().sum()
        for col, n in na[na > 0].items():
            missing_rows.append({"file": name, "column": col, "missing": int(n), "pct": round(100 * n / len(df), 2)})
    missing_table = pd.DataFrame(missing_rows) if missing_rows else pd.DataFrame([{"file": "—", "column": "—", "missing": 0, "pct": 0.0}])
    mo.vstack([
        mo.md("**Missing values** (zero-missing columns omitted):"),
        mo.ui.table(missing_table, selection=None),
    ])
    return


@app.cell
def _(mo, stations):
    dup_names = stations[stations.duplicated("station_name", keep=False)].sort_values("station_name")
    mo.vstack([
        mo.md(
            f"""**Duplicate `station_name` values in `stations_with_ubahn.csv`**: {dup_names["station_name"].nunique()}
            physical station(s) appear as more than one row (interchange platforms with separate
            `station_id`s sharing one `flows.csv` column — `build_feature_table` collapses these
            by name, and the Closure Impact Lab map shows them as two nearby dots). Not a data bug."""
        ),
        mo.ui.table(dup_names, selection=None),
    ])
    return


@app.cell
def _(flows, mo, weather):
    flow_ts = set(flows["timestamp"])
    weather_ts = set(weather["timestamp"])
    only_flow = len(flow_ts - weather_ts)
    only_weather = len(weather_ts - flow_ts)
    both = len(flow_ts & weather_ts)
    gaps = flows["timestamp"].sort_values().diff().dropna()
    expected = gaps.mode().iloc[0]
    irregular = int((gaps != expected).sum())
    mo.md(
        f"""**Timestamp alignment (flows ↔ weather)**: {both} shared 15-min marks,
        {only_flow} flow-only, {only_weather} weather-only.
        **Flow timestamp spacing**: modal step is `{expected}`; {irregular} of {len(gaps)}
        consecutive gaps deviate from it (likely the train/test file boundary, not corruption)."""
    )
    return


@app.cell
def _(flows, mo):
    flow_cols = [c for c in flows.columns if c != "timestamp"]
    neg_count = int((flows[flow_cols] < 0).to_numpy().sum())
    zero_frac = float((flows[flow_cols] == 0).to_numpy().mean())
    mo.md(
        f"""**Flow sanity**: {neg_count} negative readings (should be 0); zero-flow cells are
        {zero_frac:.1%} of the table (expected — stations are quiet at 3am). Max single reading:
        {int(flows[flow_cols].to_numpy().max())} passengers/15min."""
    )
    return (flow_cols,)


@app.cell
def _(closures, mo):
    overlap_count = 0
    sorted_c = closures.sort_values("when").reset_index(drop=True)
    for i in range(len(sorted_c) - 1):
        if sorted_c.loc[i, "end"] > sorted_c.loc[i + 1, "when"]:
            overlap_count += 1
    mo.md(
        f"""**Closure overlaps**: {overlap_count} of {len(closures)} recorded closures overlap
        in time with the next one — relevant for `network_active_closures` in the feature table
        and for the Closure Impact Lab's "what if two things break at once" edge case (not
        modelled there today)."""
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Station-level flow: pick one and look at it
    """)
    return


@app.cell
def _(mo, stations):
    station_picker = mo.ui.dropdown(
        options=sorted(stations["station_name"].unique()),
        value="S+U Alexanderplatz Bhf (Berlin)" if "S+U Alexanderplatz Bhf (Berlin)" in stations["station_name"].values else sorted(stations["station_name"].unique())[0],
        label="Station",
    )
    station_picker
    return (station_picker,)


@app.cell
def _(flows, go, mo, station_picker, weather):
    _name = station_picker.value
    _series = flows[["timestamp", _name]].rename(columns={_name: "passengers"})
    _merged = _series.merge(weather[["timestamp", "temp", "prcp"]], on="timestamp", how="left")
    _fig = go.Figure()
    _fig.add_scatter(x=_merged["timestamp"], y=_merged["passengers"], name="passengers", line=dict(width=1))
    _fig.update_layout(
        title=f"15-min passenger flow — {_name}",
        xaxis_title="timestamp", yaxis_title="passengers / 15min", height=360,
        margin=dict(t=40, b=20),
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Distributions
    """)
    return


@app.cell
def _(flow_cols, flows, mo, np, px):
    _sample = flows[flow_cols].to_numpy().ravel()
    _sample = _sample[_sample > 0]
    _fig = px.histogram(np.log1p(_sample), nbins=60, title="log1p(passengers) — all stations, all non-zero 15-min slots")
    _fig.update_layout(height=320, margin=dict(t=40, b=20), xaxis_title="log1p(passengers)", showlegend=False)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(flow_cols, flows, mo, pd, px):
    _long = flows[["timestamp"] + flow_cols[:1]].copy()  # placeholder to keep cell cheap; real hour/weekday view below uses the full melt in notebook 01
    _hourly = flows.copy()
    _hourly["hour"] = pd.to_datetime(_hourly["timestamp"]).dt.hour
    _totals = _hourly.groupby("hour")[flow_cols].sum().sum(axis=1).reset_index(name="total_passengers")
    _fig = px.bar(_totals, x="hour", y="total_passengers", title="Network-wide total passengers by hour of day")
    _fig.update_layout(height=320, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo, px, weather):
    _fig = px.histogram(weather, x="temp", nbins=40, title="Temperature distribution (°C) — real Meteostat-style data")
    _fig.update_layout(height=300, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(events, mo, px):
    _fig = px.histogram(events, x="estimated_attendance", nbins=40, title="Event estimated attendance distribution")
    _fig.update_layout(height=300, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(energy, mo, px):
    _line_cols = [c for c in energy.columns if c != "timestamp"]
    _long_energy = energy.melt(id_vars="timestamp", value_vars=_line_cols, var_name="line", value_name="mwh")
    _fig = px.line(_long_energy, x="timestamp", y="mwh", color="line", title="Daily energy consumption per line (MWh) — simulated")
    _fig.update_layout(height=380, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. What this says about usable questions

    Given what's actually in these files, three distinct modelling use cases are
    well-supported (each gets its own notebook):

    1. **Passenger demand / overcrowding** (`01_demand_forecasting_and_overcrowding.py`) —
       the richest signal: real weather + real events + simulated closures, all joined to a
       clean 15-min flow table. Two targets, one feature row, as the MCP server already does.
    2. **Closure / disruption impact** (`02_closure_impact_and_network_resilience.py`) — the
       real connection graph plus the simulated closure log gives a genuine what-if surface:
       which stations are reachable-only-through a given one (articulation points), and how
       much flow redistributes nearby.
    3. **Energy consumption forecasting** (`03_energy_consumption_forecasting.py`) — unused
       anywhere else in this repo until now. Daily, per-line, plausibly driven by ridership
       and weather — a regression task at a completely different granularity (daily vs
       15-min) than the other two, worth checking whether that changes which model wins.

    What's **not** well-supported by this data: anything needing ground-truth capacity
    figures (no platform-capacity column exists — "overcrowded" is defined relative to a
    station's own history, not a hard limit), or anything claiming real-world passenger
    counts (flows are simulated, however realistic-looking).
    """)
    return


if __name__ == "__main__":
    app.run()
