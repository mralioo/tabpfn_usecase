import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    return go, mo, np, pd, px


@app.cell
def _(mo):
    mo.md(r"""
    # 03 — Energy consumption forecasting: a use case nobody's asked yet

    `energy_consumption_pre_innotrans.csv` / `energy_consumption_rest.csv` — daily MWh per
    U-Bahn line — are loaded by `tabpfn_lab.datasets.berlin.load_energy` but, until this
    notebook, never joined to anything or fed to a model anywhere in this repo. It's a
    legitimate fourth question an operator (or an LLM agent on their behalf) could ask:
    *"how much energy will line U7 draw tomorrow, given the weather and how busy it's been?"*
    — useful for procurement and grid-load planning, same spirit as the other two questions
    but at a **completely different grain**: daily, per-line, ~110 rows per line for the whole
    dataset instead of 15-minute, per-station, millions of rows.

    That grain change is the point of doing this as its own notebook: does the
    model comparison from notebook 01 hold up when there's barely any data?
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
    from tabpfn_lab.config import BERLIN_DATA_DIR, TABPFN_MODEL_PATH, authenticate, load_dotenv
    from tabpfn_lab.datasets.berlin import (
        discover_dataset_dir, load_closures, load_energy, load_flows, load_stations, load_weather,
        station_cols,
    )
    from tabpfn_lab.xgboost_baseline import fit_regressor

    _ = load_dotenv()
    folder = discover_dataset_dir(BERLIN_DATA_DIR)
    return (
        TABPFN_MODEL_PATH,
        authenticate,
        fit_regressor,
        folder,
        load_closures,
        load_energy,
        load_flows,
        load_stations,
        load_weather,
        station_cols,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Build the feature table — cleaning + preprocessing from scratch

    No existing loader does this join, so it's built here: daily ridership per line (summing
    each line's stations' 15-min flow — an interchange station's flow is counted toward every
    line it serves, a deliberate simplification flagged rather than hidden), daily weather
    aggregates, a same-day closure count per line, and calendar features.
    """)
    return


@app.cell
def _(folder, load_energy, load_stations):
    energy_wide = load_energy(folder)
    stations = load_stations(folder)
    line_cols = [c for c in energy_wide.columns if c != "timestamp"]
    energy_long = energy_wide.melt(id_vars="timestamp", value_vars=line_cols, var_name="line", value_name="mwh")
    energy_long["date"] = energy_long["timestamp"].dt.date
    return energy_long, line_cols, stations


@app.cell
def _(energy_long, line_cols, mo):
    mo.md(f"""
    **Energy data**: {len(energy_long)} line-days across {len(line_cols)} lines ({', '.join(line_cols)}), {energy_long['date'].min()} → {energy_long['date'].max()}.
    """)
    return


@app.cell
def _(folder, line_cols, load_flows, pd, station_cols, stations):
    flows = load_flows(folder)
    fcols = station_cols(flows)
    flows_long = flows.melt(id_vars="timestamp", value_vars=fcols, var_name="station_name", value_name="passengers")
    flows_long["date"] = flows_long["timestamp"].dt.date

    station_lines = stations.set_index("station_name")["u_bahn_lines"].to_dict()
    daily_station = flows_long.groupby(["date", "station_name"])["passengers"].sum().reset_index()

    _rows = []
    for _, _row in daily_station.iterrows():
        _lines = str(station_lines.get(_row["station_name"], "")).split(",")
        for _ln in [l.strip() for l in _lines if l.strip() in line_cols]:
            _rows.append({"date": _row["date"], "line": _ln, "passengers": _row["passengers"]})
    ridership_per_line = pd.DataFrame(_rows).groupby(["date", "line"])["passengers"].sum().reset_index(name="daily_ridership")
    return (ridership_per_line,)


@app.cell
def _(folder, load_weather):
    weather = load_weather(folder)
    weather["date"] = weather["timestamp"].dt.date
    daily_weather = weather.groupby("date").agg(
        temp=("temp", "mean"), prcp=("prcp", "sum"), wspd=("wspd", "mean"), cldc=("cldc", "mean"),
    ).reset_index()
    return (daily_weather,)


@app.cell
def _(folder, load_closures):
    closures = load_closures(folder)
    return (closures,)


@app.cell
def _(closures, energy_long, line_cols, pd):
    _dates = sorted(energy_long["date"].unique())

    def _closures_for(date, line):
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)
        overlap = (closures["when"] < day_end) & (closures["end"] > day_start)
        on_line = closures["affected_line"] == line
        return int((overlap & on_line).sum())

    network_closure_rows = []
    for _d in _dates:
        for _ln in line_cols:
            network_closure_rows.append({"date": _d, "line": _ln, "line_closures_today": _closures_for(_d, _ln)})
    closures_by_line_day = pd.DataFrame(network_closure_rows)
    return (closures_by_line_day,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. Assemble, clean, and sanity-check
    """)
    return


@app.cell
def _(
    closures_by_line_day,
    daily_weather,
    energy_long,
    pd,
    ridership_per_line,
):
    table = energy_long.merge(ridership_per_line, on=["date", "line"], how="left")
    table = table.merge(daily_weather, on="date", how="left")
    table = table.merge(closures_by_line_day, on=["date", "line"], how="left")
    table["line_closures_today"] = table["line_closures_today"].fillna(0)
    table["date"] = pd.to_datetime(table["date"])
    table["dow"] = table["date"].dt.dayofweek
    table["is_weekend"] = (table["dow"] >= 5).astype(int)
    table["month"] = table["date"].dt.month
    before = len(table)
    table = table.dropna(subset=["mwh", "daily_ridership", "temp"])
    after = len(table)
    return after, before, table


@app.cell
def _(after, before, mo, table):
    mo.md(f"""
    **Assembled table**: {before} rows → {after} after dropping rows with no ridership/weather
        match ({before - after} dropped — edge days at the dataset boundary). Columns:
        `{", ".join(table.columns)}`.
    """)
    return


@app.cell
def _(mo, px, table):
    _fig = px.scatter(table, x="daily_ridership", y="mwh", color="line",
                       title="Daily energy (MWh) vs. daily ridership, per line")
    _fig.update_layout(height=420, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo, px, table):
    _fig = px.scatter(table, x="temp", y="mwh", color="line", title="Daily energy (MWh) vs. mean temperature (°C), per line")
    _fig.update_layout(height=380, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo, table):
    line_picker = mo.ui.dropdown(options=sorted(table["line"].unique()), value=sorted(table["line"].unique())[0], label="Line")
    line_picker
    return (line_picker,)


@app.cell
def _(go, line_picker, mo, table):
    _sub = table[table["line"] == line_picker.value].sort_values("date")
    _fig = go.Figure()
    _fig.add_scatter(x=_sub["date"], y=_sub["mwh"], name="MWh", yaxis="y1")
    _fig.add_scatter(x=_sub["date"], y=_sub["daily_ridership"], name="ridership", yaxis="y2")
    _fig.update_layout(
        title=f"{line_picker.value}: daily energy vs. ridership over time",
        yaxis=dict(title="MWh"), yaxis2=dict(title="ridership", overlaying="y", side="right"),
        height=380, margin=dict(t=40, b=20),
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Model comparison — small-data regime

    The whole table is small enough (~100 rows per line) that TabPFN-3.5 needs **no
    subsampling at all** — the opposite situation from notebook 01's 1.2M-row flow table.
    This is the regime TabPFN-3.5's "in-context, zero-pipeline" pitch is built for: hand it
    every row, get a usable model back with no training loop.
    """)
    return


@app.cell
def _(np, table):
    FEATURES = ["daily_ridership", "temp", "prcp", "wspd", "cldc", "line_closures_today", "dow", "is_weekend", "month"]
    CAT_FEATURE = "line"
    line_codes, line_categories = table[CAT_FEATURE].factorize()
    X_all = table[FEATURES].copy()
    X_all["line_code"] = line_codes
    y_all = table["mwh"].to_numpy(float)

    dates_sorted = np.sort(table["date"].unique())
    cutoff_idx = int(len(dates_sorted) * 0.8)
    cutoff_date = dates_sorted[cutoff_idx]
    train_mask = (table["date"] < cutoff_date).to_numpy()
    test_mask = ~train_mask
    return X_all, cutoff_date, test_mask, train_mask, y_all


@app.cell
def _(X_all, cutoff_date, mo, test_mask, train_mask):
    mo.md(f"""
    **Chronological split** at `{cutoff_date}`: {int(train_mask.sum())} train rows,
        {int(test_mask.sum())} test rows. {len(X_all.columns)} features: `{list(X_all.columns)}`,
        target `mwh`.
    """)
    return


@app.cell
def _(X_all, fit_regressor, test_mask, time, train_mask, y_all):
    X_train, X_test = X_all[train_mask], X_all[test_mask]
    y_train, y_test = y_all[train_mask], y_all[test_mask]

    xgb_model, xgb_fit_s = fit_regressor(X_train, y_train)
    _t0 = time.perf_counter()
    xgb_pred = xgb_model.predict(X_test)
    xgb_predict_s = time.perf_counter() - _t0
    return (
        X_test,
        X_train,
        xgb_fit_s,
        xgb_model,
        xgb_pred,
        xgb_predict_s,
        y_test,
        y_train,
    )


@app.cell
def _():
    import time

    return (time,)


@app.cell
def _(pd, table, test_mask, train_mask, y_train):
    _train_dow = table.loc[train_mask, "dow"].to_numpy()
    _dow_means = pd.Series(y_train).groupby(_train_dow).mean()
    baseline_pred = table.loc[test_mask, "dow"].map(_dow_means).fillna(float(y_train.mean())).to_numpy()
    return (baseline_pred,)


@app.cell
def _(mo):
    mo.md("""
    TabPFN-3.5 fit directly via `tabpfn_client` (not the `tabpfn_lab.models` wrapper, which is hard-coded to the 15-min flow feature set) — same auth helper, same model version constant, new feature table.
    """)
    return


@app.cell
def _(TABPFN_MODEL_PATH, X_test, X_train, authenticate, time, y_train):
    try:
        authenticate()
        from tabpfn_client import TabPFNRegressor
        _cat_idx = [X_train.columns.get_loc("line_code")]
        _t0 = time.perf_counter()
        tab_model = TabPFNRegressor(model_path=TABPFN_MODEL_PATH, categorical_features_indices=_cat_idx)
        tab_model.fit(X_train, y_train)
        tab_fit_s = time.perf_counter() - _t0
        _t0 = time.perf_counter()
        tab_pred = tab_model.predict(X_test)
        tab_predict_s = time.perf_counter() - _t0
        tab_error = None
    except Exception as exc:  # noqa: BLE001
        tab_pred = None
        tab_fit_s = tab_predict_s = None
        tab_error = str(exc)
    return tab_error, tab_fit_s, tab_pred, tab_predict_s


@app.cell
def _(
    mo,
    tab_error,
    tab_fit_s,
    tab_predict_s,
    train_mask,
    xgb_fit_s,
    xgb_predict_s,
):
    mo.md(f"""
    **XGBoost**: fit {xgb_fit_s * 1000:.1f}ms, predict {xgb_predict_s * 1000:.1f}ms.
        **TabPFN-3.5**: {"fit " + f"{tab_fit_s:.2f}s" + ", predict " + f"{tab_predict_s:.2f}s" if tab_error is None else "unavailable (" + tab_error + ")"} —
        on {int(train_mask.sum())} training rows this is the entire dataset, not a sample.
    """)
    return


@app.cell
def _(
    baseline_pred,
    mean_absolute_error,
    mo,
    pd,
    r2_score,
    tab_pred,
    xgb_pred,
    y_test,
):
    _rows = [
        {"model": "Day-of-week baseline", "mae": round(mean_absolute_error(y_test, baseline_pred), 2), "r2": round(r2_score(y_test, baseline_pred), 3)},
        {"model": "XGBoost", "mae": round(mean_absolute_error(y_test, xgb_pred), 2), "r2": round(r2_score(y_test, xgb_pred), 3)},
    ]
    if tab_pred is not None:
        _rows.append({"model": "TabPFN-3.5", "mae": round(mean_absolute_error(y_test, tab_pred), 2), "r2": round(r2_score(y_test, tab_pred), 3)})
    energy_metrics = pd.DataFrame(_rows)
    mo.ui.table(energy_metrics, selection=None)
    return


@app.cell
def _():
    from sklearn.metrics import mean_absolute_error, r2_score

    return mean_absolute_error, r2_score


@app.cell
def _(go, mo, tab_pred, xgb_pred, y_test):
    _fig = go.Figure()
    _fig.add_scatter(x=y_test, y=xgb_pred, mode="markers", name="XGBoost")
    if tab_pred is not None:
        _fig.add_scatter(x=y_test, y=tab_pred, mode="markers", name="TabPFN-3.5")
    _max_v = float(y_test.max())
    _fig.add_scatter(x=[0, _max_v], y=[0, _max_v], mode="lines", name="perfect", line=dict(dash="dash", color="gray"))
    _fig.update_layout(title="Predicted vs. actual daily energy (MWh)", xaxis_title="actual", yaxis_title="predicted", height=420, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. What actually drives energy use? (XGBoost feature importance)
    """)
    return


@app.cell
def _(X_train, mo, pd, px, xgb_model):
    importance = pd.DataFrame({"feature": X_train.columns, "importance": xgb_model.feature_importances_}).sort_values("importance", ascending=False)
    _fig = px.bar(importance, x="feature", y="importance", title="XGBoost feature importance — daily energy per line")
    _fig.update_layout(height=360, margin=dict(t=40, b=20))
    mo.vstack([mo.ui.plotly(_fig), mo.ui.table(importance, selection=None)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Takeaways

    - **Ridership dominates**, which is the sane result — more trains running more often
      to move more people draws more power — but check the importance chart above: if
      `line_closures_today` or weather rank surprisingly high, that's worth a second look
      rather than taking at face value (small-N feature importances are noisier than
      1.2M-row ones).
    - **TabPFN-3.5's pitch is strongest exactly here**: ~80 training rows is below where
      it would be worth building an XGBoost pipeline at all in a real team — "paste the
      table, get a model" is the realistic alternative to not modelling this at all, not
      a speed contest with a model that needed feature-engineering effort to set up.
    - **Honest caveat**: energy consumption here is Alstom-simulated, same as flow and
      closures (see `data/SOURCE.md`) — treat the ridership-energy relationship as
      plausible-by-construction, not a validated real physical model.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
