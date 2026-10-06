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
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score, mean_absolute_error, precision_recall_curve,
        r2_score, roc_auc_score, roc_curve,
    )

    return (
        average_precision_score,
        calibration_curve,
        go,
        mean_absolute_error,
        mo,
        np,
        pd,
        px,
        r2_score,
        roc_auc_score,
        roc_curve,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 01 — Demand forecasting & overcrowding: the two questions an operator actually asks

    This notebook is the **motivation** for the project's final use case. It tests the
    first, obvious framing of "overcrowding" and shows why it isn't enough:

    - **Classification** — *"will this station be at/above its own historical
      90th-percentile flow in this 15-minute slot?"*
    - **Regression** — *"how many passengers are expected?"*

    Same feature row (time, weather, events, closure state, station context), two
    model calls. We benchmark three ways to answer both questions — a historical-average
    **lookup-table baseline**, a locally-fit **XGBoost** pair, and **TabPFN-3.5** — and look
    past the headline metric at calibration, per-station spread, and latency.

    **The punchline:** "flow ≥ the station's own P90" is mostly *the daily clock* — the
    busy slots are the same hours every day — so a station × weekend × hour lookup table
    already captures nearly all of it and TabPFN-3.5 can only tie it. The refined use case —
    **anomaly early warning on normalised flows** (deviation from what's *expected* for that
    station and time, not raw volume) — lives in `02_anomaly_early_warning.py`.
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
    from tabpfn_lab.config import TABPFN_MODEL_PATH, authenticate, load_dotenv
    from tabpfn_lab.datasets.berlin import (
        CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_feature_table, chronological_split,
        discover_dataset_dir, encode_categoricals, stratified_subsample,
    )
    from tabpfn_lab.xgboost_baseline import DEFAULT_PARAMS

    _ = load_dotenv()
    return (
        CATEGORICAL_COLUMNS,
        DEFAULT_PARAMS,
        FEATURE_COLUMNS,
        TABPFN_MODEL_PATH,
        authenticate,
        build_feature_table,
        chronological_split,
        discover_dataset_dir,
        encode_categoricals,
        stratified_subsample,
    )


@app.cell
def _():
    # Historical lookup baseline: mean of the target per (station, weekend-flag, hour),
    # fit on the train pool only, fallback to the train-pool mean for unseen keys.
    BASELINE_KEYS = ["station_name", "is_weekend", "hour"]

    def fit_lookup_baseline(train_df, target_col):
        return train_df.groupby(BASELINE_KEYS)[target_col].mean().rename("baseline").reset_index()

    def predict_lookup_baseline(lookup, rows, fallback):
        merged = rows[BASELINE_KEYS].merge(lookup, on=BASELINE_KEYS, how="left")
        return merged["baseline"].fillna(fallback)

    return fit_lookup_baseline, predict_lookup_baseline


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load the feature table (cleaning already applied)

    `build_feature_table` (in `tabpfn_lab/datasets/berlin.py`) is the cleaning +
    preprocessing step for this use case: it melts the wide flow table, repairs mojibake
    in station names, joins weather/events/closures, computes each station's own P90
    threshold, and drops rows with any missing feature — reused here rather than
    reimplemented.
    """)
    return


@app.cell
def _(build_feature_table, chronological_split, discover_dataset_dir, mo):
    table = build_feature_table(discover_dataset_dir())
    train_pool, test_pool, cutoff = chronological_split(table)
    mo.md(
        f"""**Feature table**: {len(table):,} rows, {table["station_name"].nunique()} stations,
        {table["timestamp"].min()} → {table["timestamp"].max()}.
        **Chronological split** at `{cutoff}`: {len(train_pool):,} train rows,
        {len(test_pool):,} test rows (no date overlap)."""
    )
    return table, test_pool, train_pool


@app.cell
def _(table):
    table
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Target distributions
    """)
    return


@app.cell
def _(mo, px, table):
    _per_station = table.groupby("station_name")["overcrowded"].mean().sort_values()
    _fig = px.histogram(_per_station, nbins=30, title="Per-station overcrowding rate (should cluster near 10% by construction — P90 of its OWN history)")
    _fig.update_layout(height=320, margin=dict(t=40, b=20), showlegend=False, xaxis_title="P(overcrowded)")
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Fit all three — once — then compare

    - **XGBoost** (default hyperparameters from `tabpfn_lab/xgboost_baseline.py`) — fit
      locally on a random 300k-row sample of the train pool (keeps the notebook fast; more
      rows barely move the metric on this strongly periodic data).
    - **TabPFN-3.5** — one `.fit()` per task on an 8k-row stratified in-context sample of
      the same train pool, via the `tabpfn_client` API. If no `TABPFN_API_TOKEN` is set,
      the notebook falls back to baseline-vs-XGBoost only.
    """)
    return


@app.cell
def _(
    CATEGORICAL_COLUMNS,
    DEFAULT_PARAMS,
    FEATURE_COLUMNS,
    TABPFN_MODEL_PATH,
    authenticate,
    encode_categoricals,
    mo,
    stratified_subsample,
    time,
    train_pool,
):
    from types import SimpleNamespace
    from xgboost import XGBClassifier, XGBRegressor

    XGB_TRAIN_ROWS = 300_000
    _xgb_params = {**DEFAULT_PARAMS, "n_jobs": 8}  # n_jobs=-1 is pathologically slow on some machines
    _xgb_train = train_pool.sample(min(XGB_TRAIN_ROWS, len(train_pool)), random_state=0)
    _X = encode_categoricals(_xgb_train[FEATURE_COLUMNS])
    _t0 = time.perf_counter()
    _clf = XGBClassifier(**_xgb_params, eval_metric="logloss").fit(_X, _xgb_train["overcrowded"])
    _reg = XGBRegressor(**_xgb_params).fit(_X, _xgb_train["passengers"])
    xgb_bundle = {
        "clf": _clf, "reg": _reg, "reference": _xgb_train[FEATURE_COLUMNS],
        "fit_seconds": round(time.perf_counter() - _t0, 2), "n_rows": len(_xgb_train),
    }

    try:
        authenticate()
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor

        _sample = stratified_subsample(train_pool, 8_000, "overcrowded")
        _Xs = encode_categoricals(_sample[FEATURE_COLUMNS])
        _cat_idx = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]
        _t0 = time.perf_counter()
        _tclf = TabPFNClassifier(model_path=TABPFN_MODEL_PATH, categorical_features_indices=_cat_idx)
        _tclf.fit(_Xs, _sample["overcrowded"])
        _treg = TabPFNRegressor(model_path=TABPFN_MODEL_PATH, categorical_features_indices=_cat_idx)
        _treg.fit(_Xs, _sample["passengers"])
        tabpfn_models = SimpleNamespace(classifier=_tclf, regressor=_treg, train_sample=_sample,
                                        fit_seconds=time.perf_counter() - _t0)
        tabpfn_error = None
    except (SystemExit, Exception) as _exc:  # noqa: BLE001 - graceful fallback on anything
        tabpfn_models, tabpfn_error = None, str(_exc)

    mo.md(
        f"""**XGBoost**: fit on {xgb_bundle["n_rows"]:,} rows in {xgb_bundle["fit_seconds"]}s.
        **TabPFN-3.5**: {"fit on a " + str(len(tabpfn_models.train_sample)) + "-row in-context sample in " + str(round(tabpfn_models.fit_seconds, 2)) + "s" if tabpfn_models else "unavailable (" + str(tabpfn_error) + ") — falling back to baseline-only comparisons below"}."""
    )
    return tabpfn_models, xgb_bundle


@app.cell
def _(
    FEATURE_COLUMNS,
    encode_categoricals,
    fit_lookup_baseline,
    stratified_subsample,
    tabpfn_models,
    test_pool,
    time,
    train_pool,
    xgb_bundle,
):
    TEST_SAMPLE_SIZE = 2_000
    test_sample = stratified_subsample(test_pool, TEST_SAMPLE_SIZE, "overcrowded", seed=1)
    y_class = test_sample["overcrowded"].to_numpy()
    y_reg = test_sample["passengers"].to_numpy(float)

    rate_table = fit_lookup_baseline(train_pool, "overcrowded")
    mean_table = fit_lookup_baseline(train_pool, "passengers")

    X_xgb = encode_categoricals(test_sample[FEATURE_COLUMNS], reference=xgb_bundle["reference"])
    _t0 = time.perf_counter()
    xgb_proba = xgb_bundle["clf"].predict_proba(X_xgb)[:, 1]
    xgb_clf_s = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    xgb_expected = xgb_bundle["reg"].predict(X_xgb)
    xgb_reg_s = time.perf_counter() - _t0

    if tabpfn_models is not None:
        X_tab = encode_categoricals(test_sample[FEATURE_COLUMNS], reference=tabpfn_models.train_sample[FEATURE_COLUMNS])
        _t0 = time.perf_counter()
        tab_proba = tabpfn_models.classifier.predict_proba(X_tab)[:, 1]
        tab_clf_s = time.perf_counter() - _t0
        _t0 = time.perf_counter()
        tab_expected = tabpfn_models.regressor.predict(X_tab)
        tab_reg_s = time.perf_counter() - _t0
    else:
        tab_proba = tab_expected = None
        tab_clf_s = tab_reg_s = None
    return (
        mean_table,
        rate_table,
        tab_clf_s,
        tab_expected,
        tab_proba,
        tab_reg_s,
        test_sample,
        xgb_clf_s,
        xgb_expected,
        xgb_proba,
        xgb_reg_s,
        y_class,
        y_reg,
    )


@app.cell
def _():
    import time

    return (time,)


@app.cell
def _(
    mean_table,
    predict_lookup_baseline,
    rate_table,
    test_sample,
    time,
    train_pool,
):
    _t0 = time.perf_counter()
    base_proba = predict_lookup_baseline(rate_table, test_sample, float(train_pool["overcrowded"].mean())).to_numpy()
    base_clf_s = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    base_expected = predict_lookup_baseline(mean_table, test_sample, float(train_pool["passengers"].mean())).to_numpy()
    base_reg_s = time.perf_counter() - _t0
    return base_clf_s, base_expected, base_proba, base_reg_s


@app.cell
def _(mo):
    mo.md("""
    ## 4. Classification: P(≥ own P90 load)
    """)
    return


@app.cell
def _(
    average_precision_score,
    base_clf_s,
    base_proba,
    mo,
    pd,
    roc_auc_score,
    tab_clf_s,
    tab_proba,
    xgb_bundle,
    xgb_clf_s,
    xgb_proba,
    y_class,
):
    rows = [
        {"model": "Historical-average baseline", "roc_auc": round(roc_auc_score(y_class, base_proba), 3),
         "pr_auc": round(average_precision_score(y_class, base_proba), 3), "predict_ms": round(base_clf_s * 1000, 1)},
        {"model": "XGBoost", "roc_auc": round(roc_auc_score(y_class, xgb_proba), 3),
         "pr_auc": round(average_precision_score(y_class, xgb_proba), 3), "predict_ms": round(xgb_clf_s * 1000, 1)},
    ]
    if tab_proba is not None:
        rows.append({"model": "TabPFN-3.5", "roc_auc": round(roc_auc_score(y_class, tab_proba), 3),
                     "pr_auc": round(average_precision_score(y_class, tab_proba), 3), "predict_ms": round(tab_clf_s * 1000, 1)})
    clf_metrics = pd.DataFrame(rows)
    mo.vstack([mo.ui.table(clf_metrics, selection=None), mo.md(f"*(test sample: {len(y_class):,} rows, same one for every model; XGBoost trained on {xgb_bundle['n_rows']:,} rows, TabPFN-3.5 on its in-context sample)*")])
    return


@app.cell
def _(base_proba, go, mo, roc_curve, tab_proba, xgb_proba, y_class):
    _fig = go.Figure()
    for _name, _p in [("baseline", base_proba), ("XGBoost", xgb_proba), ("TabPFN-3.5", tab_proba)]:
        if _p is None:
            continue
        _fpr, _tpr, _ = roc_curve(y_class, _p)
        _fig.add_scatter(x=_fpr, y=_tpr, name=_name, mode="lines")
    _fig.add_scatter(x=[0, 1], y=[0, 1], name="chance", line=dict(dash="dash", color="gray"))
    _fig.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR", height=380, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(base_proba, calibration_curve, go, mo, tab_proba, xgb_proba, y_class):
    _fig = go.Figure()
    for _name, _p in [("baseline", base_proba), ("XGBoost", xgb_proba), ("TabPFN-3.5", tab_proba)]:
        if _p is None:
            continue
        _frac_pos, _mean_pred = calibration_curve(y_class, _p, n_bins=10, strategy="quantile")
        _fig.add_scatter(x=_mean_pred, y=_frac_pos, name=_name, mode="lines+markers")
    _fig.add_scatter(x=[0, 1], y=[0, 1], name="perfectly calibrated", line=dict(dash="dash", color="gray"))
    _fig.update_layout(title="Calibration (reliability diagram)", xaxis_title="mean predicted P(overcrowded)", yaxis_title="observed fraction overcrowded", height=380, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Regression: expected passengers
    """)
    return


@app.cell
def _(
    base_expected,
    base_reg_s,
    mean_absolute_error,
    mo,
    pd,
    r2_score,
    tab_expected,
    tab_reg_s,
    xgb_bundle,
    xgb_expected,
    xgb_reg_s,
    y_reg,
):
    rows_reg = [
        {"model": "Historical-average baseline", "mae": round(mean_absolute_error(y_reg, base_expected), 1),
         "r2": round(r2_score(y_reg, base_expected), 3), "predict_ms": round(base_reg_s * 1000, 1)},
        {"model": "XGBoost", "mae": round(mean_absolute_error(y_reg, xgb_expected), 1),
         "r2": round(r2_score(y_reg, xgb_expected), 3), "predict_ms": round(xgb_reg_s * 1000, 1)},
    ]
    if tab_expected is not None:
        rows_reg.append({"model": "TabPFN-3.5", "mae": round(mean_absolute_error(y_reg, tab_expected), 1),
                          "r2": round(r2_score(y_reg, tab_expected), 3), "predict_ms": round(tab_reg_s * 1000, 1)})
    reg_metrics = pd.DataFrame(rows_reg)
    mo.vstack([mo.ui.table(reg_metrics, selection=None), mo.md(f"*(XGBoost trained on {xgb_bundle['n_rows']:,} rows — note MAE alone doesn't show whether either model beats the baseline on genuinely hard rows; see residuals below)*")])
    return


@app.cell
def _(go, mo, tab_expected, xgb_expected, y_reg):
    _fig = go.Figure()
    _fig.add_scatter(x=y_reg, y=xgb_expected, mode="markers", name="XGBoost", marker=dict(size=4, opacity=0.5))
    if tab_expected is not None:
        _fig.add_scatter(x=y_reg, y=tab_expected, mode="markers", name="TabPFN-3.5", marker=dict(size=4, opacity=0.5))
    _max_v = float(max(y_reg.max(), xgb_expected.max()))
    _fig.add_scatter(x=[0, _max_v], y=[0, _max_v], mode="lines", name="perfect", line=dict(dash="dash", color="gray"))
    _fig.update_layout(title="Predicted vs. actual passengers", xaxis_title="actual", yaxis_title="predicted", height=420, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Per-station heterogeneity — is the headline metric hiding anything?
    """)
    return


@app.cell
def _(mo, np, pd, px, tab_expected, test_sample, xgb_expected, y_reg):
    _df = test_sample[["station_name"]].copy()
    _df["abs_err_xgb"] = np.abs(y_reg - xgb_expected)
    if tab_expected is not None:
        _df["abs_err_tabpfn"] = np.abs(y_reg - tab_expected)
    _per_station = _df.groupby("station_name").mean(numeric_only=True)
    _per_station = _per_station[_per_station.index.isin(_df["station_name"].value_counts()[lambda s: s >= 3].index)]
    _top = pd.concat([_per_station.nlargest(8, "abs_err_xgb"), _per_station.nsmallest(8, "abs_err_xgb")])
    _fig = px.bar(_top.reset_index(), x="station_name", y=[c for c in _top.columns], barmode="group",
                  title="Mean |error| by station — 8 worst + 8 best (XGBoost ranking, min 3 test rows)")
    _fig.update_layout(height=420, margin=dict(t=40, b=120), xaxis_tickangle=-45)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Takeaways for an agent deciding which tool to call

    - **Accuracy**: TabPFN-3.5, XGBoost and the historical-average lookup table all land
      within a hair of each other (a reference run: ROC-AUC 0.855 baseline / 0.846 XGBoost /
      0.853 TabPFN-3.5; R² 0.37 / 0.38 / 0.41). The simulated flow is strongly periodic, so
      "≥ own P90" is mostly the daily clock and a one-line lookup table already captures it —
      there is little left for any model to learn. That's the motivation for reframing the
      problem as **anomaly early warning on normalised flows** in `02_anomaly_early_warning.py`.
    - **Latency**: XGBoost predicts in milliseconds, local, no network. TabPFN-3.5 pays a
      network round trip per call (seconds), largely independent of batch size — so an agent
      that needs an instant answer for a dashboard tile should prefer XGBoost; one that needs
      a *decent* answer from a feature table nobody has tuned a model for yet should reach
      for TabPFN-3.5 ("zero-pipeline" is the actual pitch, not raw speed).
    - **Calibration** matters more than ROC-AUC for the early-warning use case specifically —
      an operator acting on "73% chance of overcrowding" needs that 73% to mean something.
      Check the reliability diagram above before trusting either model's probability at face
      value.
    - **Per-station spread**: a single network-wide metric can hide stations where every
      model does badly (new interchanges, stations near the data's edge cases) — worth
      checking before shipping a per-station alert.
    """)
    return


if __name__ == "__main__":
    app.run()
