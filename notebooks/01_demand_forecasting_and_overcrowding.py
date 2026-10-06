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

    This is the use case the MCP server (`mcp_server/server.py`) already exposes as two
    tools, picked by use case:

    - **`predict_overcrowding_risk`** (classification) — *"will this station be at/above
      its own historical 90th-percentile flow in this 15-minute slot?"* — an early-warning
      signal an operator acts on before a platform overfills.
    - **`predict_expected_flow`** (regression) — *"how many passengers are expected?"* —
      operational planning: staffing, train frequency.

    Same feature row (time, weather, events, closure state, station context), two
    specialised model calls. This notebook benchmarks three ways to answer both questions —
    a historical-average **baseline**, a locally-fit **XGBoost** pair, and **TabPFN-3.5** —
    and looks past the headline metric at calibration, per-station spread, and the
    latency/data trade-off that's the actual point of comparing them.
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
    from tabpfn_lab.config import load_dotenv
    from tabpfn_lab.datasets.berlin import (
        FEATURE_COLUMNS, chronological_split, encode_categoricals, stratified_subsample,
    )
    from tabpfn_lab.baselines import (
        fit_classification_baseline, fit_regression_baseline,
        predict_classification_baseline, predict_regression_baseline,
    )
    from tabpfn_lab import closure_impact as ci

    _ = load_dotenv()
    return (
        FEATURE_COLUMNS,
        chronological_split,
        ci,
        encode_categoricals,
        fit_classification_baseline,
        fit_regression_baseline,
        predict_classification_baseline,
        predict_regression_baseline,
        stratified_subsample,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load the feature table (cleaning already applied)

    `build_feature_table` (in `tabpfn_lab/datasets/berlin.py`) is the cleaning +
    preprocessing step for this use case: it melts the wide flow table, repairs mojibake
    in station names, joins weather/events/closures, computes each station's own P90
    threshold, and drops rows with any missing feature — reused here rather than
    reimplemented, so this notebook sees exactly what the MCP server and the Closure
    Impact Lab see.
    """)
    return


@app.cell
def _(chronological_split, ci, mo):
    table = ci.get_feature_table()
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

    `ci.get_xgb_bundle()` and `ci.get_tabpfn_models()` are the exact cached model objects
    the Closure Impact Lab webapp uses (fit once, pickled to `results/model_cache/`), so
    this notebook and that app are never comparing two different models by accident.
    """)
    return


@app.cell
def _(ci, mo):
    xgb_bundle = ci.get_xgb_bundle()
    tabpfn_models, tabpfn_error = ci.get_tabpfn_models()
    mo.md(
        f"""**XGBoost**: fit on {xgb_bundle["n_rows"]:,} rows in {xgb_bundle["fit_seconds"]}s.
        **TabPFN-3.5**: {"fit on a " + str(len(tabpfn_models.train_sample)) + "-row in-context sample in " + str(round(tabpfn_models.fit_seconds, 2)) + "s" if tabpfn_models else "unavailable (" + str(tabpfn_error) + ") — falling back to baseline-only comparisons below"}."""
    )
    return tabpfn_models, xgb_bundle


@app.cell
def _(
    FEATURE_COLUMNS,
    encode_categoricals,
    fit_classification_baseline,
    fit_regression_baseline,
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

    rate_table = fit_classification_baseline(train_pool)
    mean_table = fit_regression_baseline(train_pool)

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
    predict_classification_baseline,
    predict_regression_baseline,
    rate_table,
    test_sample,
    time,
    train_pool,
):
    _t0 = time.perf_counter()
    base_proba = predict_classification_baseline(rate_table, test_sample, float(train_pool["overcrowded"].mean())).to_numpy()
    base_clf_s = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    base_expected = predict_regression_baseline(mean_table, test_sample, float(train_pool["passengers"].mean())).to_numpy()
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

    - **Accuracy**: on this dataset TabPFN-3.5 and XGBoost typically land close together, and
      both are close to the historical-average baseline — the simulated flow is strongly
      periodic, so a one-line lookup table is already most of the way there (same honest
      finding the README reports: ROC-AUC 0.853 baseline vs 0.855 TabPFN-3.5 historically).
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
