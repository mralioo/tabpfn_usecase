import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.metrics import precision_recall_curve

    return go, mo, np, pd, precision_recall_curve


@app.cell
def _(mo):
    mo.md(r"""
    # 04 — Early warning: predict when a station runs *off its normal pattern*

    **The use case, refined.** The Alstom flows are simulated, but they behave like a real
    metro: a strong daily/weekly oscillation (rush hours, a night gap, quiet weekends) with
    a lot of hour-to-hour noise. Notebook `01` predicted *"flow ≥ the station's own 90th
    percentile"* — and a lookup table of (station, weekend, hour) tied TabPFN-3.5 on it
    (ROC-AUC 0.855 vs 0.853), because that target is mostly the clock.

    An operator does not need a model to know 08:00 is busy. They need to know when a
    station will be **busier (or emptier) than a normal 08:00** — because a concert lets
    out, it pours, or a station is shut — early enough to add trains, staff the platform,
    or reroute. So this notebook:

    1. removes the oscillation with a robust **normal profile** per (station, day type, hour),
    2. turns what is left into an **anomaly score z** and finds the outliers,
    3. checks which outliers have a **known driver** (events, closures, weather),
    4. benchmarks how well each model **predicts** those anomalies from information the
       operator has *in advance* (event calendar, planned closures, weather forecast),
    5. replays real **scenarios** from the test weeks the way an operator would see them.

    Engine: `tabpfn_lab/anomaly.py` (same code the webapp's Early-Warning Desk serves).
    Model results are read from `results/anomaly/` — produce them with `make anomaly`.
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
    from tabpfn_lab import anomaly as A
    from tabpfn_lab.config import load_dotenv

    _ = load_dotenv()
    return (A,)


@app.cell
def _():
    # Colour follows the entity everywhere in this notebook (and in the webapp).
    # Categorical slots validated with the dataviz palette checker (light + dark).
    MODEL_COLORS = {
        "TabPFN-3.5 (10k context)": "#2a78d6",
        "XGBoost (full history)": "#eb6834",
        "XGBoost (10k sample)": "#1baf7a",
        "XGBoost (profile only)": "#eda100",
        "Profile baseline": "#8a8984",
    }
    DRIVER_COLORS = {"event": "#e87ba4", "closure": "#4a3aa7", "rain": "#008300", "heat": "#008300", "none": "#b5b4ae"}
    INK, MUTED, GRID = "#3d3c39", "#8a8984", "rgba(138,137,132,0.18)"
    SURGE, DROP = "#d03b3b", "#2a78d6"

    def style(fig, title, height=320, ytitle=None):
        fig.update_layout(
            title=dict(text=title, x=0, font=dict(size=14)), height=height, template="plotly_white",
            margin=dict(l=50, r=20, t=50, b=40), hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
            yaxis_title=ytitle, font=dict(color=INK),
        )
        fig.update_xaxes(showgrid=False, linecolor=GRID)
        fig.update_yaxes(gridcolor=GRID, zeroline=False)
        return fig

    return DRIVER_COLORS, DROP, INK, MODEL_COLORS, MUTED, SURGE, style


@app.cell
def _(A, mo):
    with mo.status.spinner("Building the hourly station table (flows + normal profile + context)…"):
        table = A.anomaly_table("holdout")
    mo.md(
        f"**{len(table):,} station-hours** · {table['station'].nunique()} stations · "
        f"{table['ts'].min():%d %b} → {table['ts'].max():%d %b %Y} · normal profile fit on the "
        f"training split only (before {A.FOLDS['holdout']['start']:%d %b})."
    )
    return (table,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · The oscillation, and what is left when you remove it

    Pick a station and a window. The top chart is the raw hourly flow against the station's
    **normal** for that day type and hour (median of `log1p(flow)` over the training weeks,
    shaded band = ±2 robust standard deviations). The bottom chart is the residual as a
    robust z-score:

    $$z = \frac{\log(1+\text{flow}) - \text{normal}}{1.4826 \cdot \text{MAD}_{\text{station, hour}}}$$

    Log scale so a 30% surge at a small station counts like a 30% surge at a big one; median
    and MAD so the outliers we are hunting don't drag "normal" towards themselves. The spread
    is per station *and hour*: quiet hours (00:00, 05:00) are much noisier than rush hours,
    and a single spread per station would flag every quiet midnight as a "surge".
    Points beyond ±2 are coloured by their known driver.
    """)
    return


@app.cell
def _(mo, pd, table):
    _stations = sorted(table["station"].unique())
    station_pick = mo.ui.dropdown(
        options={s.replace(" (Berlin)", ""): s for s in _stations},
        value="S+U Warschauer Str.", label="Station", searchable=True,
    )
    window_pick = mo.ui.date_range(
        start=table["ts"].min().date(), stop=table["ts"].max().date(),
        value=(pd.Timestamp("2026-09-22").date(), pd.Timestamp("2026-09-30").date()), label="Window",
    )
    mo.hstack([station_pick, window_pick], justify="start", gap=2)
    return station_pick, window_pick


@app.cell
def _(DRIVER_COLORS, DROP, INK, MUTED, SURGE, go, mo, np, pd, station_pick, style, table, window_pick):
    _lo, _hi = (pd.Timestamp(d) for d in window_pick.value)
    _w = table[(table["station"] == station_pick.value) & (table["ts"] >= _lo)
               & (table["ts"] < _hi + pd.Timedelta(days=1))].sort_values("ts")
    _band_hi = np.expm1(_w["normal_level"] + 2 * _w["station_scale"])
    _band_lo = np.expm1(_w["normal_level"] - 2 * _w["station_scale"]).clip(lower=0)

    _f1 = go.Figure()
    _f1.add_trace(go.Scatter(x=_w["ts"], y=_band_hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    _f1.add_trace(go.Scatter(x=_w["ts"], y=_band_lo, fill="tonexty", fillcolor="rgba(138,137,132,0.18)",
                             line=dict(width=0), name="normal ±2σ", hoverinfo="skip"))
    _f1.add_trace(go.Scatter(x=_w["ts"], y=_w["normal_passengers"], name="normal", line=dict(color=MUTED, width=2, dash="dot")))
    _f1.add_trace(go.Scatter(x=_w["ts"], y=_w["passengers"], name="actual", line=dict(color=INK, width=2)))
    style(_f1, f"{station_pick.value.replace(' (Berlin)', '')} — raw hourly flow vs normal", ytitle="passengers / h")

    _f2 = go.Figure()
    _f2.add_hrect(y0=-2, y1=2, fillcolor="rgba(138,137,132,0.10)", line_width=0)
    _f2.add_trace(go.Bar(x=_w["ts"], y=_w["z"], name="z",
                         marker_color=np.where(_w["z"] >= 0, SURGE, DROP), opacity=0.55))
    _out = _w[_w["z"].abs() >= 2]
    for _d, _g in _out.groupby("driver"):
        _f2.add_trace(go.Scatter(x=_g["ts"], y=_g["z"], mode="markers", name=f"|z|≥2 · {_d}",
                                 marker=dict(size=9, color=DRIVER_COLORS[_d], line=dict(color="white", width=2))))
    style(_f2, "Anomaly score z (oscillation removed) — outliers coloured by known driver", ytitle="z")
    mo.vstack([mo.ui.plotly(_f1), mo.ui.plotly(_f2)])
    return


@app.cell
def _(mo, np, table):
    _train = table[table["split"] == "train"]
    _r2 = 1 - np.var(_train["log_p"] - _train["normal_level"]) / np.var(_train["log_p"])
    _out = (table["z"].abs() >= 2).mean()
    mo.hstack([
        mo.stat(f"{_r2:.0%}", label="hourly log-flow variance explained by the normal profile",
                caption="the oscillation — predictable from the clock alone"),
        mo.stat(f"{_out:.1%}", label="station-hours with |z| ≥ 2",
                caption="the outliers — what an operator has to react to"),
        mo.stat(f"{(table['surge'] == 1).mean():.1%}", label="surge hours (z ≥ 2)",
                caption="target of the early-warning classifier"),
    ], widths="equal")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Which outliers have a known cause?

    Training weeks only (no peeking at the test weeks). Each context feature is something
    an operator knows **before** the hour happens: the event calendar (venues mapped to
    the nearest U-Bahn station, `VENUE_TO_STATION` in the engine), planned closures (line
    suspensions resolved to the stations on the suspended segment), and the weather forecast.
    """)
    return


@app.cell
def _(DRIVER_COLORS, go, mo, np, pd, style, table):
    _t = table[table["split"] == "train"]
    _rows = []

    def _add(group, label, mask, color):
        _rows.append({"group": group, "condition": label, "hours": int(mask.sum()),
                      "mean z": float(_t.loc[mask, "z"].mean()), "surge rate": float(_t.loc[mask, "surge"].mean()),
                      "color": color})

    _add("baseline", "no known driver", _t["driver"] == "none", DRIVER_COLORS["none"])
    _eg = _t["event_egress_att"]
    for _lo, _hi in [(1, 1000), (1000, 2000), (2000, 10_000)]:
        _add("event", f"event egress {_lo:,}–{_hi:,} att.", (_eg >= _lo) & (_eg < _hi), DRIVER_COLORS["event"])
    _add("event", "event ingress (start ≤1 h)", _t["event_ingress_att"] > 0, DRIVER_COLORS["event"])
    _add("event", "neighbour station egress", (_t["event_nearby_egress_att"] > 0) & (_eg == 0), DRIVER_COLORS["event"])
    _add("closure", "station closed", _t["station_closed"] == 1, DRIVER_COLORS["closure"])
    _add("closure", "on suspended line segment", _t["line_suspended_here"] == 1, DRIVER_COLORS["closure"])
    _add("closure", "neighbour closed", _t["neighbor_closed"] == 1, DRIVER_COLORS["closure"])
    for _lo, _hi in [(0.1, 1), (1, 3), (3, 100)]:
        _add("weather", f"rain {_lo}–{_hi} mm/h", (_t["prcp"] >= _lo) & (_t["prcp"] < _hi), DRIVER_COLORS["rain"])
    for _lo, _hi in [(28, 32), (32, 45)]:
        _add("weather", f"heat {_lo}–{_hi} °C", (_t["temp"] >= _lo) & (_t["temp"] < _hi), DRIVER_COLORS["heat"])
    driver_effects = pd.DataFrame(_rows)

    _fig = go.Figure(go.Bar(
        y=driver_effects["condition"], x=driver_effects["mean z"], orientation="h",
        marker_color=driver_effects["color"],
        customdata=np.stack([driver_effects["hours"], driver_effects["surge rate"]], axis=1),
        hovertemplate="%{y}<br>mean z %{x:.2f}<br>%{customdata[0]:,} station-hours<br>surge rate %{customdata[1]:.1%}<extra></extra>",
    ))
    style(_fig, "Mean anomaly z by driver (training weeks)", height=440)
    _fig.update_layout(hovermode="closest", yaxis=dict(autorange="reversed"))
    _fig.add_vline(x=0, line_color="#8a8984", line_width=1)
    mo.vstack([
        mo.ui.plotly(_fig),
        mo.ui.table(driver_effects.drop(columns="color").round(3), selection=None, pagination=False),
        mo.md(r"""
    **What the simulation contains** — and therefore what a model *can* learn:

    - **Event egress is the big surge driver**: shows of 1.5k+ attendees push the venue
      station about 2σ above normal on average in the hour the show ends — over half of
      those hours are surges, against ~1% of ordinary hours. Ingress is much weaker
      (arrivals spread out), neighbouring stations barely move.
    - **Station closures collapse flow** at the closed station (z ≈ −5 and below). Line
      suspensions do *not* move their endpoints, and there is no spill-over surge to
      neighbours in this simulation — a real network would show one.
    - **Weather is a soft, network-wide shift**: rain lifts demand, heat suppresses it. It
      changes the surge *rate* but rarely creates an alarm-level spike on its own.
    - Most |z| ≥ 2 hours have **no known driver** — simulation noise. No model can predict
      those from context, which caps every model's overall precision. The fair test is how
      many of the *explainable* anomalies each model catches.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Can a model see them coming?

    Every model gets the same test weeks and the same information:

    | Model | Features | Training rows |
    | --- | --- | --- |
    | **Profile baseline** | none — "tomorrow is a normal day" (historical surge rate per station/day type/hour) | all history |
    | **XGBoost (profile only)** | clock + station descriptors | all history |
    | **XGBoost (full history)** | profile + context | all history (~280–350k) |
    | **XGBoost (10k sample)** | profile + context | the same 10k rows TabPFN sees |
    | **TabPFN-3.5 (10k context)** | profile + context | 10k-row in-context sample, no training loop |

    Two expanding-window folds: the **Alstom hold-out** (`*_rest` files, Sep 22–30, the
    InnoTrans week) and a **backtest** (Sep 1–21) that adds closures, rain and heat the short
    hold-out lacks. An operator can act on a limited number of alarms, so besides PR-AUC
    we score a fixed **alarm budget: the top 2% of station-hours** each model flags.
    """)
    return


@app.cell
def _(A, mo):
    try:
        cached = A.load_cached()
    except FileNotFoundError:
        cached = None
    mo.stop(cached is None, mo.callout(mo.md(
        "No cached results in `results/anomaly/`. Run `make anomaly` (≈4 min with live TabPFN-3.5, "
        "≈30 s with `make anomaly-offline`, XGBoost only) and re-run this cell."), kind="warn"))
    metrics, scenarios, preds = cached["metrics"], cached["scenarios"], cached["predictions"]
    fold_pick = mo.ui.dropdown(
        options={"Both folds pooled": "pooled", **{v["label"]: k for k, v in metrics["folds"].items()}},
        value="Both folds pooled", label="Evaluation window",
    )
    mo.vstack([
        mo.md(f"Results from `{metrics['created']}` · live TabPFN-3.5: **{metrics['live_tabpfn']}**"),
        fold_pick,
    ])
    return fold_pick, metrics, preds, scenarios


@app.cell
def _(fold_pick, metrics, mo, pd):
    _src = metrics["pooled"] if fold_pick.value == "pooled" else metrics["folds"][fold_pick.value]
    score_table = pd.DataFrame(_src["models"]).T
    _cols = {
        "pr_auc": "PR-AUC (surge)", "roc_auc": "ROC-AUC", "alarm_precision": "alarm precision",
        "alarm_recall": "alarm recall", "recall_event": "event surges caught",
        "recall_explained": "explained surges caught", "closure_drop_recall": "closure collapses caught",
        "z_mae_context": "z MAE (driver hours)", "z_mae": "z MAE (all)",
    }
    _show = score_table[[c for c in _cols if c in score_table]].rename(columns=_cols).astype(float).round(3)
    if fold_pick.value != "pooled":
        _show["train rows"] = score_table["n_train"].astype(int)
        _show["fit s"] = score_table["fit_s"].astype(float).round(1)
        _show["predict s"] = score_table["predict_s"].astype(float).round(1)
    _n = _src["n_test_rows"]
    mo.vstack([
        mo.ui.table(_show.reset_index(names="model"), selection=None, pagination=False),
        mo.md(f"*{_n:,} test station-hours · surge rate {_src['test_surge_rate']:.1%} · "
              f"alarm budget = top {metrics['definitions']['alarm_budget']:.0%} of station-hours per model.*"),
    ])
    return (score_table,)


@app.cell
def _(MODEL_COLORS, fold_pick, go, metrics, mo, precision_recall_curve, preds, style):
    _p = preds if fold_pick.value == "pooled" else preds[preds["fold"] == fold_pick.value]
    _keys = {n: v["key"] for n, v in metrics["folds"]["holdout"]["models"].items()}
    _fig = go.Figure()
    for _name, _key in _keys.items():
        _prec, _rec, _ = precision_recall_curve(_p["surge"], _p[f"p_{_key}"])
        _fig.add_trace(go.Scatter(x=_rec, y=_prec, name=_name, mode="lines",
                                  line=dict(color=MODEL_COLORS[_name], width=2)))
    _fig.add_hline(y=_p["surge"].mean(), line_dash="dot", line_color="#8a8984",
                   annotation_text="no-skill", annotation_position="bottom right")
    style(_fig, "Precision–recall for surge hours", height=360, ytitle="precision")
    _fig.update_layout(hovermode="closest", xaxis_title="recall", xaxis_range=[0, 1], yaxis_range=[0, 1])
    mo.ui.plotly(_fig)
    return


@app.cell
def _(MODEL_COLORS, go, mo, score_table, style):
    _metrics = {"event surges caught": "recall_event", "explained surges caught": "recall_explained",
                "closure collapses caught": "closure_drop_recall", "all surges caught": "alarm_recall"}
    _fig = go.Figure()
    for _name in score_table.index:
        _fig.add_trace(go.Bar(name=_name, x=list(_metrics), marker_color=MODEL_COLORS[_name],
                              y=[float(score_table.loc[_name, c] or 0) for c in _metrics.values()]))
    style(_fig, "Recall inside the 2% alarm budget, by kind of anomaly", height=360, ytitle="share caught")
    _fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="closest", yaxis_range=[0, 1])
    mo.vstack([
        mo.ui.plotly(_fig),
        mo.md(r"""
    How to read this:

    - The **profile baseline** and **profile-only XGBoost** cannot warn about anything —
      they only know the clock, so they flag the noisiest *usual* hours. This is the
      "overcrowding = p90" model from notebook 01 in disguise.
    - Adding **context** is what turns a demand model into an early-warning model.
    - **At equal data, TabPFN-3.5 wins**: on the *same 10k rows* it beats XGBoost on PR-AUC
      and catches roughly 1.5× as many event surges — with no training loop, no
      hyper-parameters and no per-task pipeline. The learning curve below shows how much
      more history XGBoost needs to catch up.
    - **With the full history (28–35× more rows), XGBoost pulls ahead.** For a mature
      network with months of clean labelled data, a trained gradient-boosted model is still
      the stronger production model for this exact task.
    - Rain and heat are soft drivers: they shift the whole network by a fraction of a σ, so
      no model turns them into many alarm-level hits. They still matter for the *expected*
      flow (regression z).
        """),
    ])
    return


@app.cell
def _(MODEL_COLORS, fold_pick, go, metrics, mo, style):
    _folds = list(metrics["folds"]) if fold_pick.value == "pooled" else [fold_pick.value]
    _fig = go.Figure()
    _notes = []
    for _f in _folds:
        _fm = metrics["folds"][_f]
        _curve = _fm.get("xgboost_learning_curve") or []
        _tab = _fm["models"].get("TabPFN-3.5 (10k context)")
        _short = "hold-out" if _f == "holdout" else "backtest"
        _dash = "solid" if _f == "holdout" else "dot"
        _fig.add_trace(go.Scatter(x=[c["n_train"] for c in _curve], y=[c["pr_auc"] for c in _curve],
                                  mode="lines+markers", name=f"XGBoost · {_short}",
                                  line=dict(color=MODEL_COLORS["XGBoost (full history)"], width=2, dash=_dash),
                                  marker=dict(size=8)))
        if _tab:
            _fig.add_trace(go.Scatter(x=[_tab["n_train"]], y=[_tab["pr_auc"]], mode="markers+text",
                                      name=f"TabPFN-3.5 · {_short}", text=["TabPFN-3.5"], textposition="top center",
                                      marker=dict(size=13, symbol="diamond", color=MODEL_COLORS["TabPFN-3.5 (10k context)"],
                                                  line=dict(color="white", width=2))))
            _match = next((c["n_train"] for c in _curve if c["pr_auc"] >= _tab["pr_auc"]), None)
            _notes.append(f"**{_short}**: XGBoost needs ~**{_match:,}** rows to match TabPFN-3.5's PR-AUC "
                          f"{_tab['pr_auc']:.3f} from {_tab['n_train']:,}" if _match else
                          f"**{_short}**: XGBoost never matches TabPFN-3.5 ({_tab['pr_auc']:.3f})")
    style(_fig, "How much history does XGBoost need? (PR-AUC vs training rows, log scale)", height=360, ytitle="PR-AUC (surge)")
    _fig.update_layout(hovermode="closest", xaxis_type="log", xaxis_title="training rows (station-hours)")
    mo.vstack([mo.ui.plotly(_fig), mo.md(" · ".join(_notes) + ". Same context-aware sampling for every size "
                                         "below the full history, same features, same test window.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Scenario replay — what the operator would have seen

    Each scenario is a real event, closure or rain spell inside a test window. The forecast
    for every hour uses only day-ahead information, so an alarm shown here is an alarm the
    control room would have had **before** the hour started. Sorted by the size of the
    actual anomaly.
    """)
    return


@app.cell
def _(mo, scenarios):
    _ranked = sorted(scenarios, key=lambda s: -abs(s["peak"]["z"]))
    _opts = {}
    for _s in _ranked:
        _label = f"{'↑' if _s['peak']['z'] > 0 else '↓'} z {_s['peak']['z']:+.1f} · {_s['kind']} · {_s['title'][:70]} ({_s['fold']})"
        _opts[_label] = _s["id"]
    scenario_pick = mo.ui.dropdown(options=_opts, value=next(iter(_opts)), label="Scenario", searchable=True)
    scenario_pick
    return (scenario_pick,)


@app.cell
def _(DROP, INK, MODEL_COLORS, MUTED, SURGE, go, metrics, mo, np, pd, scenario_pick, scenarios, style):
    scenario = next(s for s in scenarios if s["id"] == scenario_pick.value)
    _df = pd.DataFrame(scenario["series"])
    _df["ts"] = pd.to_datetime(_df["ts"])
    _fold = metrics["folds"][scenario["fold"]]
    _models = _fold["models"]

    _f1 = go.Figure()
    _f1.add_trace(go.Scatter(x=_df["ts"], y=_df["normal"], name="normal", line=dict(color=MUTED, width=2, dash="dot")))
    _f1.add_trace(go.Scatter(x=_df["ts"], y=_df["actual"], name="actual", line=dict(color=INK, width=2)))
    style(_f1, f"{scenario['station_short']} — flow", height=260, ytitle="passengers / h")

    _f2 = go.Figure()
    _f2.add_trace(go.Bar(x=_df["ts"], y=_df["z"], name="actual z",
                         marker_color=np.where(_df["z"] >= 0, SURGE, DROP), opacity=0.35))
    for _name, _m in _models.items():
        if _m["key"] == "profile":
            continue
        _f2.add_trace(go.Scatter(x=_df["ts"], y=_df[f"z_hat_{_m['key']}"], name=_name, mode="lines+markers",
                                 line=dict(color=MODEL_COLORS[_name], width=2), marker=dict(size=8)))
    style(_f2, "Forecast anomaly z vs what happened", height=280, ytitle="z")

    _f3 = go.Figure()
    _alarm_rows = []
    for _name, _m in _models.items():
        _col = f"p_{_m['key']}"
        _rel = _df[_col] / max(_m["alarm_threshold"], 1e-9)
        _f3.add_trace(go.Scatter(x=_df["ts"], y=_rel, name=_name, mode="lines+markers",
                                 line=dict(color=MODEL_COLORS[_name], width=2), marker=dict(size=8)))
        _hits = _df.loc[_df[_col] >= _m["alarm_threshold"], "ts"]
        _alarm_rows.append({"model": _name, "alarm hours": ", ".join(f"{t:%a %H:%M}" for t in _hits) or "—",
                            "max surge prob": round(float(_df[_col].max()), 3)})
    _f3.add_hline(y=1, line_dash="dash", line_color=SURGE, annotation_text="alarm threshold (top 2%)",
                  annotation_position="top left")
    style(_f3, "Surge alarm score ÷ that model's alarm threshold (≥ 1 → alarm raised)", height=280, ytitle="× threshold")

    _peak = scenario["peak"]
    mo.vstack([
        mo.callout(mo.md(f"**{scenario['title']}**  \n{scenario['cause']}  \n"
                         f"Peak anomaly **z {_peak['z']:+.1f}** at {pd.Timestamp(_peak['ts']):%a %d %b %H:%M} — "
                         f"{_peak['actual']:,.0f} passengers vs {_peak['normal']:,.0f} normal."),
                   kind="danger" if abs(_peak["z"]) >= 2 else "info"),
        mo.ui.plotly(_f1), mo.ui.plotly(_f2), mo.ui.plotly(_f3),
        mo.ui.table(pd.DataFrame(_alarm_rows), selection=None, pagination=False),
    ])
    return (scenario,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Alarm feed — the hold-out week as a control-room list

    The top TabPFN-3.5 alarms of the Alstom hold-out week, with the driver the context
    features point to and whether a surge actually happened. This is the list the
    webapp's Early-Warning Desk shows on the right rail.
    """)
    return


@app.cell
def _(metrics, mo, preds):
    _hold = preds[preds["fold"] == "holdout"].copy()
    _thr = metrics["folds"]["holdout"]["models"]["TabPFN-3.5 (10k context)"]["alarm_threshold"] \
        if "TabPFN-3.5 (10k context)" in metrics["folds"]["holdout"]["models"] else None
    mo.stop(_thr is None, mo.md("*TabPFN-3.5 results not in the cache (offline run).*"))
    _feed = _hold[_hold["p_tabpfn"] >= _thr].sort_values("p_tabpfn", ascending=False)
    _feed = _feed.assign(
        station=_feed["station"].str.replace(" (Berlin)", "", regex=False),
        hour=_feed["ts"].dt.strftime("%a %d %b %H:%M"),
        outcome=(_feed["surge"].map({1: "surge ✓", 0: "no surge"})),
    )[["hour", "station", "driver", "p_tabpfn", "z_hat_tabpfn", "z", "outcome", "event_egress_att", "station_closed", "prcp"]]
    _hit = (_feed["outcome"] == "surge ✓").mean()
    _hit_ctx = (_feed.loc[_feed["driver"] != "none", "outcome"] == "surge ✓").mean()
    mo.vstack([
        mo.hstack([
            mo.stat(f"{len(_feed):,}", label="alarms in the hold-out week", caption="top 2% of station-hours"),
            mo.stat(f"{_hit:.0%}", label="alarms followed by a surge", caption=f"vs {_hold['surge'].mean():.1%} base rate"),
            mo.stat(f"{_hit_ctx:.0%}", label="hit rate when a driver is named", caption="event / closure / weather alarms"),
        ], widths="equal"),
        mo.ui.table(_feed.round(3), selection=None, page_size=15),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 · Live TabPFN-3.5 re-score (optional, uses API quota)

    Re-fits TabPFN-3.5 on the fold's 10k-row context sample and scores just the selected
    scenario's hours — the call an operator tool would make for "tomorrow at this station".
    """)
    return


@app.cell
def _(mo):
    live_button = mo.ui.run_button(label="Re-score this scenario live with TabPFN-3.5")
    live_button
    return (live_button,)


@app.cell
def _(A, live_button, mo, pd, scenario):
    mo.stop(not live_button.value, mo.md("*Press the button to call TabPFN-3.5.*"))
    import time as _time

    _tbl = A.anomaly_table(scenario["fold"])
    _train = _tbl[_tbl["split"] == "train"]
    _rows = _tbl[(_tbl["split"] == "test") & (_tbl["ts"] >= pd.Timestamp(scenario["start"]).floor("h"))
                 & (_tbl["ts"] <= pd.Timestamp(scenario["end"]).ceil("h"))]
    if scenario["station"]:
        _rows = _rows[_rows["station"] == scenario["station"]]
    with mo.status.spinner(f"TabPFN-3.5: fit on 10k context rows, predict {len(_rows)} hours…"):
        _t0 = _time.perf_counter()
        _bundle = A.fit_tabpfn(A.context_sample(_train))
        _p = A.predict_tabpfn(_bundle, _rows)
        _secs = _time.perf_counter() - _t0
    _out = _rows[["ts", "station", "passengers", "normal_passengers", "z"]].assign(
        z_hat_live=_p["z_hat"].round(2), p_surge_live=_p["p_surge"].round(3))
    mo.vstack([mo.md(f"Live round-trip **{_secs:.1f} s** (fit {_bundle['fit_s']:.1f} s, predict "
                     f"{_p.attrs['predict_s']:.1f} s for {len(_rows)} rows)."),
               mo.ui.table(_out.round(2), selection=None)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · Across the network — where the load goes, hour by hour

    Station forecasts become network forecasts when they are read along the topology. For a
    chosen hour, `tabpfn_lab/operations.py` aggregates the TabPFN-3.5 forecast per line (load vs
    normal) and lays one line out in running order, terminus to terminus — the **line
    corridor** — so you see where along the line pressure builds (event egress) or collapses
    (closure). These are the same functions the MCP tools `network_outlook` and `line_corridor`
    expose to the agent (webapp page `/agent`).
    """)
    return


@app.cell
def _(A, mo, pd):
    from tabpfn_lab import operations as ops

    _hours = sorted(A.anomaly_table("holdout").query("split == 'test'")["ts"].unique())
    hour_pick = mo.ui.dropdown(options={pd.Timestamp(h).strftime("%a %d %b %H:00"): pd.Timestamp(h).isoformat() for h in _hours},
                               value="Mon 28 Sep 22:00", label="Hour (hold-out week)", searchable=True)
    line_pick = mo.ui.dropdown(options=list(ops.line_sequences()), value="U1", label="Line")
    mo.hstack([hour_pick, line_pick], justify="start", gap=2)
    return hour_pick, line_pick, ops


@app.cell
def _(INK, MODEL_COLORS, SURGE, go, hour_pick, line_pick, mo, ops, pd, style):
    outlook = ops.network_outlook(hour_pick.value)
    corridor = ops.line_corridor(line_pick.value, hour_pick.value)
    _lines = pd.DataFrame(outlook["lines"])
    _stops = pd.DataFrame(corridor["stops"])
    _fig = go.Figure()
    _fig.add_trace(go.Bar(x=_stops["station"].str.replace(r"^(S\+U|U) ", "", regex=True), y=_stops["forecast_passengers"],
                          name="TabPFN-3.5 forecast",
                          marker_color=[SURGE if a else MODEL_COLORS["TabPFN-3.5 (10k context)"] for a in _stops["alarm"]]))
    _fig.add_trace(go.Scatter(x=_stops["station"].str.replace(r"^(S\+U|U) ", "", regex=True), y=_stops["normal_passengers"],
                              name="normal", mode="markers", marker=dict(symbol="line-ew-open", size=18, color=INK, line=dict(width=3))))
    _fig.add_trace(go.Scatter(x=_stops["station"].str.replace(r"^(S\+U|U) ", "", regex=True), y=_stops["observed_replay_passengers"],
                              name="observed (replay)", mode="markers", marker=dict(size=7, color="#8a8984")))
    style(_fig, f"{corridor['line']} corridor {' → '.join(corridor['terminus_to_terminus'])} — {pd.Timestamp(hour_pick.value):%a %d %b %H:00} (red = surge alarm)",
          height=380, ytitle="passengers / h")
    _fig.update_layout(hovermode="x", bargap=0.25)
    _net = outlook["network"]
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_net['forecast_passengers']:,}", label="network forecast, riders/h",
                    caption=f"normal {_net['normal_passengers']:,} ({100 * (_net['forecast_passengers'] / max(_net['normal_passengers'], 1) - 1):+.0f}%)"),
            mo.stat(f"{_net['alarms']}", label="surge alarms (TabPFN-3.5)", caption=f"{outlook['observed_replay']['alarms_that_surged']} surged in reality"),
            mo.stat(f"{outlook['weather']['temp_c']} °C · {outlook['weather']['rain_mm']} mm", label="weather"),
        ], widths="equal"),
        mo.ui.plotly(_fig),
        mo.md("**Load vs normal per line** (sum of the TabPFN-3.5 station forecasts along each line):"),
        mo.ui.table(_lines, selection=None, pagination=False),
        mo.md("**Alarms this hour, with drivers:**"),
        mo.ui.table(pd.DataFrame(outlook["top_alarms"]).drop(columns=["observed_replay"]), selection=None, pagination=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What-if, live: TabPFN-3.5 as a counterfactual engine

    The same context features let an operator ask about a situation that is *not* in the
    calendar: an extra event, a closure, rain. `ops.what_if` scores the station-hours twice in
    one TabPFN-3.5 call — as scheduled and with the hypothetical context — and for a closure it
    scores the neighbouring stations too.
    """)
    return


@app.cell
def _(mo):
    wi_station = mo.ui.text(value="Olympia-Stadion", label="Station")
    wi_start = mo.ui.text(value="2026-09-29 19:00", label="From")
    wi_att = mo.ui.number(start=0, stop=60000, step=500, value=3000, label="Event attendance")
    wi_end = mo.ui.text(value="2026-09-29 22:00", label="Event ends")
    wi_close = mo.ui.checkbox(label="Close the station")
    wi_go = mo.ui.run_button(label="Run what-if with TabPFN-3.5 (live API)")
    mo.vstack([mo.hstack([wi_station, wi_start, wi_att, wi_end, wi_close], justify="start", wrap=True), wi_go])
    return wi_att, wi_close, wi_end, wi_go, wi_start, wi_station


@app.cell
def _(mo, ops, pd, wi_att, wi_close, wi_end, wi_go, wi_start, wi_station):
    mo.stop(not wi_go.value, mo.md("*Press the button to call TabPFN-3.5.*"))
    with mo.status.spinner("TabPFN-3.5: scoring as-scheduled and what-if…"):
        whatif = ops.what_if(wi_station.value, wi_start.value, hours=5, event_attendance=int(wi_att.value),
                             event_end=wi_end.value or None, close_station=wi_close.value)
    mo.stop("error" in whatif, mo.callout(mo.md(whatif.get("error", "")), kind="warn"))
    _rows = pd.DataFrame([{"station": h["station"], "hour": pd.Timestamp(h["hour"]).strftime("%a %H:00"),
                           "normal": h["normal_passengers"], "as scheduled": h["as_scheduled"]["forecast_passengers"],
                           "what-if": h["what_if"]["forecast_passengers"], "Δ riders": h["delta_passengers"],
                           "what-if surge prob.": h["what_if"]["surge_probability"], "alarm": h["what_if"]["alarm"]}
                          for h in whatif["hours"]])
    _pk = whatif["peak_change"]
    mo.vstack([
        mo.callout(mo.md(f"**{whatif['scenario']}** → peak change at {_pk['station']} {pd.Timestamp(_pk['hour']):%H:00}: "
                         f"{_pk['as_scheduled']['forecast_passengers']:,} → **{_pk['what_if']['forecast_passengers']:,}** riders/h "
                         f"(normal {_pk['normal_passengers']:,}), surge probability {_pk['what_if']['surge_probability']:.0%}."
                         f"  \n*{whatif['inference']['engine']} · fit {whatif['inference']['fit_s']} s · predict {whatif['inference']['predict_s']} s*"),
                   kind="danger" if _pk["what_if"]["alarm"] else "info"),
        mo.ui.table(_rows, selection=None, pagination=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 · The control-room agent

    Everything above is also available to an LLM agent as MCP tools (`mcp_server/server.py`):
    `network_outlook`, `line_corridor`, `station_forecast`, `explain_anomaly`, `what_if`,
    `list_scenarios`, `model_scoreboard`. The agent (`agent/harness.py`, any LiteLLM model)
    picks the tools, the tools run TabPFN-3.5, and the answer is written from the returned
    numbers — with the tool trace below it. Needs `AGENT_LITELLM_MODEL` + its API key in `.env`.
    """)
    return


@app.cell
def _(mo):
    agent_q = mo.ui.text_area(value="A 3,000-person concert ends at Olympia-Stadion on 29 September at 22:00. "
                                    "What happens and what should we do?", label="Question", full_width=True)
    agent_go = mo.ui.run_button(label="Ask the agent (LLM + live TabPFN-3.5)")
    mo.vstack([agent_q, agent_go])
    return agent_go, agent_q


@app.cell
async def _(agent_go, agent_q, mo, pd):
    mo.stop(not agent_go.value, mo.md("*Press the button to run the agent.*"))
    from agent.harness import ask as _ask

    with mo.status.spinner("Agent is calling MCP tools…"):
        _out = await _ask(agent_q.value)
    _trace = pd.DataFrame([{"tool": t["tool"], "args": str(t["args"]), "seconds": t["seconds"]}
                           for t in _out["trace"] if t["step"] == "tool"])
    mo.vstack([mo.md(_out["answer"]), mo.md(f"*{_out['model']} · {_out['seconds']} s total*"),
               mo.ui.table(_trace, selection=None, pagination=False)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Normalise first.** Over 80% of hourly log-flow variance is the clock. Removing it with a
      robust (median/MAD) station-hour profile turns "busy" into "unusually busy", which is
      the signal an operator acts on — and the one a lookup table cannot provide.
    - **The anomalies with a cause are predictable a day ahead**: context-aware models flag
      station-closure collapses almost every time and catch far more event-egress surges
      than the clock-only models. Rain and heat are soft, network-wide shifts.
    - **Where TabPFN-3.5 fits**: from a 10k-row context with no training pipeline it beats
      XGBoost trained on the same rows, and XGBoost needs about 5× more history to match it.
      With the full history, a trained XGBoost is better. So: TabPFN-3.5 for short-history
      situations (new station, new venue, rebuilt line, a new question that needs an answer
      today); a trained GBM once months of labelled data exist. Both use the same feature
      table, so the switch costs nothing.
    - **Latency**: ~5 s to fit, ~1–2 s per 1,000 station-hours to score through the API —
      fine for a day-ahead or hour-ahead plan (168 stations × 20 h ≈ 3,400 rows), not for a
      sub-second loop. XGBoost scores in milliseconds once trained.

    **Limitations.** Flows, closures and energy are Alstom-simulated; most |z| ≥ 2 hours
    are noise with no driver, which caps precision for every model. The venue→station map is
    hand-curated (`VENUE_TO_STATION`). The hold-out week has only a handful of closure
    hours, hence the second backtest fold. No spill-over to neighbouring stations exists in
    the simulation, so the model cannot learn reroute surges a real network would show.
    """)
    return


if __name__ == "__main__":
    app.run()
