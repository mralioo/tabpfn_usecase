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

    return go, mo, pd, px


@app.cell
def _(mo):
    mo.md(r"""
    # 02 — Closure impact & network resilience: "what happens if X closes?"

    The third distinct question this dataset supports, and the one with no simple lookup-table
    answer: an operator asks *"what if {station} closes?"* or *"what if {line} is suspended
    between A and B?"* — a question about the **network graph**, not just one station's own
    history. This is exactly what the Closure Impact Lab (`webapp/`) answers live for one
    scenario at a time; this notebook does the same scoring (`tabpfn_lab.closure_impact`,
    unchanged) but **systematically across every closure actually recorded** in
    `closures_pre_innotrans.csv` / `closures_rest.csv`, plus pure graph-theoretic resilience
    analysis (articulation points, betweenness) that needs no model at all.
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
    from tabpfn_lab import closure_impact as ci

    _ = load_dotenv()
    folder = bl.discover_dataset_dir(BERLIN_DATA_DIR)
    return bl, ci, folder


@app.cell
def _(mo):
    mo.md("""
    ## 1. The graph — no model needed for this part

    `tabpfn_lab.datasets.berlin.build_graph` turns `berlin_ubahn_connections.csv` into a
    `networkx.Graph`; `network_resilience` then ranks every station by whether removing it
    **disconnects** the network (an articulation point — no bypass exists) and by
    **betweenness centrality** (how much shortest-path traffic routes through it), joined with
    real average daily ridership. Pure topology + the real flow table, zero ML.
    """)
    return


@app.cell
def _(bl, folder, mo):
    resilience = bl.network_resilience(folder)
    n_articulation = int(resilience["is_articulation_point"].sum())
    mo.md(
        f"""**{n_articulation} of {len(resilience)} stations are articulation points** — closing
        any one of them, with nothing else changing, fragments the network (a train can no longer
        reach some stations at all, not just with a detour)."""
    )
    return (resilience,)


@app.cell
def _(mo, resilience):
    mo.ui.table(
        resilience.sort_values("fragmentation_score", ascending=False).head(15)
        [["station_name", "is_articulation_point", "betweenness_centrality", "resulting_components", "avg_daily_passengers", "fragmentation_score"]],
        selection=None,
    )
    return


@app.cell
def _(mo, px, resilience):
    _fig = px.scatter(
        resilience, x="betweenness_centrality", y="avg_daily_passengers",
        color="is_articulation_point", hover_name="station_name",
        title="Every station: graph centrality vs. real ridership (color = articulation point)",
        labels={"betweenness_centrality": "betweenness centrality", "avg_daily_passengers": "avg daily passengers"},
    )
    _fig.update_layout(height=440, margin=dict(t=40, b=20))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    **Reading this chart**: top-right is the dangerous quadrant — high centrality *and* high
    ridership. Top-left (articulation point, lower ridership) is structurally fragile but
    lower-consequence; bottom-right (high ridership, not an articulation point) is busy but
    has a bypass. The model-based scoring below picks up where this leaves off: *how much*
    flow redistributes nearby, not just *whether* the network survives.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Systematic sweep: every recorded closure, scored with XGBoost
    """)
    return


@app.cell
def _(ci, mo):
    topo = ci.get_topology()
    mo.md(
        f"""{len(topo["scenarios"])} closures from the dataset resolve to a known station/line —
        scoring every one with XGBoost (local, no network call) to rank them by impact before
        picking any single one to look at with TabPFN-3.5."""
    )
    return (topo,)


@app.cell
def _(ci, mo, pd, topo):
    _rows = []
    for _s in topo["scenarios"]:
        _r = ci.score_with_xgboost(_s)
        _top_p = max((row["xgb"]["overcrowding_probability"] for row in _r["rows"]), default=0.0)
        _rows.append({
            "scenario": _s["text"], "kind": _s["kind"],
            "affected_count": _r["affected_count"],
            "is_articulation_point": _r["is_articulation_point"],
            "max_P_overcrowded_nearby": round(_top_p, 3),
        })
    sweep = pd.DataFrame(_rows).sort_values(["is_articulation_point", "affected_count"], ascending=[False, False])
    mo.ui.table(sweep, selection=None)
    return (sweep,)


@app.cell
def _(mo, px, sweep):
    _fig = px.bar(
        sweep.sort_values("affected_count", ascending=False).head(15),
        x="scenario", y="affected_count", color="is_articulation_point",
        title="Top 15 closures by stations affected within 2 hops",
    )
    _fig.update_layout(height=420, margin=dict(t=40, b=180), xaxis_tickangle=-60)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Deep dive: pick one closure, compare XGBoost vs. TabPFN-3.5

    This is the one call in the notebook that reaches TabPFN-3.5's real API (a single batched
    request for all affected stations — see `tabpfn_lab/closure_impact.py`'s module docstring
    for why batching matters here). Everything above was XGBoost-only on purpose: a systematic
    30-scenario sweep is exactly the kind of fast, cheap, "screen everything" job a local model
    is for; the specialised call is for the one scenario an operator actually wants a second
    opinion on.
    """)
    return


@app.cell
def _(mo, topo):
    scenario_picker = mo.ui.dropdown(
        options={s["text"]: i for i, s in enumerate(topo["scenarios"])},
        value=topo["scenarios"][0]["text"],
        label="Closure scenario",
    )
    scenario_picker
    return (scenario_picker,)


@app.cell
def _(ci, scenario_picker, topo):
    _scenario = topo["scenarios"][scenario_picker.value]
    xgb_result = ci.score_with_xgboost(_scenario)
    tabpfn_result = ci.score_with_tabpfn(_scenario, live_tabpfn=True)
    return tabpfn_result, xgb_result


@app.cell
def _(mo, tabpfn_result, xgb_result):
    _tab_by_id = {r["id"]: r["tabpfn"] for r in tabpfn_result["rows"]}
    mo.md(
        f"""**{xgb_result["affected_count"]} stations** within 2 hops
        {"— **critical node**, no bypass exists" if xgb_result["is_articulation_point"] else ""}.
        XGBoost: local, {xgb_result["xgb_meta"]["fit_seconds"]}s one-time fit on
        {xgb_result["xgb_meta"]["train_rows"]:,} rows. TabPFN-3.5:
        {"live API call, " + str(tabpfn_result["tabpfn_meta"]["context_rows"]) + "-row context" if tabpfn_result["tabpfn_meta"]["available"] else "unavailable this run — showing labelled simulated fallback (" + str(tabpfn_result["tabpfn_meta"]["error"]) + ")"}."""
    )
    return


@app.cell
def _(go, mo, tabpfn_result, xgb_result):
    _tab_by_id = {r["id"]: r["tabpfn"] for r in tabpfn_result["rows"]}
    _names = [r["name"] for r in xgb_result["rows"]]
    _xgb_p = [r["xgb"]["overcrowding_probability"] for r in xgb_result["rows"]]
    _tab_p = [_tab_by_id.get(r["id"], {}).get("overcrowding_probability", 0) for r in xgb_result["rows"]]
    _fig = go.Figure()
    _fig.add_bar(x=_names, y=_xgb_p, name="XGBoost")
    _fig.add_bar(x=_names, y=_tab_p, name="TabPFN-3.5")
    _fig.update_layout(
        barmode="group", title="P(≥ own P90 load) at affected stations",
        height=380, margin=dict(t=40, b=100), xaxis_tickangle=-30, yaxis_title="probability",
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Takeaways

    - **Graph structure and model predictions answer different questions.** An articulation
      point can have *low* predicted overcrowding risk nearby (few riders, but no detour) and
      a *non*-articulation-point hub can show high redistributed risk despite the network
      staying connected. An operator-facing tool needs both, not just one.
    - **Tiered tool use is the efficient pattern**: XGBoost for a cheap, local sweep across
      every scenario worth watching; TabPFN-3.5 for the one case that just got flagged and
      needs a second, differently-built model's opinion before paging someone. Reaching for
      the specialised model on all 30 scenarios would have cost 30 real network round trips
      for a question a free `networkx` call mostly already answered.
    - **This dataset's closures are simulated**, but the graph and ridership they're scored
      against are real — so the *ranking* of which closures matter most is trustworthy even
      where the exact probability numbers are illustrative.
    """)
    return


if __name__ == "__main__":
    app.run()
