"""What-if closure impact engine behind the Closure Impact Lab web app (`webapp/`).

Reuses the project's existing Berlin pipeline end to end rather than standing up a
parallel one: `datasets/berlin.py` for the feature table, graph and closure parsing,
`models.py` for TabPFN-3.5 (classifier + regressor, same `predict_row` the MCP server
calls), and a newly-fit XGBoost pair (`xgboost_baseline.py` was Deutsche-Bahn-only
before this) as the baseline this dashboard compares it against.

For an affected station, "what if X closes" is modelled as: take that station's real
historical feature row nearest the closure's start time, and bump
`network_active_closures` by one (a live incident elsewhere on the network) — the
closed station/segment's own row additionally gets `in_closure=1`. Both fitted models
score that row; the dashboard shows both numbers side by side.

TabPFN-3.5 needs `TABPFN_API_TOKEN` and a network call per request. If it's missing or
the call fails, `run_comparison` falls back to a deterministic simulation (same shape
of output, `source: "simulated"`) so the app still runs end to end offline.
"""
from __future__ import annotations

import pickle
import re
import threading
import time
from functools import lru_cache

import networkx as nx
import pandas as pd

from .config import BERLIN_DATA_DIR, RESULTS_DIR
from .datasets.berlin import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    LINE_COLORS,
    build_feature_table,
    build_graph,
    chronological_split,
    discover_dataset_dir,
    encode_categoricals,
    load_closures,
    load_connections,
    load_flows,
    load_lines,
    load_stations,
    station_cols,
)
from .models import FittedModels, load_or_fit
from .xgboost_baseline import fit_classifier, fit_regressor

XGB_CACHE_PATH = RESULTS_DIR / "model_cache" / "xgboost_berlin.pkl"

_tabpfn_lock = threading.Lock()
_xgb_lock = threading.Lock()
_tabpfn_cache: FittedModels | None = None
_tabpfn_error: str | None = None
_xgb_cache: dict | None = None


def _folder() -> str:
    return discover_dataset_dir(BERLIN_DATA_DIR)


def _short(name: str) -> str:
    return re.sub(r"\s*\(Berlin\)\s*$", "", name)


def _norm(name: str) -> str:
    n = re.sub(r"^(S\+U|S/U|U|S)\s+", "", str(name).strip())
    n = re.sub(r"\s*\(Berlin\)\s*$", "", n)
    return n.strip().lower()


# --------------------------------------------------------------------------- topology
@lru_cache(maxsize=1)
def get_topology() -> dict:
    folder = _folder()
    stations = load_stations(folder)
    connections = load_connections(folder)
    lines = load_lines(folder)
    flows = load_flows(folder)
    cols = station_cols(flows)
    means = flows[cols].mean()
    p90s = flows[cols].quantile(0.9)
    maxs = flows[cols].max()

    station_records = []
    for _, row in stations.iterrows():
        name = row["station_name"]
        line_list = [l.strip() for l in str(row["u_bahn_lines"]).split(",") if l.strip()]
        station_records.append({
            "id": row["station_id"],
            "name": name,
            "lon": float(row["longitude"]),
            "lat": float(row["latitude"]),
            "lines": line_list,
            "mean": round(float(means.get(name, 0.0)), 1),
            "p90": round(float(p90s.get(name, 0.0)), 1),
            "max": round(float(maxs.get(name, 0.0)), 1),
        })

    connection_records = [
        [row["station_id_1"], row["station_id_2"]] for _, row in connections.iterrows()
    ]
    line_records = [
        {
            "id": row["line_id"], "name": row["line_name"],
            "operator_id": int(row["operator"]), "mode": row["mode"],
            "product": row["product"], "n_variants": int(row["n_variants"]),
            "color": LINE_COLORS.get(row["line_name"]),
        }
        for _, row in lines.iterrows()
    ]

    hourly = flows.copy()
    hourly["hour"] = pd.to_datetime(hourly["timestamp"]).dt.hour
    hourly_profile = [round(float(v), 0) for v in hourly.groupby("hour")[cols].sum().sum(axis=1).values]

    return {
        "stations": station_records,
        "connections": connection_records,
        "lines": line_records,
        "hourly_profile": hourly_profile,
        "scenarios": _build_scenarios(folder, station_records),
        "meta": {
            "n_stations": len(station_records),
            "n_connections": len(connection_records),
            "n_lines": len(line_records),
            "flow_rows": int(flows.shape[0]),
            "operator_id": int(lines["operator"].iloc[0]),
        },
    }


def _build_scenarios(folder: str, station_records: list[dict]) -> list[dict]:
    name_lookup = {_norm(s["name"]): s for s in station_records}
    scenarios = []
    for _, c in load_closures(folder).iterrows():
        if c["closure_type"] == "Station closure" and c["affected_segment"]:
            st = name_lookup.get(_norm(c["affected_segment"]))
            if not st:
                continue
            scenarios.append({
                "kind": "station",
                "station_id": st["id"],
                "line": None,
                "when": c["when"].isoformat(),
                "duration": c["duration"],
                "reason": c["reason"],
                "text": f"What if {_short(st['name'])} closes for {c['duration']} ({c['reason']})?",
            })
        elif c["closure_type"] == "Line suspension" and c["affected_segment"] and "↔" in str(c["affected_segment"]):
            a_raw, b_raw = [p.strip() for p in c["affected_segment"].split("↔")]
            a, b = name_lookup.get(_norm(a_raw)), name_lookup.get(_norm(b_raw))
            if not (a and b):
                continue
            scenarios.append({
                "kind": "line_segment",
                "from_id": a["id"], "to_id": b["id"], "line": c["affected_line"],
                "when": c["when"].isoformat(),
                "duration": c["duration"],
                "reason": c["reason"],
                "text": (f"What if {c['affected_line']} is suspended between {_short(a['name'])} "
                         f"and {_short(b['name'])} for {c['duration']} ({c['reason']})?"),
            })
    return scenarios


# ------------------------------------------------------------------------------ graph
def _bfs_hops(epicenters: list[str], exclude_edge: tuple[str, str] | None = None) -> dict[str, int]:
    g = build_graph(_folder())
    if exclude_edge and g.has_edge(*exclude_edge):
        g = g.copy()
        g.remove_edge(*exclude_edge)
    dist: dict[str, int] = {}
    for e in epicenters:
        if not g.has_node(e):
            continue
        for node, hop in nx.single_source_shortest_path_length(g, e, cutoff=2).items():
            if node not in dist or hop < dist[node]:
                dist[node] = hop
    return dist


@lru_cache(maxsize=1)
def _articulation_points() -> frozenset:
    return frozenset(nx.articulation_points(build_graph(_folder())))


# ------------------------------------------------------------------------------ models
@lru_cache(maxsize=1)
def get_feature_table() -> pd.DataFrame:
    # build_feature_table() itself isn't cached (unlike the load_* helpers it calls) and
    # re-scans every closure row against the full ~1.1M-row table — expensive enough
    # (seconds) that it must only run once per process, not once per request.
    return build_feature_table(_folder())


def get_tabpfn_models() -> tuple[FittedModels | None, str | None]:
    global _tabpfn_cache, _tabpfn_error
    with _tabpfn_lock:
        if _tabpfn_cache is not None or _tabpfn_error is not None:
            return _tabpfn_cache, _tabpfn_error
        try:
            table = get_feature_table()
            train_pool, _test_pool, cutoff = chronological_split(table)
            _tabpfn_cache = load_or_fit(train_pool, cutoff)
            return _tabpfn_cache, None
        except (SystemExit, Exception) as exc:  # noqa: BLE001 - genuinely want the fallback on anything
            _tabpfn_error = str(exc)
            return None, _tabpfn_error


def get_xgb_bundle() -> dict:
    global _xgb_cache
    with _xgb_lock:
        if _xgb_cache is not None:
            return _xgb_cache
        if XGB_CACHE_PATH.exists():
            _xgb_cache = pickle.loads(XGB_CACHE_PATH.read_bytes())
            return _xgb_cache

        table = get_feature_table()
        train_pool, _test_pool, _cutoff = chronological_split(table)
        X = encode_categoricals(train_pool[FEATURE_COLUMNS])
        clf, fit_c = fit_classifier(X, train_pool["overcrowded"])
        reg, fit_r = fit_regressor(X, train_pool["passengers"])
        reference = pd.DataFrame({col: train_pool[col].unique() for col in CATEGORICAL_COLUMNS})
        _xgb_cache = {
            "clf": clf, "reg": reg, "reference": reference,
            "fit_seconds": round(fit_c + fit_r, 2), "n_rows": len(train_pool),
        }
        XGB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        XGB_CACHE_PATH.write_bytes(pickle.dumps(_xgb_cache))
        return _xgb_cache


def _predict_batch_xgb(bundle: dict, rows: pd.DataFrame) -> list[dict]:
    """One predict_proba + one predict call for every affected station at once —
    XGBoost is already local/fast, but this keeps it symmetric with the TabPFN batch
    call below and means latency is a single number regardless of station count."""
    X = encode_categoricals(rows[FEATURE_COLUMNS], reference=bundle["reference"])
    t0 = time.perf_counter()
    probas = bundle["clf"].predict_proba(X)[:, 1]
    classify_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    expected = bundle["reg"].predict(X)
    regress_ms = (time.perf_counter() - t0) * 1000
    return [
        {
            "overcrowding_probability": round(float(p), 3),
            "predicted_passengers": round(float(e), 1),
            "classify_latency_ms": round(classify_ms, 2),
            "regress_latency_ms": round(regress_ms, 2),
            "source": "live",
        }
        for p, e in zip(probas, expected)
    ]


def _predict_batch_tabpfn(models: FittedModels, rows: pd.DataFrame) -> list[dict]:
    """Same batching idea as `_predict_batch_xgb`, but it matters far more here: each
    TabPFN-3.5 call re-sends the whole in-context sample, so its latency is dominated
    by that fixed cost, not by how many query rows ride along. One call for N affected
    stations costs about the same as one call for a single station — querying them
    one-by-one (what `predict_row` does) pays that fixed cost N times for nothing."""
    X = encode_categoricals(rows[FEATURE_COLUMNS], reference=models.train_sample[FEATURE_COLUMNS])
    t0 = time.perf_counter()
    probas = models.classifier.predict_proba(X)[:, 1]
    classify_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    expected = models.regressor.predict(X)
    regress_ms = (time.perf_counter() - t0) * 1000
    return [
        {
            "overcrowding_probability": round(float(p), 3),
            "predicted_passengers": round(float(e), 1),
            "classify_latency_ms": round(classify_ms, 2),
            "regress_latency_ms": round(regress_ms, 2),
            "source": "live",
        }
        for p, e in zip(probas, expected)
    ]


def _hash01(s: str) -> float:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 4294967296.0


def _simulate(model_tag: str, station_id: str, scenario_key: str, mean_flow: float, hop: int) -> dict:
    hop_decay = 0.62 if hop == 1 else 0.30
    base_p = max(0.04, min(0.96, 0.12 + hop_decay * 0.55))
    noise = (_hash01(station_id + scenario_key + model_tag) - 0.5) * 0.08
    proba = max(0.02, min(0.98, base_p + noise))
    extra = round(mean_flow * hop_decay * (0.22 + _hash01(station_id + scenario_key + model_tag + "x") * 0.35), 1)
    is_tab = model_tag == "tabpfn"
    latency_ms = (1550 + _hash01(scenario_key + model_tag + "t") * 950) if is_tab else (8 + _hash01(scenario_key + model_tag + "t") * 9)
    return {
        "overcrowding_probability": round(proba, 3),
        "predicted_passengers": extra,
        "classify_latency_ms": round(latency_ms, 2),
        "regress_latency_ms": round(latency_ms * 0.15, 2),
        "source": "simulated",
    }


def _lookup_feature_row(table: pd.DataFrame, station_name: str, ts: pd.Timestamp) -> pd.Series:
    exact = table[(table["station_name"] == station_name) & (table["timestamp"] == ts)]
    if not exact.empty:
        return exact.iloc[0]
    same_station = table[table["station_name"] == station_name]
    if same_station.empty:
        raise ValueError(f"no feature rows for station '{station_name}'")
    candidate = same_station[
        (same_station["timestamp"].dt.hour == ts.hour) & (same_station["timestamp"].dt.weekday == ts.weekday())
    ]
    if candidate.empty:
        candidate = same_station
    idx = (candidate["timestamp"] - ts).abs().idxmin()
    return candidate.loc[idx]


# --------------------------------------------------------------------------- scenario
def scenario_key(scenario: dict) -> str:
    if scenario["kind"] == "station":
        return f"st:{scenario['station_id']}"
    return f"ln:{scenario.get('line')}:{scenario['from_id']}:{scenario['to_id']}"


def _prepare(scenario: dict) -> dict:
    """BFS + feature-row lookup shared by both model-scoring entry points below — cheap
    (no model calls), so it's fine to redo it once per endpoint call rather than thread
    state between the two HTTP requests a scenario run makes."""
    topo = get_topology()
    by_id = {s["id"]: s for s in topo["stations"]}

    if scenario["kind"] == "station":
        epicenters = [scenario["station_id"]]
        dist = _bfs_hops(epicenters)
    else:
        a, b = scenario["from_id"], scenario["to_id"]
        epicenters = [a, b]
        dist = _bfs_hops(epicenters, exclude_edge=(a, b))

    affected_all = sorted(
        [(sid, hop) for sid, hop in dist.items() if sid not in epicenters and 1 <= hop <= 2],
        key=lambda t: (t[1], -by_id.get(t[0], {}).get("mean", 0)),
    )
    top = affected_all[:3]

    table = get_feature_table()
    when = pd.Timestamp(scenario["when"]).floor("15min") if scenario.get("when") else table["timestamp"].max()

    prepared = []
    for sid, hop in top:
        station = by_id[sid]
        try:
            feat_row = _lookup_feature_row(table, station["name"], when).copy()
            feat_row["network_active_closures"] = feat_row["network_active_closures"] + 1
        except Exception:
            feat_row = None
        prepared.append((sid, hop, station, feat_row))

    is_articulation = scenario["kind"] == "station" and scenario["station_id"] in _articulation_points()
    return {
        "key": scenario_key(scenario),
        "epicenters": [{"id": e, "name": by_id.get(e, {}).get("name", e)} for e in epicenters],
        "affected_count": len(affected_all),
        "is_articulation_point": is_articulation,
        "prepared": prepared,
    }


def score_with_xgboost(scenario: dict) -> dict:
    """The fast baseline tool call — local, no network, trained on the full history."""
    ctx = _prepare(scenario)
    prepared, key = ctx["prepared"], ctx["key"]
    valid = [(sid, hop, station, fr) for sid, hop, station, fr in prepared if fr is not None]
    missing = [(sid, hop, station) for sid, hop, station, fr in prepared if fr is None]

    xgb_bundle = None
    try:
        xgb_bundle = get_xgb_bundle()
    except Exception:  # noqa: BLE001
        xgb_bundle = None

    out_by_id = {}
    if valid and xgb_bundle is not None:
        try:
            batch_df = pd.DataFrame([fr for *_rest, fr in valid])
            for (sid, *_rest), out in zip(valid, _predict_batch_xgb(xgb_bundle, batch_df)):
                out_by_id[sid] = out
        except Exception:  # noqa: BLE001
            xgb_bundle = None
    if xgb_bundle is None:
        for sid, hop, station, _fr in valid:
            out_by_id[sid] = _simulate("xgb", sid, key, station["mean"], hop)
    for sid, hop, station in missing:
        out_by_id[sid] = _simulate("xgb", sid, key, station["mean"], hop)

    return {
        "scenario": scenario,
        "epicenters": ctx["epicenters"],
        "affected_count": ctx["affected_count"],
        "is_articulation_point": ctx["is_articulation_point"],
        "rows": [
            {"id": sid, "name": station["name"], "hop": hop, "mean": station["mean"], "xgb": out_by_id[sid]}
            for sid, hop, station, _fr in prepared
        ],
        "xgb_meta": {
            "available": xgb_bundle is not None,
            "train_rows": xgb_bundle["n_rows"] if xgb_bundle else None,
            "fit_seconds": xgb_bundle["fit_seconds"] if xgb_bundle else None,
        },
    }


def score_with_tabpfn(scenario: dict, live_tabpfn: bool = True) -> dict:
    """The specialised-tool call the agent hands off to — one batched TabPFN-3.5 request
    covering every affected station, or the deterministic simulation when `live_tabpfn`
    is off or the live call isn't available (see module docstring)."""
    ctx = _prepare(scenario)
    prepared, key = ctx["prepared"], ctx["key"]
    valid = [(sid, hop, station, fr) for sid, hop, station, fr in prepared if fr is not None]
    missing = [(sid, hop, station) for sid, hop, station, fr in prepared if fr is None]

    tab_models, tab_error = (
        get_tabpfn_models() if live_tabpfn else (None, "live TabPFN calls disabled for this request")
    )

    out_by_id = {}
    if valid and tab_models is not None:
        try:
            batch_df = pd.DataFrame([fr for *_rest, fr in valid])
            for (sid, *_rest), out in zip(valid, _predict_batch_tabpfn(tab_models, batch_df)):
                out_by_id[sid] = out
        except Exception as exc:  # noqa: BLE001
            tab_error = str(exc)
            tab_models = None
    if tab_models is None:
        for sid, hop, station, _fr in valid:
            out = _simulate("tabpfn", sid, key, station["mean"], hop)
            if tab_error:
                out["note"] = tab_error[:160]
            out_by_id[sid] = out
    for sid, hop, station in missing:
        out = _simulate("tabpfn", sid, key, station["mean"], hop)
        if tab_error:
            out["note"] = tab_error[:160]
        out_by_id[sid] = out

    return {
        "scenario": scenario,
        "epicenters": ctx["epicenters"],
        "affected_count": ctx["affected_count"],
        "is_articulation_point": ctx["is_articulation_point"],
        "rows": [
            {"id": sid, "name": station["name"], "hop": hop, "mean": station["mean"], "tabpfn": out_by_id[sid]}
            for sid, hop, station, _fr in prepared
        ],
        "tabpfn_meta": {
            "available": tab_models is not None,
            "live_requested": live_tabpfn,
            "context_rows": len(tab_models.train_sample) if tab_models else None,
            "error": None if not live_tabpfn else tab_error,
        },
    }


def run_comparison(scenario: dict, live_tabpfn: bool = True) -> dict:
    """Convenience wrapper combining both tool calls in one Python call (used by tests /
    the CLI); the web app calls `score_with_xgboost` and `score_with_tabpfn` separately
    so the UI can show the fast result first."""
    xgb_result = score_with_xgboost(scenario)
    tab_result = score_with_tabpfn(scenario, live_tabpfn=live_tabpfn)
    tab_by_id = {r["id"]: r["tabpfn"] for r in tab_result["rows"]}
    rows = [{**row, "tabpfn": tab_by_id.get(row["id"])} for row in xgb_result["rows"]]
    return {
        **xgb_result,
        "rows": rows,
        "tabpfn_meta": tab_result["tabpfn_meta"],
    }
