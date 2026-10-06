"""Operator-facing query layer over the anomaly engine (`tabpfn_lab/anomaly.py`).

One place that answers the questions a control room asks, used by the MCP server
(`mcp_server/server.py`, which the LLM agent calls), the webapp (`webapp/server.py`) and the
notebook. Two kinds of answers:

- **Replay** of the test windows (Sep 1–21 backtest, Sep 22–30 Alstom hold-out): every model's
  day-ahead forecast was precomputed by `make anomaly` (`results/anomaly/`), so network-wide
  questions ("what does the network look like at 22:00?") are instant.
- **Live TabPFN-3.5** calls: `station_forecast(live=True)` and `what_if(...)` re-fit TabPFN-3.5 on
  the fold's 10k-row context (once per process, ~5 s) and score the requested station-hours,
  optionally with hypothetical context (an extra event, a closure, rain). If TabPFN-3.5 is
  unreachable, `what_if` falls back to the full-history XGBoost model and says so.

Timestamps are Berlin local time on the hour; forecasts exist for service hours 00 and 05–23.
"""
from __future__ import annotations

import difflib
import threading
import time
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd

from . import anomaly as A
from .datasets.berlin import LINE_COLORS

MODEL_NAMES = {
    "tabpfn": "TabPFN-3.5 (10k context)", "xgb_full": "XGBoost (full history)",
    "xgb_small": "XGBoost (10k sample)", "xgb_profile": "XGBoost (profile only)", "profile": "Profile baseline",
}
REPLAY_INFO = {"engine": "TabPFN-3.5 day-ahead forecast precomputed by `make anomaly` (replay — no live call)"}
_lock = threading.Lock()
_bundles: dict[tuple[str, str], dict] = {}


def short(name: str | None) -> str:
    return (name or "").replace(" (Berlin)", "")


# ------------------------------------------------------------------------------ cached replay
@lru_cache(maxsize=1)
def cache() -> dict:
    """`make anomaly` artifacts, indexed for per-hour / per-station lookups. Raises
    FileNotFoundError if the benchmark has not been run."""
    data = A.load_cached()
    preds, metrics, scenarios = data["predictions"], data["metrics"], data["scenarios"]
    keys = [v["key"] for v in metrics["folds"]["holdout"]["models"].values()]
    for k in keys:
        preds[f"p_{k}"] = preds[f"p_{k}"].astype("float64")
        preds[f"z_hat_{k}"] = preds[f"z_hat_{k}"].astype("float64")
        preds[f"pax_{k}"] = np.expm1(preds["normal_level"] + preds[f"z_hat_{k}"] * preds["station_scale"]).clip(lower=0)
    by_fold = {f: g.sort_values(["ts", "station"]) for f, g in preds.groupby("fold")}
    for sc in scenarios:  # which model flagged the scenario, and from which hour
        sc["alarms"], sc["drops"] = {}, {}
        if not sc["station"]:
            continue
        g = by_fold[sc["fold"]]
        w = g[(g["station"] == sc["station"]) & (g["ts"] >= pd.Timestamp(sc["start"]).floor("h"))
              & (g["ts"] <= pd.Timestamp(sc["end"]).ceil("h"))]
        for v in metrics["folds"][sc["fold"]]["models"].values():
            hits = w.loc[w[f"p_{v['key']}"] >= v["alarm_threshold"], "ts"]
            drops = w.loc[w[f"z_hat_{v['key']}"] <= A.DROP_Z, "ts"]
            sc["alarms"][v["key"]] = hits.min().isoformat() if len(hits) else None
            sc["drops"][v["key"]] = drops.min().isoformat() if len(drops) else None
    return {"metrics": metrics, "scenarios": scenarios, "preds": preds, "keys": keys, "by_fold": by_fold}


def thresholds(fold: str) -> dict[str, float]:
    return {v["key"]: v["alarm_threshold"] for v in cache()["metrics"]["folds"][fold]["models"].values()}


def fold_of(ts: pd.Timestamp) -> str | None:
    for f, spec in A.FOLDS.items():
        if spec["start"] <= ts < spec["end"]:
            return f
    return None


def _parse_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Europe/Berlin").tz_localize(None)
    return ts.floor("h")


def _window_error(ts: pd.Timestamp) -> dict:
    return {"error": f"{ts} is outside the forecast windows. Forecasts exist for "
                     + "; ".join(f"{v['label']}: {v['start']:%Y-%m-%d} to {v['end'] - pd.Timedelta(hours=1):%Y-%m-%d %H:00}"
                                 for v in A.FOLDS.values())
                     + " (service hours 00 and 05–23)."}


# --------------------------------------------------------------------------------- stations
@lru_cache(maxsize=1)
def station_names() -> list[str]:
    return sorted(A.hourly_flows()["station"].unique())


def resolve_station(query: str, max_results: int = 5) -> list[str]:
    names = station_names()
    exact = A.find_station(query, names)
    q = query.strip().lower()
    subs = [n for n in names if q in n.lower()]
    fuzzy = difflib.get_close_matches(query, [short(n) for n in names], n=max_results, cutoff=0.5)
    fuzzy = [n for n in names if short(n) in fuzzy]
    return list(dict.fromkeys(([exact] if exact else []) + subs + fuzzy))[:max_results]


def _station(query: str) -> str:
    hits = resolve_station(query, 1)
    if not hits:
        raise ValueError(f"No station matches {query!r}. Use resolve_station to search.")
    return hits[0]


@lru_cache(maxsize=1)
def line_sequences() -> dict[str, list[str]]:
    """Each line's stations in running order (longest terminus-to-terminus path on the line)."""
    g, meta = A.station_graph(), A.station_meta().set_index("station")
    out = {}
    for line in sorted({l for ls in meta["lines"] for l in ls.split(",")}):
        sub = g.subgraph([n for n in g.nodes if line in meta.loc[n, "lines"].split(",")])
        ends = [n for n in sub.nodes if sub.degree(n) == 1] or list(sub.nodes)[:1]
        best: list[str] = []
        for i, a in enumerate(ends):
            for b in ends[i + 1:]:
                try:
                    p = nx.shortest_path(sub, a, b)
                except nx.NetworkXNoPath:
                    continue
                if len(p) > len(best):
                    best = p
        out[line] = best or list(sub.nodes)
    return out


# ------------------------------------------------------------------------------- live models
def live_bundle(fold: str) -> dict:
    """TabPFN-3.5 fitted on the fold's 10k-row context (fit once per process, ~5 s). Falls back
    to the full-history XGBoost model when TabPFN-3.5 is unreachable (no token, API error)."""
    with _lock:
        for engine in ("tabpfn", "xgb_full"):
            if (fold, engine) in _bundles:
                return _bundles[(fold, engine)]
        table = A.anomaly_table(fold)
        train = table[table["split"] == "train"]
        try:
            b = A.fit_tabpfn(A.context_sample(train))
            b["engine"], b["predict"] = "TabPFN-3.5 (live API)", A.predict_tabpfn
            _bundles[(fold, "tabpfn")] = b
        except (Exception, SystemExit) as exc:  # noqa: BLE001 — config.tabpfn_token raises SystemExit
            b = A.fit_xgboost(train, A.FEATURES)
            b["engine"], b["predict"] = f"XGBoost full history (fallback: TabPFN-3.5 unavailable — {exc})", A.predict_xgboost
            _bundles[(fold, "xgb_full")] = b
        return b


def _score(fold: str, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    b = live_bundle(fold)
    t0 = time.perf_counter()
    p = b["predict"](b, rows)
    pax = np.expm1(rows["normal_level"].to_numpy() + p["z_hat"].to_numpy() * rows["station_scale"].to_numpy()).clip(min=0)
    out = rows[["ts", "station", "normal_passengers"]].assign(p_surge=p["p_surge"].to_numpy(),
                                                              z_hat=p["z_hat"].to_numpy(), pax=pax)
    return out, {"engine": b["engine"], "fit_s": round(b["fit_s"], 2), "predict_s": round(time.perf_counter() - t0, 2),
                 "rows_scored": len(rows)}


def _hours(start: str, hours: int) -> tuple[str, pd.DataFrame] | dict:
    ts = _parse_ts(start)
    fold = fold_of(ts)
    if fold is None:
        return _window_error(ts)
    table = A.anomaly_table(fold)
    test = table[table["split"] == "test"]
    end = ts + pd.Timedelta(hours=max(1, min(int(hours), 24)))
    return fold, test[(test["ts"] >= ts) & (test["ts"] < end)]


# ------------------------------------------------------------------------------- the queries
def scoreboard() -> dict:
    """Benchmark summary: both folds pooled + per fold, plus the XGBoost learning curve."""
    m = cache()["metrics"]
    keep = ["pr_auc", "roc_auc", "alarm_precision", "recall_event", "closure_drop_recall", "z_mae"]
    pm = m["pooled"]["models"]
    tab, small, full = pm["TabPFN-3.5 (10k context)"], pm["XGBoost (10k sample)"], pm["XGBoost (full history)"]
    matches = {}
    for f, fm in m["folds"].items():
        t = fm["models"]["TabPFN-3.5 (10k context)"]["pr_auc"]
        hit = next((c["n_train"] for c in fm.get("xgboost_learning_curve") or [] if c["pr_auc"] >= t), None)
        matches[fm["label"]] = hit
    return {
        "key_findings": [
            f"Same 10k training rows: TabPFN-3.5 PR-AUC {tab['pr_auc']:.3f} vs XGBoost {small['pr_auc']:.3f} "
            f"(+{100 * (tab['pr_auc'] / small['pr_auc'] - 1):.0f}%); event surges caught {tab['recall_event']:.0%} vs {small['recall_event']:.0%}.",
            "Rows XGBoost needs to match TabPFN-3.5's 10k-row PR-AUC: "
            + ", ".join(f"{k}: ~{v:,}" if v else f"{k}: not reached" for k, v in matches.items()) + ".",
            f"XGBoost on the full history (28-35x more rows) is better: PR-AUC {full['pr_auc']:.3f}, event surges {full['recall_event']:.0%}.",
            f"Station-closure collapses forecast: TabPFN-3.5 {tab['closure_drop_recall']:.0%}, XGBoost full {full['closure_drop_recall']:.0%}, "
            "clock-only models 0%.",
            "Use TabPFN-3.5 where history is short (new station/venue/line, new question, no training pipeline); "
            "a trained GBM once months of labelled data exist. TabPFN-3.5 inference is an API call (seconds), XGBoost local (ms).",
        ],
        "what_is_measured": "Forecast of hourly anomalies (z = deviation from the station's normal for that day "
                            "type and hour). Surge = z >= 2. Alarm = a model's top 2% of station-hours.",
        "pooled": {n: {k: v.get(k) for k in keep} for n, v in m["pooled"]["models"].items()},
        "folds": {f: {"label": fm["label"], "n_test_rows": fm["n_test_rows"],
                      "models": {n: {k: v.get(k) for k in [*keep, "n_train", "fit_s", "predict_s"]}
                                 for n, v in fm["models"].items()},
                      "xgboost_learning_curve": fm.get("xgboost_learning_curve")}
                  for f, fm in m["folds"].items()},
    }


def network_outlook(timestamp: str, top: int = 8) -> dict:
    """Day-ahead outlook for one hour across the whole network: alarms, per-line load vs normal,
    active drivers. Replay of the precomputed forecasts (all five models)."""
    ts = _parse_ts(timestamp)
    fold = fold_of(ts)
    if fold is None:
        return _window_error(ts)
    g = cache()["by_fold"][fold]
    h = g[g["ts"] == ts]
    if h.empty:
        return {"error": f"No service at {ts:%H:00} (forecasts cover 00 and 05–23)."}
    thr = thresholds(fold)["tabpfn"]
    alarms = h[h["p_tabpfn"] >= thr].sort_values("p_tabpfn", ascending=False)
    meta = A.station_meta().set_index("station")
    lines = []
    for line, seq in line_sequences().items():
        w = h[h["station"].isin(seq)]
        if w.empty:
            continue
        lines.append({"line": line, "stations": len(w),
                      "forecast_passengers": round(float(w["pax_tabpfn"].sum())),
                      "normal_passengers": round(float(w["normal_passengers"].sum())),
                      "load_vs_normal_pct": round(float(100 * (w["pax_tabpfn"].sum() / max(w["normal_passengers"].sum(), 1) - 1)), 1),
                      "alarms": int((w["p_tabpfn"] >= thr).sum())})
    return {
        "timestamp": ts.isoformat(), "window": A.FOLDS[fold]["label"], "model": MODEL_NAMES["tabpfn"],
        "inference": REPLAY_INFO,
        "weather": {"temp_c": round(float(h["temp"].iloc[0]), 1), "rain_mm": round(float(h["prcp"].iloc[0]), 1)},
        "network": {"forecast_passengers": round(float(h["pax_tabpfn"].sum())),
                    "normal_passengers": round(float(h["normal_passengers"].sum())),
                    "median_forecast_z": round(float(h["z_hat_tabpfn"].median()), 2),
                    "alarms": len(alarms), "stations_closed": int(h["station_closed"].sum())},
        "lines": sorted(lines, key=lambda r: -r["load_vs_normal_pct"]),
        "top_alarms": [_row_summary(r, meta) for r in alarms.head(top).itertuples()],
        "observed_replay": {"actual_surges": int(h["surge"].sum()),
                            "alarms_that_surged": int(alarms["surge"].sum())},
    }


def _row_summary(r, meta) -> dict:
    drivers = []
    if r.event_egress_att > 0:
        drivers.append(f"event crowd leaving (~{int(r.event_egress_att):,})")
    if r.event_ingress_att > 0:
        drivers.append(f"event crowd arriving (~{int(r.event_ingress_att):,})")
    if r.event_nearby_egress_att > 0:
        drivers.append(f"event at a neighbouring station (~{int(r.event_nearby_egress_att):,})")
    if r.station_closed:
        drivers.append("station closed")
    if r.line_suspended_here:
        drivers.append("line suspended here")
    if r.neighbor_closed:
        drivers.append("neighbouring station closed")
    if r.prcp >= 1:
        drivers.append(f"rain {r.prcp:.1f} mm")
    return {"station": short(r.station), "lines": meta.loc[r.station, "lines"] if r.station in meta.index else "",
            "surge_probability": round(float(r.p_tabpfn), 3), "forecast_z": round(float(r.z_hat_tabpfn), 2),
            "forecast_passengers": round(float(r.pax_tabpfn)), "normal_passengers": round(float(r.normal_passengers)),
            "drivers": drivers or ["no scheduled driver (pattern-based)"],
            "observed_replay": {"passengers": round(float(r.passengers)), "z": round(float(r.z), 2)}}


def line_corridor(line: str, timestamp: str) -> dict:
    """Forecast load along one line in running order for one hour — where along the corridor the
    pressure builds or collapses (TabPFN-3.5 forecast vs each station's normal)."""
    ts = _parse_ts(timestamp)
    fold = fold_of(ts)
    if fold is None:
        return _window_error(ts)
    line = line.strip().upper()
    seq = line_sequences().get(line)
    if not seq:
        return {"error": f"Unknown line {line!r}. Lines: {', '.join(line_sequences())}."}
    g = cache()["by_fold"][fold]
    h = g[g["ts"] == ts].set_index("station")
    thr = thresholds(fold)["tabpfn"]
    stops = [{"station": short(s), "forecast_passengers": round(float(h.at[s, "pax_tabpfn"])),
              "normal_passengers": round(float(h.at[s, "normal_passengers"])),
              "forecast_z": round(float(h.at[s, "z_hat_tabpfn"]), 2),
              "surge_probability": round(float(h.at[s, "p_tabpfn"]), 3),
              "alarm": bool(h.at[s, "p_tabpfn"] >= thr), "closed": bool(h.at[s, "station_closed"]),
              "observed_replay_passengers": round(float(h.at[s, "passengers"]))}
             for s in seq if s in h.index]
    hot = [s["station"] for s in stops if s["alarm"]]
    return {"line": line, "color": LINE_COLORS.get(line), "timestamp": ts.isoformat(), "model": MODEL_NAMES["tabpfn"],
            "inference": REPLAY_INFO,
            "terminus_to_terminus": [stops[0]["station"], stops[-1]["station"]] if stops else [],
            "forecast_passengers": sum(s["forecast_passengers"] for s in stops),
            "normal_passengers": sum(s["normal_passengers"] for s in stops),
            "alarm_stations": hot, "stops": stops}


def station_forecast(station: str, start: str, hours: int = 6, live: bool = True) -> dict:
    """Hour-by-hour forecast for one station: normal, TabPFN-3.5 forecast (live API call when
    `live`), the other models' precomputed forecasts, and what was observed (replay)."""
    st = _station(station)
    res = _hours(start, hours)
    if isinstance(res, dict):
        return res
    fold, rows = res
    rows = rows[rows["station"] == st]
    if rows.empty:
        return {"error": f"No service hours for {short(st)} in that range."}
    g = cache()["by_fold"][fold]
    cached = g[(g["station"] == st) & g["ts"].isin(rows["ts"])].set_index("ts")
    thr = thresholds(fold)
    info = REPLAY_INFO
    live_df = None
    if live:
        live_df, info = _score(fold, rows)
        live_df = live_df.set_index("ts")
    hours_out = []
    for t in rows["ts"]:
        c = cached.loc[t]
        p = float(live_df.at[t, "p_surge"]) if live_df is not None else float(c["p_tabpfn"])
        z = float(live_df.at[t, "z_hat"]) if live_df is not None else float(c["z_hat_tabpfn"])
        pax = float(live_df.at[t, "pax"]) if live_df is not None else float(c["pax_tabpfn"])
        hours_out.append({
            "hour": t.isoformat(), "normal_passengers": round(float(c["normal_passengers"])),
            "tabpfn": {"forecast_passengers": round(pax), "forecast_z": round(z, 2), "surge_probability": round(p, 3),
                       "alarm": p >= thr["tabpfn"]},
            "xgboost_full_history": {"forecast_passengers": round(float(c["pax_xgb_full"])),
                                     "surge_probability": round(float(c["p_xgb_full"]), 3),
                                     "alarm": bool(c["p_xgb_full"] >= thr["xgb_full"])},
            "profile_baseline_alarm": bool(c["p_profile"] >= thr["profile"]),
            "drivers": _row_summary(next(cached.loc[[t]].reset_index().assign(station=st).itertuples()),
                                    A.station_meta().set_index("station"))["drivers"],
            "observed_replay": {"passengers": round(float(c["passengers"])), "z": round(float(c["z"]), 2),
                                "surge": bool(c["surge"])},
        })
    return {"station": short(st), "window": A.FOLDS[fold]["label"], "inference": info,
            "alarm_rule": "alarm = surge probability in the model's top 2% of station-hours", "hours": hours_out}


def explain_anomaly(station: str, timestamp: str) -> dict:
    """Why is (or isn't) this station-hour off its normal pattern? Context drivers, all five
    models' forecasts, the observed outcome, and how similar situations behaved in training."""
    st, ts = _station(station), _parse_ts(timestamp)
    fold = fold_of(ts)
    if fold is None:
        return _window_error(ts)
    g = cache()["by_fold"][fold]
    r = g[(g["station"] == st) & (g["ts"] == ts)]
    if r.empty:
        return {"error": f"No service hour for {short(st)} at {ts}."}
    r = r.iloc[0]
    thr = thresholds(fold)
    meta = A.station_meta().set_index("station")
    train = A.anomaly_table(fold)
    train = train[(train["split"] == "train") & (train["station"] == st)]
    history = {"normal_hours_mean_z": round(float(train.loc[train["driver"] == "none", "z"].mean()), 2)}
    if r["event_egress_att"] > 0:
        m = train["event_egress_att"] > 0
        history["past_event_egress_hours"] = {"n": int(m.sum()), "mean_z": round(float(train.loc[m, "z"].mean()), 2) if m.any() else None,
                                             "surge_rate": round(float(train.loc[m, "surge"].mean()), 3) if m.any() else None}
    if r["station_closed"]:
        m = train["station_closed"] == 1
        history["past_closed_hours"] = {"n": int(m.sum()), "mean_z": round(float(train.loc[m, "z"].mean()), 2) if m.any() else None}
    return {
        "station": short(st), "hour": ts.isoformat(), "lines": meta.loc[st, "lines"], "inference": REPLAY_INFO,
        "tabpfn_forecast": {"forecast_passengers": round(float(r["pax_tabpfn"])), "forecast_z": round(float(r["z_hat_tabpfn"]), 2),
                            "surge_probability": round(float(r["p_tabpfn"]), 3), "alarm": bool(r["p_tabpfn"] >= thr["tabpfn"])},
        "normal_passengers": round(float(r["normal_passengers"])),
        "drivers": _row_summary(next(pd.DataFrame([r]).itertuples()), meta)["drivers"],
        "context": {"event_egress_attendance": float(r["event_egress_att"]), "event_ingress_attendance": float(r["event_ingress_att"]),
                    "station_closed": bool(r["station_closed"]), "line_suspended_here": bool(r["line_suspended_here"]),
                    "neighbour_closed": bool(r["neighbor_closed"]), "rain_mm": float(r["prcp"]), "temp_c": float(r["temp"])},
        "models": {MODEL_NAMES[k]: {"forecast_z": round(float(r[f"z_hat_{k}"]), 2), "surge_probability": round(float(r[f"p_{k}"]), 3),
                                    "forecast_passengers": round(float(r[f"pax_{k}"])), "alarm": bool(r[f"p_{k}"] >= thr[k])}
                   for k in cache()["keys"]},
        "observed_replay": {"passengers": round(float(r["passengers"])), "z": round(float(r["z"]), 2),
                            "surge": bool(r["surge"]), "drop": bool(r["drop"])},
        "same_station_history_in_training": history,
    }


def what_if(station: str, start: str, hours: int = 4, event_attendance: int = 0, event_end: str | None = None,
            close_station: bool = False, rain_mm: float | None = None) -> dict:
    """Live TabPFN-3.5 counterfactual: score the same station-hours as scheduled and with a
    hypothetical event (crowd leaving at `event_end`), a closure of the station, or rain. For a
    closure the neighbouring stations are scored too, so the answer covers the network around it."""
    st = _station(station)
    res = _hours(start, hours)
    if isinstance(res, dict):
        return res
    fold, rows = res
    neighbours = sorted(A.station_graph().neighbors(st)) if close_station else []
    rows = rows[rows["station"].isin([st, *neighbours])]
    if rows.empty:
        return {"error": f"No service hours for {short(st)} in that range."}
    hyp = rows.copy()
    changes = []
    on_station = hyp["station"] == st
    if event_attendance:
        end = _parse_ts(event_end) if event_end else rows["ts"].min() + pd.Timedelta(hours=1)
        egress = on_station & hyp["ts"].isin([end, end + pd.Timedelta(hours=1)])
        hyp.loc[egress, "event_egress_att"] += float(event_attendance)
        near = hyp["station"].isin(A.station_graph().neighbors(st)) & hyp["ts"].isin([end, end + pd.Timedelta(hours=1)])
        hyp.loc[near, "event_nearby_egress_att"] += float(event_attendance)
        hyp.loc[:, "city_event_att_day"] += float(event_attendance)
        changes.append(f"event with ~{int(event_attendance):,} attendees ending {end:%a %d %b %H:00} at {short(st)}")
    if close_station:
        hyp.loc[on_station, "station_closed"] = 1
        hyp.loc[~on_station, "neighbor_closed"] = 1
        hyp["network_closures_active"] += 1
        changes.append(f"{short(st)} closed for the whole window")
    if rain_mm is not None:
        hyp["prcp"], hyp["prcp_3h"] = float(rain_mm), float(rain_mm) * 3
        changes.append(f"rain {rain_mm:.1f} mm/h")
    if not changes:
        return {"error": "Give at least one change: event_attendance, close_station or rain_mm."}
    both = pd.concat([rows, hyp], ignore_index=True)
    scored, info = _score(fold, both)
    base, scen = scored.iloc[:len(rows)].reset_index(drop=True), scored.iloc[len(rows):].reset_index(drop=True)
    thr = thresholds(fold)["tabpfn"] if info["engine"].startswith("TabPFN") else thresholds(fold)["xgb_full"]
    out = []
    for i in range(len(rows)):
        b, s = base.iloc[i], scen.iloc[i]
        out.append({"station": short(b["station"]), "hour": b["ts"].isoformat(), "normal_passengers": round(float(b["normal_passengers"])),
                    "as_scheduled": {"forecast_passengers": round(float(b["pax"])), "forecast_z": round(float(b["z_hat"]), 2),
                                     "surge_probability": round(float(b["p_surge"]), 3), "alarm": bool(b["p_surge"] >= thr)},
                    "what_if": {"forecast_passengers": round(float(s["pax"])), "forecast_z": round(float(s["z_hat"]), 2),
                                "surge_probability": round(float(s["p_surge"]), 3), "alarm": bool(s["p_surge"] >= thr)},
                    "delta_passengers": round(float(s["pax"] - b["pax"]))})
    focus = [o for o in out if o["station"] == short(st)]
    peak = max(focus, key=lambda o: abs(o["delta_passengers"]))
    neighbour_peaks = []
    for nb in neighbours:
        hrs = [o for o in out if o["station"] == short(nb)]
        if hrs:
            p = max(hrs, key=lambda o: abs(o["delta_passengers"]))
            neighbour_peaks.append({"station": p["station"], "hour": p["hour"],
                                    "as_scheduled_passengers": p["as_scheduled"]["forecast_passengers"],
                                    "what_if_passengers": p["what_if"]["forecast_passengers"],
                                    "delta_passengers": p["delta_passengers"], "what_if_alarm": p["what_if"]["alarm"]})
    return {"station": short(st), "scenario": "; ".join(changes), "inference": info,
            "peak_change": peak, "neighbours_peak_change": neighbour_peaks, "alarm_threshold": round(thr, 4),
            "note": "Counterfactual on the same context features the models were trained on; the simulated data has "
                    "no spill-over to neighbouring stations, so neighbour effects are expected to be small.",
            "hours": out}


def list_scenarios(kind: str | None = None, window: str | None = None, limit: int = 8) -> list[dict]:
    """Real events / closures / rain spells in the test windows, ranked by observed anomaly size,
    with which models raised an alarm (surge) or forecast the collapse (closure)."""
    out = []
    for s in cache()["scenarios"]:
        if (kind and s["kind"] != kind) or (window and s["fold"] != window):
            continue
        flagged = s["drops"] if s["peak"]["z"] < 0 else s["alarms"]
        out.append({"id": s["id"], "kind": s["kind"], "window": A.FOLDS[s["fold"]]["label"], "title": s["title"],
                    "cause": s["cause"], "station": s["station_short"], "peak_hour": s["peak"]["ts"],
                    "observed_peak_z": s["peak"]["z"], "observed_passengers": s["peak"]["actual"],
                    "normal_passengers": s["peak"]["normal"],
                    "flagged_by": {MODEL_NAMES[k]: (v or False) for k, v in (flagged or {}).items()}})
    return sorted(out, key=lambda s: -abs(s["observed_peak_z"]))[:limit]


def describe() -> dict:
    m = cache()["metrics"]
    return {
        "network": {"stations": len(station_names()), "lines": list(line_sequences()), "operator": "BVG (Berlin U-Bahn)"},
        "data": "Alstom-simulated hourly passenger flows (Jun 10 – Sep 30 2026), real events/weather, simulated closures.",
        "forecast_windows": {f: {"label": v["label"], "from": f"{v['start']:%Y-%m-%d}",
                                 "to": f"{v['end'] - pd.Timedelta(hours=1):%Y-%m-%d %H:00}"} for f, v in A.FOLDS.items()},
        "service_hours": "00 and 05–23 (no service 01–04)",
        "target": "z = deviation from the station's normal for that day type and hour (robust, per station-hour); "
                  "surge = z >= 2, collapse = z <= -2",
        "models": list(MODEL_NAMES.values()),
        "results_created": m["created"], "live_tabpfn_in_benchmark": m["live_tabpfn"],
    }
