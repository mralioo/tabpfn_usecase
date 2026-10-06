"""Operator webapp, two pages:

- `/` — **Early-Warning Desk**: the anomaly use case (`tabpfn_lab.anomaly`). Flows are
  normalised against each station's normal profile; the map, timeline and alarm feed show
  where and when a station runs off its normal pattern (event egress, closures, weather),
  and how the profile baseline, XGBoost and TabPFN-3.5 forecast it. Reads the artifacts
  written by `make anomaly` (`results/anomaly/`); a live TabPFN-3.5 re-score is one click.
- `/closures` — **Closure Impact Lab**: the map + chat dashboard that runs real XGBoost
  and TabPFN-3.5 calls side by side on hypothetical station/line closures
  (`tabpfn_lab.closure_impact`).

Built on Starlette + uvicorn (already pulled in by `fastmcp`/`streamlit`, so this adds
no new runtime dependency) rather than FastAPI, in keeping with the rest of this repo's
"smallest harness that demonstrates the pattern" approach.

Run: `make dashboard-closures` (http://127.0.0.1:8000), or directly:
    .venv/bin/uvicorn webapp.server:app --reload
"""
from __future__ import annotations

import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tabpfn_lab.config import load_dotenv  # noqa: E402
from tabpfn_lab import anomaly as warn  # noqa: E402
from tabpfn_lab import closure_impact as engine  # noqa: E402
from tabpfn_lab.datasets.berlin import LINE_COLORS  # noqa: E402

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _json_safe(obj):
    """`engine` mostly returns plain Python types, but numpy/pandas scalars can leak in
    from pandas aggregates — convert them rather than let JSONResponse's encoder choke."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


async def topology(request: Request) -> JSONResponse:
    return JSONResponse(_json_safe(engine.get_topology()))


async def impact_xgboost(request: Request) -> JSONResponse:
    scenario = await request.json()
    try:
        result = engine.score_with_xgboost(scenario)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(_json_safe(result))


async def impact_tabpfn(request: Request) -> JSONResponse:
    body = await request.json()
    live = bool(body.pop("live_tabpfn", True))
    try:
        result = engine.score_with_tabpfn(body, live_tabpfn=live)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(_json_safe(result))


async def index(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "warning.html")


async def closures_page(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ------------------------------------------------------------------ Early-Warning Desk API
def _short(name: str) -> str:
    return name.replace(" (Berlin)", "")


@lru_cache(maxsize=1)
def _warn_cache() -> dict:
    """Artifacts from `make anomaly`, indexed for the per-hour / per-station endpoints."""
    data = warn.load_cached()
    preds = data["predictions"]
    keys = {v["key"]: name for name, v in data["metrics"]["folds"]["holdout"]["models"].items()}
    for k in keys:  # float32 from the models -> float64 so rounding serialises cleanly
        preds[f"p_{k}"] = preds[f"p_{k}"].astype("float64")
        preds[f"z_hat_{k}"] = preds[f"z_hat_{k}"].astype("float64")
        # forecast passengers per model, back from z to the flow scale
        preds[f"pax_{k}"] = np.expm1(preds["normal_level"] + preds[f"z_hat_{k}"] * preds["station_scale"]).clip(lower=0)
    by_fold = {f: g.sort_values(["ts", "station"]) for f, g in preds.groupby("fold")}
    metrics = data["metrics"]
    scenarios = data["scenarios"]
    for sc in scenarios:  # which model raised an alarm inside the scenario window, and when first
        sc["alarms"] = {}
        if not sc["station"]:
            continue
        g = by_fold[sc["fold"]]
        w = g[(g["station"] == sc["station"]) & (g["ts"] >= pd.Timestamp(sc["start"]).floor("h"))
              & (g["ts"] <= pd.Timestamp(sc["end"]).ceil("h"))]
        sc["drops"] = {}
        for name, v in metrics["folds"][sc["fold"]]["models"].items():
            hits = w.loc[w[f"p_{v['key']}"] >= v["alarm_threshold"], "ts"]
            sc["alarms"][v["key"]] = hits.min().isoformat() if len(hits) else None
            drops = w.loc[w[f"z_hat_{v['key']}"] <= warn.DROP_Z, "ts"]  # forecast collapse (closures)
            sc["drops"][v["key"]] = drops.min().isoformat() if len(drops) else None
    return {"metrics": metrics, "scenarios": scenarios, "preds": preds, "keys": keys, "by_fold": by_fold}


def _thresholds(fold: str) -> dict[str, float]:
    return {v["key"]: v["alarm_threshold"] for v in _warn_cache()["metrics"]["folds"][fold]["models"].values()}


def _fold(request: Request) -> str:
    fold = request.query_params.get("fold", "holdout")
    if fold not in warn.FOLDS:
        raise ValueError(f"unknown fold {fold!r}")
    return fold


def _warn_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, FileNotFoundError):
        return JSONResponse({"error": "No results in results/anomaly/ — run `make anomaly` (or `make anomaly-offline`) "
                                      "and reload."}, status_code=503)
    return JSONResponse({"error": str(exc)}, status_code=400)


async def warn_network(request: Request) -> JSONResponse:
    meta = warn.station_meta()
    names = set(warn.hourly_flows()["station"].unique())
    stations = [{"name": r.station, "short": _short(r.station), "lon": float(r.lon), "lat": float(r.lat),
                 "lines": r.lines.split(","), "primary_line": r.primary_line}
                for r in meta.itertuples() if r.station in names]
    edges = [[a, b] for a, b in warn.station_graph().edges() if a in names and b in names]
    return JSONResponse({"stations": stations, "edges": edges, "line_colors": LINE_COLORS})


async def warn_summary(request: Request) -> JSONResponse:
    try:
        c = _warn_cache()
    except Exception as exc:  # noqa: BLE001
        return _warn_error(exc)
    scen = [{k: v for k, v in s.items() if k != "series"} for s in c["scenarios"]]
    folds = {f: {"label": warn.FOLDS[f]["label"],
                 "hours": [t.isoformat() for t in sorted(g["ts"].unique())]} for f, g in c["by_fold"].items()}
    return JSONResponse(_json_safe({"metrics": c["metrics"], "scenarios": scen, "folds": folds,
                                    "models": c["keys"]}))


async def warn_timeline(request: Request) -> JSONResponse:
    try:
        fold, c = _fold(request), _warn_cache()
    except Exception as exc:  # noqa: BLE001
        return _warn_error(exc)
    g, thr = c["by_fold"][fold], _thresholds(fold)
    agg = g.groupby("ts").agg(surges=("surge", "sum"), drops=("drop", "sum"), median_z=("z", "median"),
                              prcp=("prcp", "first"), temp=("temp", "first"),
                              events=("event_egress_att", lambda s: int((s > 0).sum())))
    for k, t in thr.items():
        agg[f"alarms_{k}"] = g.assign(_a=g[f"p_{k}"] >= t).groupby("ts")["_a"].sum()
    agg = agg.reset_index()
    agg["ts"] = agg["ts"].map(pd.Timestamp.isoformat)
    return JSONResponse(_json_safe(agg.round(3).to_dict(orient="list")))


async def warn_hour(request: Request) -> JSONResponse:
    try:
        fold, c = _fold(request), _warn_cache()
        ts = pd.Timestamp(request.query_params["ts"])
    except Exception as exc:  # noqa: BLE001
        return _warn_error(exc)
    g, thr = c["by_fold"][fold], _thresholds(fold)
    rows = g[g["ts"] == ts]
    out = []
    for r in rows.itertuples(index=False):
        d = r._asdict()
        rec = {"station": d["station"], "actual": round(d["passengers"], 1), "normal": round(d["normal_passengers"], 1),
               "z": round(d["z"], 2), "driver": d["driver"], "event_att": d["event_egress_att"] + d["event_ingress_att"],
               "closed": int(d["station_closed"]), "suspended": int(d["line_suspended_here"]), "prcp": d["prcp"]}
        for k, t in thr.items():
            rec[f"p_{k}"] = round(d[f"p_{k}"], 4)
            rec[f"z_hat_{k}"] = round(d[f"z_hat_{k}"], 2)
            rec[f"alarm_{k}"] = bool(d[f"p_{k}"] >= t)
        out.append(rec)
    return JSONResponse(_json_safe({"ts": ts.isoformat(), "stations": out}))


def _series(fold: str, station: str, lo: pd.Timestamp | None = None, hi: pd.Timestamp | None = None) -> list[dict]:
    c = _warn_cache()
    g = c["by_fold"][fold]
    g = g[g["station"] == station]
    if lo is not None:
        g = g[(g["ts"] >= lo) & (g["ts"] <= hi)]
    thr = _thresholds(fold)
    cols = ["ts", "passengers", "normal_passengers", "z", "driver", "event_egress_att", "station_closed", "prcp"]
    cols += [f"{p}_{k}" for k in thr for p in ("p", "z_hat", "pax")]
    recs = g[cols].round(3).to_dict(orient="records")
    for r in recs:
        r["ts"] = r["ts"].isoformat()
        for k, t in thr.items():
            r[f"alarm_{k}"] = r[f"p_{k}"] >= t
    return recs


async def warn_station(request: Request) -> JSONResponse:
    try:
        fold, station = _fold(request), request.query_params["station"]
        return JSONResponse(_json_safe({"station": station, "fold": fold, "series": _series(fold, station),
                                        "thresholds": _thresholds(fold)}))
    except Exception as exc:  # noqa: BLE001
        return _warn_error(exc)


async def warn_scenario(request: Request) -> JSONResponse:
    try:
        c = _warn_cache()
        s = next(x for x in c["scenarios"] if x["id"] == request.query_params["id"])
    except StopIteration:
        return JSONResponse({"error": "unknown scenario"}, status_code=404)
    except Exception as exc:  # noqa: BLE001
        return _warn_error(exc)
    out = {k: v for k, v in s.items() if k != "series"}
    if s["station"]:
        out["series"] = _series(s["fold"], s["station"], pd.Timestamp(s["start"]).floor("h"), pd.Timestamp(s["end"]).ceil("h"))
    else:  # network-wide scenario (rain): aggregated in the engine, same keys as the station series
        out["series"] = [{**{k: v for k, v in r.items() if k not in ("actual", "normal")},
                          "passengers": r["actual"], "normal_passengers": r["normal"]} for r in s["series"]]
    out["thresholds"] = _thresholds(s["fold"])
    return JSONResponse(_json_safe(out))


_LIVE_BUNDLES: dict[str, dict] = {}


async def warn_live(request: Request) -> JSONResponse:
    """Re-score one scenario's station-hours with a live TabPFN-3.5 call (fit once per fold per process)."""
    body = await request.json()
    try:
        s = next(x for x in _warn_cache()["scenarios"] if x["id"] == body["scenario_id"])
        table = warn.anomaly_table(s["fold"])
        t0 = time.perf_counter()
        fitted_now = s["fold"] not in _LIVE_BUNDLES
        if fitted_now:
            _LIVE_BUNDLES[s["fold"]] = warn.fit_tabpfn(warn.context_sample(table[table["split"] == "train"]))
        bundle = _LIVE_BUNDLES[s["fold"]]
        rows = table[(table["split"] == "test") & (table["ts"] >= pd.Timestamp(s["start"]).floor("h"))
                     & (table["ts"] <= pd.Timestamp(s["end"]).ceil("h"))]
        rows = rows[rows["station"] == s["station"]] if s["station"] else rows
        p = warn.predict_tabpfn(bundle, rows)
        if not s["station"]:
            p = p.assign(ts=rows["ts"].to_numpy()).groupby("ts").median()
            idx = [t.isoformat() for t in p.index]
        else:
            idx = [t.isoformat() for t in rows["ts"]]
        thr = _thresholds(s["fold"]).get("tabpfn")
        return JSONResponse(_json_safe({
            "scenario_id": s["id"], "n_rows": len(rows), "fitted_now": fitted_now,
            "fit_s": round(bundle["fit_s"], 2) if fitted_now else 0.0,
            "predict_s": round(p.attrs.get("predict_s", 0.0), 2),
            "total_s": round(time.perf_counter() - t0, 2), "threshold": thr,
            "series": [{"ts": t, "p": round(float(a), 4), "z_hat": round(float(b), 2)}
                       for t, a, b in zip(idx, p["p_surge"], p["z_hat"])],
        }))
    except StopIteration:
        return JSONResponse({"error": "unknown scenario"}, status_code=404)
    except SystemExit as exc:  # missing TABPFN_API_TOKEN (config.tabpfn_token raises SystemExit)
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:  # noqa: BLE001
        return _warn_error(exc)


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


routes = [
    Route("/", index),
    Route("/closures", closures_page),
    Route("/api/health", health),
    Route("/api/warning/network", warn_network),
    Route("/api/warning/summary", warn_summary),
    Route("/api/warning/timeline", warn_timeline),
    Route("/api/warning/hour", warn_hour),
    Route("/api/warning/station", warn_station),
    Route("/api/warning/scenario", warn_scenario),
    Route("/api/warning/live", warn_live, methods=["POST"]),
    Route("/api/topology", topology),
    Route("/api/impact/xgboost", impact_xgboost, methods=["POST"]),
    Route("/api/impact/tabpfn", impact_tabpfn, methods=["POST"]),
    Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
]

app = Starlette(routes=routes)
