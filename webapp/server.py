"""Early-Warning Desk — the operator webapp for the Berlin U-Bahn anomaly use case.

Two pages:
- `/` (`static/warning.html`): stations coloured by forecast/actual anomaly z, a timeline of
  alarms vs actual surges, scenario replays with each model's verdict, a model scoreboard, a
  line-corridor view and a one-click live TabPFN-3.5 re-score.
- `/agent` (`static/agent.html`): the control-room agent (LLM -> MCP tools -> TabPFN-3.5). Each
  answer comes with evidence charts drawn from the exact tool results, plus the tool trace.

Data comes from `tabpfn_lab/operations.py` (which reads `results/anomaly/`, written by
`make anomaly`) and live TabPFN-3.5 calls. Starlette + uvicorn, no build step.

Run: `make app` (http://127.0.0.1:8000 and /agent), or `.venv/bin/uvicorn webapp.server:app --reload`.
"""
from __future__ import annotations

import sys
import time
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

from tabpfn_lab import anomaly as warn  # noqa: E402
from tabpfn_lab import operations as ops  # noqa: E402
from tabpfn_lab.config import load_dotenv  # noqa: E402
from tabpfn_lab.datasets.berlin import LINE_COLORS  # noqa: E402

load_dotenv()

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _json_safe(obj):
    """numpy/pandas scalars leak out of aggregates — convert them for JSONResponse."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _error(exc: BaseException) -> JSONResponse:
    if isinstance(exc, FileNotFoundError):
        return JSONResponse({"error": "No results in results/anomaly/ — run `make anomaly` (or `make anomaly-offline`) "
                                      "and reload."}, status_code=503)
    return JSONResponse({"error": str(exc)}, status_code=400)


def _fold(request: Request) -> str:
    fold = request.query_params.get("fold", "holdout")
    if fold not in warn.FOLDS:
        raise ValueError(f"unknown fold {fold!r}")
    return fold


async def index(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "warning.html")


async def agent_page(request: Request) -> FileResponse:
    return FileResponse(STATIC_DIR / "agent.html")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def network(request: Request) -> JSONResponse:
    meta = warn.station_meta()
    names = set(ops.station_names())
    stations = [{"name": r.station, "short": ops.short(r.station), "lon": float(r.lon), "lat": float(r.lat),
                 "lines": r.lines.split(","), "primary_line": r.primary_line}
                for r in meta.itertuples() if r.station in names]
    edges = [[a, b] for a, b in warn.station_graph().edges() if a in names and b in names]
    return JSONResponse({"stations": stations, "edges": edges, "line_colors": LINE_COLORS,
                         "lines": list(ops.line_sequences())})


async def summary(request: Request) -> JSONResponse:
    try:
        c = ops.cache()
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    scen = [{k: v for k, v in s.items() if k != "series"} for s in c["scenarios"]]
    folds = {f: {"label": warn.FOLDS[f]["label"], "hours": [t.isoformat() for t in sorted(g["ts"].unique())]}
             for f, g in c["by_fold"].items()}
    models = {v["key"]: n for n, v in c["metrics"]["folds"]["holdout"]["models"].items()}
    return JSONResponse(_json_safe({"metrics": c["metrics"], "scenarios": scen, "folds": folds, "models": models}))


async def timeline(request: Request) -> JSONResponse:
    try:
        fold = _fold(request)
        g, thr = ops.cache()["by_fold"][fold], ops.thresholds(fold)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    agg = g.groupby("ts").agg(surges=("surge", "sum"), drops=("drop", "sum"), median_z=("z", "median"),
                              prcp=("prcp", "first"), temp=("temp", "first"))
    for k, t in thr.items():
        agg[f"alarms_{k}"] = g.assign(_a=g[f"p_{k}"] >= t).groupby("ts")["_a"].sum()
    agg = agg.reset_index()
    agg["ts"] = agg["ts"].map(pd.Timestamp.isoformat)
    return JSONResponse(_json_safe(agg.round(3).to_dict(orient="list")))


async def hour(request: Request) -> JSONResponse:
    try:
        fold, ts = _fold(request), pd.Timestamp(request.query_params["ts"])
        g, thr = ops.cache()["by_fold"][fold], ops.thresholds(fold)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    out = []
    for d in g[g["ts"] == ts].to_dict(orient="records"):
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
    g = ops.cache()["by_fold"][fold]
    g = g[g["station"] == station]
    if lo is not None:
        g = g[(g["ts"] >= lo) & (g["ts"] <= hi)]
    thr = ops.thresholds(fold)
    cols = ["ts", "passengers", "normal_passengers", "z", "driver", "event_egress_att", "station_closed", "prcp"]
    cols += [f"{p}_{k}" for k in thr for p in ("p", "z_hat", "pax")]
    recs = g[cols].round(3).to_dict(orient="records")
    for r in recs:
        r["ts"] = r["ts"].isoformat()
        for k, t in thr.items():
            r[f"alarm_{k}"] = r[f"p_{k}"] >= t
    return recs


async def station(request: Request) -> JSONResponse:
    try:
        fold, name = _fold(request), request.query_params["station"]
        return JSONResponse(_json_safe({"station": name, "fold": fold, "series": _series(fold, name),
                                        "thresholds": ops.thresholds(fold)}))
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


async def scenario(request: Request) -> JSONResponse:
    try:
        s = next(x for x in ops.cache()["scenarios"] if x["id"] == request.query_params["id"])
    except StopIteration:
        return JSONResponse({"error": "unknown scenario"}, status_code=404)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)
    out = {k: v for k, v in s.items() if k != "series"}
    if s["station"]:
        out["series"] = _series(s["fold"], s["station"], pd.Timestamp(s["start"]).floor("h"), pd.Timestamp(s["end"]).ceil("h"))
    else:  # network-wide scenario (rain): aggregated in the engine, same keys as a station series
        out["series"] = [{**{k: v for k, v in r.items() if k not in ("actual", "normal")},
                          "passengers": r["actual"], "normal_passengers": r["normal"]} for r in s["series"]]
    out["thresholds"] = ops.thresholds(s["fold"])
    return JSONResponse(_json_safe(out))


async def corridor(request: Request) -> JSONResponse:
    try:
        return JSONResponse(_json_safe(ops.line_corridor(request.query_params["line"], request.query_params["ts"])))
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


async def live(request: Request) -> JSONResponse:
    """Re-score one scenario's station-hours with a live TabPFN-3.5 call."""
    body = await request.json()
    try:
        s = next(x for x in ops.cache()["scenarios"] if x["id"] == body["scenario_id"])
        table = warn.anomaly_table(s["fold"])
        rows = table[(table["split"] == "test") & (table["ts"] >= pd.Timestamp(s["start"]).floor("h"))
                     & (table["ts"] <= pd.Timestamp(s["end"]).ceil("h"))]
        rows = rows[rows["station"] == s["station"]] if s["station"] else rows
        t0 = time.perf_counter()
        scored, info = ops._score(s["fold"], rows)
        if not s["station"]:
            scored = scored.groupby("ts")[["p_surge", "z_hat"]].median().reset_index()
        return JSONResponse(_json_safe({
            "scenario_id": s["id"], **info, "total_s": round(time.perf_counter() - t0, 2),
            "threshold": ops.thresholds(s["fold"]).get("tabpfn"),
            "series": [{"ts": r.ts.isoformat(), "p": round(float(r.p_surge), 4), "z_hat": round(float(r.z_hat), 2)}
                       for r in scored.itertuples()],
        }))
    except StopIteration:
        return JSONResponse({"error": "unknown scenario"}, status_code=404)
    except Exception as exc:  # noqa: BLE001
        return _error(exc)


async def agent_ask(request: Request) -> JSONResponse:
    """Control-room agent: LLM (LiteLLM) -> MCP tools -> TabPFN-3.5. Returns the answer and the tool trace."""
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)
    try:
        from agent.harness import ask
        out = await ask(question, body.get("history"))
    except Exception as exc:  # noqa: BLE001 — usually a missing/invalid LLM key
        return JSONResponse({"error": f"Agent unavailable: {exc}. Set AGENT_LITELLM_MODEL and its API key in .env."},
                            status_code=503)
    return JSONResponse(_json_safe(out))


routes = [
    Route("/", index),
    Route("/agent", agent_page),
    Route("/api/health", health),
    Route("/api/warning/network", network),
    Route("/api/warning/summary", summary),
    Route("/api/warning/timeline", timeline),
    Route("/api/warning/hour", hour),
    Route("/api/warning/station", station),
    Route("/api/warning/scenario", scenario),
    Route("/api/warning/corridor", corridor),
    Route("/api/warning/live", live, methods=["POST"]),
    Route("/api/agent/ask", agent_ask, methods=["POST"]),
    Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
]

app = Starlette(routes=routes)
