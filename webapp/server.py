"""Closure Impact Lab — the interactive map + chat dashboard that runs real XGBoost
and TabPFN-3.5 calls side by side on hypothetical Berlin U-Bahn station/line closures.
A genuine second dashboard next to `dashboard/app.py` (Streamlit): that one explores
the datasets, this one runs live what-if comparisons through `tabpfn_lab.closure_impact`.

Built on Starlette + uvicorn (already pulled in by `fastmcp`/`streamlit`, so this adds
no new runtime dependency) rather than FastAPI, in keeping with the rest of this repo's
"smallest harness that demonstrates the pattern" approach.

Run: `make dashboard-closures` (http://127.0.0.1:8000), or directly:
    .venv/bin/uvicorn webapp.server:app --reload
"""
from __future__ import annotations

import sys
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
from tabpfn_lab import closure_impact as engine  # noqa: E402

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
    return FileResponse(STATIC_DIR / "index.html")


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


routes = [
    Route("/", index),
    Route("/api/health", health),
    Route("/api/topology", topology),
    Route("/api/impact/xgboost", impact_xgboost, methods=["POST"]),
    Route("/api/impact/tabpfn", impact_tabpfn, methods=["POST"]),
    Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
]

app = Starlette(routes=routes)
