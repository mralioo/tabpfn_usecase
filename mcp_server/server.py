"""FastMCP server: the Berlin U-Bahn anomaly early-warning engine as MCP tools.

Every tool is a thin wrapper over `tabpfn_lab/operations.py`. The live tools (`station_forecast`,
`what_if`) call TabPFN-3.5 through `tabpfn_client` at request time; the network-wide tools replay
the day-ahead forecasts that `make anomaly` computed for the test windows.

The LLM agent (`agent/harness.py`) connects to this server over MCP and picks the tool for the
operator's question. Any MCP client can use it the same way.

Run standalone (stdio transport, e.g. for Claude Desktop or another MCP client):
    python mcp_server/server.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tabpfn_lab import operations as ops  # noqa: E402
from tabpfn_lab.config import load_dotenv  # noqa: E402

load_dotenv()

mcp = FastMCP(
    "tabpfn-ubahn-early-warning",
    instructions=(
        "Early-warning tools for the Berlin U-Bahn, powered by TabPFN-3.5. Flows are normalised against each "
        "station's normal pattern for that day type and hour; z = deviation in robust standard units "
        "(surge z >= 2, collapse z <= -2). Forecasts use information known a day ahead: event calendar, planned "
        "closures, weather. Timestamps are Berlin local time on the hour inside the forecast windows (call "
        "describe_network). Network questions -> network_outlook / line_corridor. One station over time -> "
        "station_forecast (live TabPFN-3.5). 'Why' -> explain_anomaly. Hypotheticals (extra event, closure, rain) "
        "-> what_if (live TabPFN-3.5). Past incidents -> list_scenarios. Model quality -> model_scoreboard."
    ),
)


@mcp.tool
def describe_network() -> dict:
    """Coverage: stations, lines, forecast windows (valid timestamps), service hours, the anomaly target and
    the models. Call first when unsure whether a date/time is in scope."""
    return ops.describe()


@mcp.tool
def resolve_station(query: str, max_results: int = 5) -> list[str]:
    """Fuzzy-resolve an informal station name ('Alex', 'warschauer', 'Kotti' -> exact names)."""
    return [ops.short(n) for n in ops.resolve_station(query, max_results)] or [f"no station matches {query!r}"]


@mcp.tool
def network_outlook(timestamp: str, top: int = 8) -> dict:
    """Whole-network day-ahead outlook for one hour (TabPFN-3.5): stations with surge alarms and their drivers,
    load vs normal per line, weather, closures. Example timestamp: '2026-09-28 22:00'."""
    return ops.network_outlook(timestamp, top)


@mcp.tool
def line_corridor(line: str, timestamp: str) -> dict:
    """Forecast passenger load at every station along one line (U1–U9, running order, terminus to terminus) for
    one hour, vs normal — shows where along the corridor pressure builds or collapses."""
    return ops.line_corridor(line, timestamp)


@mcp.tool
def station_forecast(station: str, start: str, hours: int = 6, live: bool = True) -> dict:
    """Hour-by-hour forecast for one station: normal flow, TabPFN-3.5 forecast (live API call when live=True,
    takes seconds), surge probability and alarm, XGBoost comparison, drivers, and the observed outcome (replay)."""
    return ops.station_forecast(station, start, hours, live)


@mcp.tool
def explain_anomaly(station: str, timestamp: str) -> dict:
    """Why a station-hour is (or isn't) off its normal pattern: context drivers (events, closures, weather), all
    five models' forecasts, observed outcome, and how similar hours at this station behaved in training."""
    return ops.explain_anomaly(station, timestamp)


@mcp.tool
def what_if(station: str, start: str, hours: int = 4, event_attendance: int = 0, event_end: str | None = None,
            close_station: bool = False, rain_mm: float | None = None) -> dict:
    """Live TabPFN-3.5 counterfactual for a hypothetical situation at a station: an event of `event_attendance`
    people ending at `event_end`, closing the station (neighbours are scored too), and/or rain of `rain_mm` mm/h.
    Returns as-scheduled vs what-if forecast per hour, alarm flags and the peak change."""
    return ops.what_if(station, start, hours, event_attendance, event_end, close_station, rain_mm)


@mcp.tool
def list_scenarios(kind: str | None = None, window: str | None = None, limit: int = 8) -> list[dict]:
    """Real incidents in the test windows (kind: 'event' | 'closure' | 'rain'; window: 'holdout' | 'backtest'),
    ranked by observed anomaly size, with which models flagged each one and from which hour."""
    return ops.list_scenarios(kind, window, limit)


@mcp.tool
def model_scoreboard() -> dict:
    """Benchmark of profile baseline, XGBoost (clock-only, 10k sample, full history) and TabPFN-3.5 (10k context)
    on forecasting anomalies: PR-AUC, event-surge recall, closure-collapse recall, plus the XGBoost learning curve."""
    return ops.scoreboard()


if __name__ == "__main__":
    mcp.run()
