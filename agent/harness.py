"""Control-room agent: one LLM tool-calling loop over the TabPFN-3.5 MCP server.

The operator asks a question in plain language. The LLM picks MCP tools
(`mcp_server/server.py`), the tools run the TabPFN-3.5 engine (`tabpfn_lab/operations.py`), and
the LLM writes the answer from the returned numbers only. Every tool call is recorded in a
trace (tool, arguments, latency, result) so the webapp can show exactly what was computed.

The MCP server is used in-process through fastmcp's in-memory transport (`Client(mcp)`): the
real MCP client/server protocol, without a subprocess. Any MCP client can launch
`python mcp_server/server.py` over stdio instead.

LLM: any LiteLLM model (`AGENT_LITELLM_MODEL`, default gpt-4o-mini) — see .env.example.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from fastmcp import Client

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tabpfn_lab.config import load_dotenv  # noqa: E402

load_dotenv()

AGENT_MODEL = os.environ.get("AGENT_LITELLM_MODEL", "gpt-4o-mini")
MAX_ROUNDS = 6

SYSTEM_PROMPT = """You are the control-room assistant for the Berlin U-Bahn (BVG network, Alstom-simulated flows).
You answer with numbers computed by TabPFN-3.5 through your MCP tools. Never invent numbers.

How the engine works: each station's hourly flow is compared with its normal for that day type and hour.
z = deviation in robust standard units; surge = z >= 2, collapse = z <= -2. An alarm = a model's top 2% of
station-hours. Forecasts use what is known a day ahead (event calendar, planned closures, weather).
Valid timestamps: Berlin local time, on the hour, inside the forecast windows (call describe_network if unsure).
Service hours are 05:00-23:00 plus 00:00 (00:00 IS valid); only 01:00-04:00 have no service. Never refuse a
timestamp yourself — call the tool and let it report an error if there is one.

Pick tools by the question:
- network / "what should I watch" at a time -> network_outlook (then line_corridor for a busy line)
- one station over hours -> station_forecast (live TabPFN-3.5)
- "why" -> explain_anomaly
- hypothetical (extra event, closure, rain) -> what_if (live TabPFN-3.5)
- past incidents / examples -> list_scenarios
- model quality, TabPFN vs XGBoost -> model_scoreboard (lead with its key_findings, both the wins and the losses)
Resolve informal station names with resolve_station when needed.

Answer format (concise, operator tone, markdown, no numbered section labels):
- First line in bold: the key FORECAST (from the model, not the observed value) in words, e.g. "3 surge alarms at 22:00 — network at +40% vs normal" or
  "Olympia-Stadion: 565 riders/h vs 72 normal, surge probability 82% → alarm".
- 2-5 bullets: drivers, where on the network (stations, lines; for a closure use neighbours_peak_change), and
  the observed outcome when the tool gives
  `observed_replay` (the windows replay past days: say "observed" for those values).
- "**Action:**" one or two concrete steps (extra trains/short-turns, platform staff, crowd control at exits,
  announcements, replacement buses, reroute via named neighbouring stations).
- Last line in italics, starting "Source:", naming the engine as given in the tool's inference.engine value
  (do not print the field name), plus timings ONLY if the tool returned them (fit_s / predict_s). Never claim a
  live call when the engine says precomputed.
Every number you write must appear in a tool result; probabilities belong to a named station-hour.
There is no "collapse probability": surge_probability is only about surges; describe a closure/collapse with
the forecast passengers and forecast_z (z <= -2 = collapse).
The page shows charts of the tool results under your answer, so you may refer to them ("see the evidence charts").
If a tool returns an error, explain it plainly and suggest a valid input."""


def _mcp_tools_to_llm_schema(tools) -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {"type": "object", "properties": {}},
        },
    } for t in tools]


def _payload(result) -> object:
    data = getattr(result, "structured_content", None)
    if data is not None:
        return data.get("result", data) if isinstance(data, dict) and set(data) == {"result"} else data
    return [getattr(c, "text", str(c)) for c in result.content]


def _preview(payload: object, limit: int = 600) -> str:
    text = json.dumps(payload, default=str)
    return text if len(text) <= limit else text[:limit] + " …"


async def ask(question: str, history: list[dict] | None = None) -> dict:
    """One operator turn. Returns {answer, trace, history, model, seconds}."""
    import litellm

    from mcp_server.server import mcp

    t_start = time.perf_counter()
    trace: list[dict] = []
    async with Client(mcp) as client:
        tool_schema = _mcp_tools_to_llm_schema(await client.list_tools())
        messages = list(history or [{"role": "system", "content": SYSTEM_PROMPT}])
        messages.append({"role": "user", "content": question})

        for _ in range(MAX_ROUNDS):
            t0 = time.perf_counter()
            resp = await litellm.acompletion(model=AGENT_MODEL, messages=messages, tools=tool_schema, temperature=0.2)
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            tool_calls = getattr(msg, "tool_calls", None)
            trace.append({"step": "llm", "model": AGENT_MODEL, "seconds": round(time.perf_counter() - t0, 2),
                          "decision": [c.function.name for c in tool_calls] if tool_calls else "answer"})
            if not tool_calls:
                return {"answer": msg.content or "", "trace": trace, "history": messages, "model": AGENT_MODEL,
                        "seconds": round(time.perf_counter() - t_start, 2)}

            for call in tool_calls:
                args = json.loads(call.function.arguments or "{}")
                t0 = time.perf_counter()
                try:
                    payload = _payload(await client.call_tool(call.function.name, args))
                except Exception as exc:  # noqa: BLE001 — surface tool errors to the LLM, it can recover
                    payload = {"error": str(exc)}
                trace.append({"step": "tool", "tool": call.function.name, "args": args,
                              "seconds": round(time.perf_counter() - t0, 2), "result_preview": _preview(payload),
                              "result": payload})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(payload, default=str)})

    return {"answer": "I reached the tool-call limit without a final answer — try a narrower question.",
            "trace": trace, "history": messages, "model": AGENT_MODEL, "seconds": round(time.perf_counter() - t_start, 2)}


def ask_sync(question: str, history: list[dict] | None = None) -> dict:
    return asyncio.run(ask(question, history))
