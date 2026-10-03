"""A minimal single-agent tool-calling harness: one LLM call loop that decides
*which* TabPFN-3.5 tool a question needs and calls it over MCP — no multi-role
pipeline, no verifier, no dispatcher/analyst/inspector/writer split. This is
deliberately the opposite of NextMove's four-role architecture: the point
here is to show the hand-off pattern itself ("the same way an agent hands
over to a weather tool, it should hand over to a specialised model for
predictions") as plainly as possible.

The MCP server (`mcp_server/server.py`) is used in-process via fastmcp's
in-memory transport (`Client(mcp_server_instance)`) — real MCP client/server
protocol, no subprocess needed for the demo. Point `MCP_SERVER_CMD` at the
same script with stdio transport instead if you want a separate process.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tabpfn_lab.config import load_dotenv  # noqa: E402

load_dotenv()

AGENT_MODEL = __import__("os").environ.get("AGENT_LITELLM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a control-room assistant for the Berlin U-Bahn. You have TabPFN-3.5 "
    "prediction tools available over MCP. Pick the tool by what the operator is "
    "actually asking:\n"
    "- a crowd-safety / early-warning question ('will it be overcrowded', 'is it "
    "safe to run as scheduled') -> predict_overcrowding_risk\n"
    "- an operational-planning question ('how many passengers', 'expected demand') "
    "-> predict_expected_flow\n"
    "Call resolve_station first if the station name is informal. Call "
    "describe_dataset if you need the covered date range. Always state the "
    "predicted number/probability AND which model produced it. Be concise."
)


def _mcp_tools_to_llm_schema(tools) -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": getattr(t, "input_schema", None) or t.inputSchema or {"type": "object", "properties": {}},
        },
    } for t in tools]


async def ask(question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Run one turn: question -> (possibly) a TabPFN MCP tool call -> final answer.
    Returns (answer_text, updated_history) so a CLI can keep a conversation going."""
    import litellm

    from mcp_server.server import mcp  # the FastMCP server instance, imported in-process

    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_schema = _mcp_tools_to_llm_schema(tools)

        messages = history or [{"role": "system", "content": SYSTEM_PROMPT}]
        messages = messages + [{"role": "user", "content": question}]

        for _ in range(4):  # a couple of tool calls at most (resolve_station -> predict_*)
            resp = litellm.completion(model=AGENT_MODEL, messages=messages, tools=tool_schema)
            msg = resp.choices[0].message
            messages.append(msg.model_dump())

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                return msg.content or "", messages

            for call in tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = await client.call_tool(call.function.name, args)
                payload = result.data if hasattr(result, "data") else [c.text for c in result.content]
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": json.dumps(payload, default=str),
                })

        return "Reached the tool-call limit without a final answer — try rephrasing.", messages


def ask_sync(question: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    return asyncio.run(ask(question, history))
