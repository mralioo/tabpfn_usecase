#!/usr/bin/env python
"""Interactive CLI chat against the agent harness.

    python agent/cli.py
    > will U Rudow (Berlin) be overcrowded at 2026-07-15 08:00:00?

Needs TABPFN_API_TOKEN (for the MCP tools) and an LLM key the harness's
`AGENT_LITELLM_MODEL` can reach (OPENAI_API_KEY by default) — see .env.example.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.harness import ask_sync


def main() -> None:
    print("TabPFN-3.5 U-Bahn agent — ask about crowd risk or expected flow. Ctrl-D to quit.\n")
    history = None
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        answer, history = ask_sync(question, history)
        print(answer, "\n")


if __name__ == "__main__":
    main()
