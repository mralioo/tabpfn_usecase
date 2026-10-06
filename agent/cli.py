#!/usr/bin/env python
"""Interactive CLI chat against the agent harness.

    python agent/cli.py
    > What should we watch on the network on Monday 28 September at 22:00?
    > A 3,000-person concert ends at Olympia-Stadion on 29 Sep at 22:00 — what happens?

Needs TABPFN_API_TOKEN (for the MCP tools) and an LLM key the harness's
`AGENT_LITELLM_MODEL` can reach (OPENAI_API_KEY by default) — see .env.example.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.harness import ask_sync


def main() -> None:
    print("TabPFN-3.5 U-Bahn early-warning agent — ask about anomalies, the network outlook or a what-if. "
          "Ctrl-D to quit.\n")
    history = None
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        out = ask_sync(question, history)
        history = out["history"]
        for step in out["trace"]:
            if step["step"] == "tool":
                print(f"  · {step['tool']}({json.dumps(step['args'])}) — {step['seconds']} s")
        print("\n" + out["answer"], "\n")


if __name__ == "__main__":
    main()
