#!/usr/bin/env python3
"""
Enterprise Agent CLI - simple interface for running tasks
Usage: python agent_cli.py "your task description"
"""

import json
import sys
from pathlib import Path

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_orchestrator import AgentOrchestrator


def _normalise_plan(raw):
    if isinstance(raw, dict):
        return {
            "text": raw.get("text", ""),
            "epics": raw.get("epics", []),
            "model": raw.get("model"),
        }
    if raw is None:
        return {"text": "", "epics": [], "model": None}
    return {"text": str(raw), "epics": [], "model": None}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python agent_cli.py 'your task description'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    domain = "coding"
    print(f"[Enterprise Agent] Processing task: {task}")

    agent = AgentOrchestrator()
    result = agent.run_mode(domain, task)

    plan = _normalise_plan(result.get("plan"))
    code_output = result.get("code") or ""
    summary = {
        "plan": plan,
        "code": code_output,
        "confidence": result.get("confidence"),
        "needs_reflect": result.get("needs_reflect"),
        "governance_blocked": result.get("governance_blocked", False),
        "code_source": result.get("code_source"),
        "cost_summary": result.get("cost_summary", {}),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
