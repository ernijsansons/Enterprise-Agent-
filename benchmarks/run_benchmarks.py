"""Lightweight benchmark harness for key domains."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from src.agent_orchestrator import AgentOrchestrator

DOMAINS: List[str] = ["coding", "social_media", "trading"]
BASELINE_PATH = Path("benchmarks/baseline.json")


def benchmark_domain(domain: str) -> Dict[str, float]:
    agent = AgentOrchestrator()
    start = time.time()
    result = agent.run_mode(domain, "Benchmark task")
    duration = time.time() - start
    success = float(result.get("confidence", 0.0) >= 0.8)
    return {"domain": domain, "duration": duration, "success": success}


def run_benchmarks() -> List[Dict[str, float]]:
    return [benchmark_domain(domain) for domain in DOMAINS]


def compare_to_baseline(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not BASELINE_PATH.exists():
        return {}
    baseline = {item["domain"]: item for item in json.loads(BASELINE_PATH.read_text())}
    deltas: Dict[str, Dict[str, float]] = {}
    for result in results:
        base = baseline.get(result["domain"])
        if not base:
            continue
        deltas[result["domain"]] = {
            "success_delta": result["success"] - base.get("success", 0),
            "duration_delta": result["duration"] - base.get("duration", result["duration"]),
        }
    return deltas


def main() -> None:  # pragma: no cover
    results = run_benchmarks()
    deltas = compare_to_baseline(results)
    payload = {"results": results, "deltas": deltas}
    output = Path("benchmark_results.json")
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
