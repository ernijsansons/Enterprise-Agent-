#!/bin/bash
# Demo run for UI domain
set -euo pipefail
poetry run python -c "from src.agent_orchestrator import AgentOrchestrator; agent = AgentOrchestrator(); print(agent.run_mode('ui', 'Design a thermonuclear SaaS analytics dashboard'))"
