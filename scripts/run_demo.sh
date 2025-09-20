#!/bin/bash
# SAFETY: No user-provided command execution
poetry run python -c "from src.agent_orchestrator import AgentOrchestrator; agent = AgentOrchestrator(); print(agent.run_mode('coding', 'demo run'))"
