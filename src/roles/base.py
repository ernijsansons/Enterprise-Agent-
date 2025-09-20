"""Role primitives leveraged by the agent orchestrator."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover - for linting only
    from src.agent_orchestrator import AgentOrchestrator


class BaseRole:
    """Shared helpers for role implementations."""

    def __init__(self, orchestrator: "AgentOrchestrator") -> None:
        self.orchestrator = orchestrator

    # ------------------------------------------------------------------ helpers
    def call_model(self, model: str, prompt: str, role: str, operation: str, **kwargs) -> str:
        # Pass project context if available
        project_context = kwargs.get('project_context') or getattr(self.orchestrator, 'project_context', None)

        # Access role transition context if available (for Claude Code coordination)
        if hasattr(self.orchestrator, '_use_claude_code') and self.orchestrator._use_claude_code:
            # Get state from orchestrator if available
            current_state = getattr(self.orchestrator, '_current_state', {})
            role_context = current_state.get(f"{role.lower()}_context")

            if role_context:
                # Enhance prompt with role transition context for better coordination
                enhanced_prompt = self._enhance_prompt_with_role_context(prompt, role_context)
                prompt = enhanced_prompt

        return self.orchestrator._call_model(
            model, prompt, role, operation,
            project_context=project_context, **kwargs
        )

    def _enhance_prompt_with_role_context(self, prompt: str, role_context: Dict[str, Any]) -> str:
        """Enhance prompt with role transition context for better Claude coordination."""
        if not role_context:
            return prompt

        context_parts = []
        from_role = role_context.get("from_role", "")
        previous_outputs = role_context.get("previous_outputs", {})

        if from_role and previous_outputs:
            context_parts.append(f"<previous_role_context>")
            context_parts.append(f"Previous role: {from_role}")

            # Add relevant previous outputs
            for output_type, output_value in previous_outputs.items():
                if output_value:  # Only add non-empty outputs
                    if isinstance(output_value, (list, dict)):
                        context_parts.append(f"{output_type}: {str(output_value)[:200]}...")
                    else:
                        context_parts.append(f"{output_type}: {str(output_value)[:200]}...")

            context_parts.append("</previous_role_context>")
            context_parts.append("")

        context_parts.append(prompt)
        return "\n".join(context_parts)

    def route_to_model(self, text: str, domain: str, vuln_flag: bool = False) -> str:
        return self.orchestrator.route_to_model(text, domain, vuln_flag)

    def domain_pack(self, domain: str) -> Dict[str, Any]:
        return self.orchestrator.domain_packs.get(domain, {})

    def invoke_codex_cli(self, task_type: str, params: list[str], domain: str) -> str:
        return self.orchestrator._invoke_codex_cli(task_type, params, domain)

    def parse_json(self, text: str) -> Dict[str, Any]:
        return self.orchestrator._parse_json(text)

    def store_memory(self, scope: str, key: str, value: Any) -> None:
        self.orchestrator.memory.store(scope, key, value)

    def telemetry(self, event: Dict[str, Any]) -> None:
        telemetry = getattr(self.orchestrator, "telemetry", None)
        if callable(telemetry):
            telemetry(event)


__all__ = ["BaseRole"]
