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
    def call_model(
        self, model: str, prompt: str, role: str, operation: str, **kwargs
    ) -> str:
        """Call model with enhanced error handling and validation.

        Args:
            model: Model identifier
            prompt: Input prompt
            role: Role context
            operation: Operation context
            **kwargs: Additional arguments

        Returns:
            Model response

        Raises:
            EnterpriseAgentError: On validation or model call failures
        """
        # Input validation
        if not isinstance(model, str) or not model.strip():
            from src.utils.errors import create_validation_error, ErrorCode
            raise create_validation_error(
                "Model must be a non-empty string",
                validation_type="model_name",
                error_code=ErrorCode.INVALID_PARAMETERS,
                context={"model": model, "role": role, "operation": operation}
            )

        if not isinstance(prompt, str) or not prompt.strip():
            from src.utils.errors import create_validation_error, ErrorCode
            raise create_validation_error(
                "Prompt must be a non-empty string",
                validation_type="prompt",
                error_code=ErrorCode.INVALID_PARAMETERS,
                context={"prompt_length": len(prompt) if prompt else 0, "role": role, "operation": operation}
            )

        try:
            # Pass project context if available
            project_context = kwargs.get("project_context") or getattr(
                self.orchestrator, "project_context", None
            )

            # Access role transition context if available (for Claude Code coordination)
            if (
                hasattr(self.orchestrator, "_use_claude_code")
                and self.orchestrator._use_claude_code
            ):
                # Get state from orchestrator if available
                current_state = getattr(self.orchestrator, "_current_state", {})
                role_context = current_state.get(f"{role.lower()}_context")

                if role_context:
                    # Enhance prompt with role transition context for better coordination
                    enhanced_prompt = self._enhance_prompt_with_role_context(
                        prompt, role_context
                    )
                    prompt = enhanced_prompt

            return self.orchestrator._call_model(
                model, prompt, role, operation, project_context=project_context, **kwargs
            )
        except Exception as e:
            # Enhance error with role context
            from src.utils.errors import handle_error, create_model_error, ErrorCode

            error_context = {
                "role": role,
                "operation": operation,
                "model": model,
                "prompt_length": len(prompt),
                "has_project_context": project_context is not None
            }

            if hasattr(e, 'details'):
                # Already a structured error, add context
                e.add_context("role_call_context", error_context)
                raise
            else:
                # Convert to structured error
                model_error = create_model_error(
                    f"Model call failed in {role}/{operation}: {str(e)}",
                    model=model,
                    context=error_context,
                    cause=e
                )
                handle_error(model_error)
                raise model_error

    def _enhance_prompt_with_role_context(
        self, prompt: str, role_context: Dict[str, Any]
    ) -> str:
        """Enhance prompt with role transition context for better Claude coordination."""
        if not role_context:
            return prompt

        context_parts = []
        from_role = role_context.get("from_role", "")
        previous_outputs = role_context.get("previous_outputs", {})

        if from_role and previous_outputs:
            context_parts.append("<previous_role_context>")
            context_parts.append(f"Previous role: {from_role}")

            # Add relevant previous outputs
            for output_type, output_value in previous_outputs.items():
                if output_value:  # Only add non-empty outputs
                    if isinstance(output_value, (list, dict)):
                        context_parts.append(
                            f"{output_type}: {str(output_value)[:200]}..."
                        )
                    else:
                        context_parts.append(
                            f"{output_type}: {str(output_value)[:200]}..."
                        )

            context_parts.append("</previous_role_context>")
            context_parts.append("")

        context_parts.append(prompt)
        return "\n".join(context_parts)

    def route_to_model(self, text: str, domain: str, vuln_flag: bool = False) -> str:
        return self.orchestrator.route_to_model(text, domain, vuln_flag)

    def domain_pack(self, domain: str) -> Dict[str, Any]:
        """Get domain configuration pack with validation.

        Args:
            domain: Domain name

        Returns:
            Domain configuration dictionary

        Raises:
            EnterpriseAgentError: If domain is invalid
        """
        if not isinstance(domain, str) or not domain.strip():
            from src.utils.errors import create_validation_error, ErrorCode
            raise create_validation_error(
                "Domain must be a non-empty string",
                validation_type="domain",
                error_code=ErrorCode.INVALID_DOMAIN,
                context={"domain": domain}
            )

        domain_packs = getattr(self.orchestrator, 'domain_packs', {})
        if domain not in domain_packs:
            from src.utils.errors import create_validation_error, ErrorCode
            available_domains = list(domain_packs.keys())
            raise create_validation_error(
                f"Unknown domain '{domain}'. Available domains: {available_domains}",
                validation_type="domain",
                error_code=ErrorCode.INVALID_DOMAIN,
                context={"domain": domain, "available_domains": available_domains}
            )

        return domain_packs.get(domain, {})

    def invoke_codex_cli(self, task_type: str, params: list[str], domain: str) -> str:
        return self.orchestrator._invoke_codex_cli(task_type, params, domain)

    def parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON with enhanced error handling.

        Args:
            text: JSON string to parse

        Returns:
            Parsed dictionary

        Raises:
            EnterpriseAgentError: On parsing failures
        """
        if not isinstance(text, str):
            from src.utils.errors import create_validation_error, ErrorCode
            raise create_validation_error(
                f"Expected string for JSON parsing, got {type(text).__name__}",
                validation_type="json_input",
                error_code=ErrorCode.VALIDATION_PARSE_ERROR,
                context={"input_type": type(text).__name__, "input_value": str(text)[:100]}
            )

        try:
            return self.orchestrator._parse_json(text)
        except Exception as e:
            from src.utils.errors import create_validation_error, ErrorCode
            raise create_validation_error(
                f"JSON parsing failed: {str(e)}",
                validation_type="json_parse",
                error_code=ErrorCode.VALIDATION_PARSE_ERROR,
                context={"text_length": len(text), "text_preview": text[:200]},
                cause=e
            )

    def store_memory(self, scope: str, key: str, value: Any) -> None:
        self.orchestrator.memory.store(scope, key, value)

    def telemetry(self, event: Dict[str, Any]) -> None:
        telemetry = getattr(self.orchestrator, "telemetry", None)
        if callable(telemetry):
            telemetry(event)


__all__ = ["BaseRole"]
