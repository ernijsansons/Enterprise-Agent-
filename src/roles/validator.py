"""Validator role for domain-specific checks."""
from __future__ import annotations

import os
from typing import Any, Dict

from .base import BaseRole
from .validators import (
    validate_coding,
    validate_content,
    validate_real_estate,
    validate_social_media,
    validate_trading,
)


class Validator(BaseRole):
    def validate(self, output: str, domain: str) -> Dict[str, Any]:
        pack = self.domain_pack(domain)
        agent_cfg = getattr(self.orchestrator, "agent_cfg", {}) or {}
        workspace_root = agent_cfg.get("workspace_root") or os.getcwd()
        secrets = getattr(self.orchestrator, "secrets", {}) or {}

        # Add Claude-enhanced validation through LLM when appropriate
        llm_validation = self._perform_llm_validation(output, domain, pack)

        payload = {
            "output": output,
            "coverage_threshold": float(pack.get("coverage_threshold", 0.97)),
            "workspace": workspace_root,
            "secrets": secrets,
            "llm_validation": llm_validation,
        }
        domain_validators = {
            "coding": validate_coding,
            "social_media": validate_social_media,
            "content": validate_content,
            "trading": validate_trading,
            "real_estate": validate_real_estate,
        }
        validator_fn = domain_validators.get(domain)
        parsed = validator_fn(payload) if validator_fn else {}
        model = self.route_to_model(output, domain)

        # Combine traditional validation with LLM insights
        result = {"raw": parsed, "parsed": parsed, "model": model}
        if llm_validation:
            result["llm_insights"] = llm_validation

        return result

    def _perform_llm_validation(self, output: str, domain: str, pack: Dict[str, Any]) -> Dict[str, Any]:
        """Perform LLM-based validation for quality insights."""
        try:
            validation_prompt = self._build_validation_prompt(output, domain, pack)
            model = self.route_to_model(output, domain)

            validation_result = self.call_model(model, validation_prompt, "Validator", "validate")
            parsed_result = self.parse_json(validation_result)

            return parsed_result if isinstance(parsed_result, dict) else {"insights": validation_result}
        except Exception as e:
            return {"error": f"LLM validation failed: {str(e)}"}

    def _build_validation_prompt(self, output: str, domain: str, pack: Dict[str, Any]) -> str:
        """Build Claude-optimized validation prompt."""
        criteria = pack.get("validation_criteria", "correctness, completeness, quality")

        prompt_parts = [
            "You are a quality assurance expert performing detailed validation.",
            "",
            "Your task is to systematically validate the provided output for quality and correctness.",
            "",
            "## Validation Process",
            "1. **Check correctness** - Is the output accurate and error-free?",
            "2. **Verify completeness** - Does it fully address the requirements?",
            "3. **Assess quality** - Is it well-structured and maintainable?",
            "4. **Identify risks** - Are there any potential issues or vulnerabilities?",
            "",
            f"## Domain: {domain}",
            f"## Validation Criteria: {criteria}",
            "",
            "## Output to Validate",
            "```",
            output,
            "```",
            "",
            "## Response Format",
            "Provide validation results as JSON:",
            '{"passed": boolean, "score": float, "issues": [list of issues], "recommendations": [list of recommendations]}',
        ]

        return "\n".join(prompt_parts)


__all__ = ["Validator"]
