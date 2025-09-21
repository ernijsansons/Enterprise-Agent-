"""Validator role for domain-specific checks."""
from __future__ import annotations

import os
import time
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
        """Validate output with comprehensive error handling and actionable feedback.

        Args:
            output: Output to validate
            domain: Domain context

        Returns:
            Validation results with structured feedback

        Raises:
            EnterpriseAgentError: On validation setup or execution failures
        """
        # Input validation
        if not isinstance(output, str):
            from src.utils.errors import ErrorCode, create_validation_error

            raise create_validation_error(
                f"Output must be a string, got {type(output).__name__}",
                validation_type="output_type",
                error_code=ErrorCode.INVALID_PARAMETERS,
                context={"output_type": type(output).__name__, "domain": domain},
            )

        if not output.strip():
            from src.utils.errors import ErrorCode, create_validation_error

            raise create_validation_error(
                "Output cannot be empty",
                validation_type="output_content",
                error_code=ErrorCode.INVALID_PARAMETERS,
                context={"output_length": len(output), "domain": domain},
            )

        try:
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
            if not validator_fn:
                from src.utils.errors import ErrorCode, create_validation_error

                available_domains = list(domain_validators.keys())
                raise create_validation_error(
                    f"No validator available for domain '{domain}'. Available: {available_domains}",
                    validation_type="domain_validator",
                    error_code=ErrorCode.VALIDATION_FAILED,
                    context={"domain": domain, "available_domains": available_domains},
                )

            # Execute domain-specific validation with error handling
            try:
                parsed = validator_fn(payload)
            except Exception as e:
                from src.utils.errors import ErrorCode, create_validation_error

                raise create_validation_error(
                    f"Domain validation failed for '{domain}': {str(e)}",
                    validation_type="domain_execution",
                    error_code=ErrorCode.VALIDATION_FAILED,
                    context={"domain": domain, "error": str(e)},
                    cause=e,
                )

            model = self.route_to_model(output, domain)

            # Enhanced result with actionable feedback
            result = self._build_enhanced_result(parsed, llm_validation, model, domain)

            # Add actionable failure analysis
            if not result["parsed"].get("passes", True):
                result["actionable_feedback"] = self._generate_actionable_feedback(
                    result["parsed"], domain, output
                )

            return result

        except Exception as e:
            # Ensure all validation errors are properly structured
            if hasattr(e, "details"):
                raise  # Already a structured error
            else:
                from src.utils.errors import ErrorCode, create_validation_error

                raise create_validation_error(
                    f"Validation process failed: {str(e)}",
                    validation_type="validation_process",
                    error_code=ErrorCode.VALIDATION_FAILED,
                    context={"domain": domain, "output_length": len(output)},
                    cause=e,
                )

    def _perform_llm_validation(
        self, output: str, domain: str, pack: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform LLM-based validation for quality insights with enhanced error handling."""
        try:
            validation_prompt = self._build_validation_prompt(output, domain, pack)
            model = self.route_to_model(output, domain)

            validation_result = self.call_model(
                model, validation_prompt, "Validator", "validate"
            )
            parsed_result = self.parse_json(validation_result)

            # Validate LLM response structure
            if isinstance(parsed_result, dict):
                # Ensure required fields are present
                required_fields = ["passed", "score"]
                missing_fields = [f for f in required_fields if f not in parsed_result]
                if missing_fields:
                    parsed_result[
                        "validation_warnings"
                    ] = f"Missing fields: {missing_fields}"
                    # Set defaults for missing fields
                    if "passed" not in parsed_result:
                        parsed_result["passed"] = False
                    if "score" not in parsed_result:
                        parsed_result["score"] = 0.0

                return parsed_result
            else:
                return {
                    "insights": validation_result,
                    "validation_warnings": "Non-structured response",
                }

        except Exception as e:
            from src.utils.errors import (
                ErrorCode,
                create_validation_error,
                handle_error,
            )

            error = create_validation_error(
                f"LLM validation failed: {str(e)}",
                validation_type="llm_validation",
                error_code=ErrorCode.VALIDATION_FAILED,
                context={"domain": domain, "output_length": len(output)},
                cause=e,
            )
            handle_error(error)
            return {
                "error": f"LLM validation failed: {str(e)}",
                "fallback_used": True,
                "error_details": error.details.to_dict(),
            }

    def _build_validation_prompt(
        self, output: str, domain: str, pack: Dict[str, Any]
    ) -> str:
        """Build Claude-optimized validation prompt with enhanced criteria."""
        criteria = pack.get("validation_criteria", "correctness, completeness, quality")
        coverage_threshold = pack.get("coverage_threshold", 0.97)

        prompt_parts = [
            "You are a quality assurance expert performing detailed validation.",
            "",
            "Your task is to systematically validate the provided output for quality and correctness.",
            "Focus on providing ACTIONABLE feedback that enables immediate improvements.",
            "",
            "## Validation Process",
            "1. **Check correctness** - Is the output accurate and error-free?",
            "2. **Verify completeness** - Does it fully address the requirements?",
            "3. **Assess quality** - Is it well-structured and maintainable?",
            "4. **Identify risks** - Are there any potential issues or vulnerabilities?",
            "5. **Provide actionable fixes** - Give specific, implementable solutions",
            "",
            f"## Domain: {domain}",
            f"## Validation Criteria: {criteria}",
            f"## Quality Threshold: {coverage_threshold:.0%}",
            "",
            "## Output to Validate",
            "```",
            output[:5000]
            + (
                "...\n[TRUNCATED]" if len(output) > 5000 else ""
            ),  # Prevent token overflow
            "```",
            "",
            "## Response Format",
            "Provide validation results as JSON with actionable feedback:",
            "{",
            '  "passed": boolean,',
            '  "score": float (0.0-1.0),',
            '  "issues": [',
            '    {"type": "error|warning|style", "description": "specific issue", "line": number_or_null, "fix": "how to fix this"}',
            "  ],",
            '  "recommendations": [',
            '    {"category": "performance|security|maintainability|correctness", "description": "specific recommendation", "priority": "high|medium|low"}',
            "  ],",
            '  "summary": "brief overall assessment"',
            "}",
        ]

        return "\n".join(prompt_parts)

    def _build_enhanced_result(
        self,
        parsed: Dict[str, Any],
        llm_validation: Dict[str, Any],
        model: str,
        domain: str,
    ) -> Dict[str, Any]:
        """Build enhanced validation result with comprehensive feedback."""
        result = {"raw": parsed, "parsed": parsed, "model": model, "domain": domain}

        if llm_validation:
            result["llm_insights"] = llm_validation

            # Merge LLM insights with traditional validation
            if "score" in llm_validation:
                result["combined_score"] = self._calculate_combined_score(
                    parsed, llm_validation
                )

        # Add validation metadata
        result["validation_metadata"] = {
            "timestamp": time.time(),
            "domain": domain,
            "model_used": model,
            "has_llm_validation": bool(llm_validation),
            "validation_type": "enhanced",
        }

        return result

    def _calculate_combined_score(
        self, traditional: Dict[str, Any], llm: Dict[str, Any]
    ) -> float:
        """Calculate combined score from traditional and LLM validation."""
        traditional_score = 1.0 if traditional.get("passes", False) else 0.0
        llm_score = float(llm.get("score", 0.0))

        # Weight traditional validation more heavily for concrete metrics
        return (traditional_score * 0.6) + (llm_score * 0.4)

    def _generate_actionable_feedback(
        self, validation_result: Dict[str, Any], domain: str, output: str
    ) -> Dict[str, Any]:
        """Generate actionable feedback for validation failures."""
        feedback = {
            "immediate_actions": [],
            "improvement_suggestions": [],
            "domain_specific_guidance": [],
            "priority_fixes": [],
        }

        # Domain-specific actionable feedback
        if domain == "coding":
            feedback["domain_specific_guidance"].extend(
                [
                    "Run tests locally before validation",
                    "Check code formatting with linter",
                    "Ensure all imports are correct",
                    "Verify function signatures match requirements",
                ]
            )

            if not validation_result.get("tests_passed", True):
                feedback["immediate_actions"].append(
                    "Fix failing tests - check pytest output for specific errors"
                )
                feedback["priority_fixes"].append("test_failures")

            coverage = validation_result.get("coverage", 0.0)
            threshold = validation_result.get("coverage_threshold", 0.97)
            if coverage < threshold:
                missing_coverage = (threshold - coverage) * 100
                feedback["immediate_actions"].append(
                    f"Add tests to increase coverage by {missing_coverage:.1f}%"
                )
                feedback["priority_fixes"].append("insufficient_coverage")

        elif domain == "social_media":
            if not validation_result.get("tone_ok", True):
                feedback["immediate_actions"].append(
                    "Revise content tone to be more professional"
                )
            length = validation_result.get("length", 0)
            if length > 280:
                feedback["immediate_actions"].append(
                    f"Reduce content length by {length - 280} characters"
                )

        elif domain == "trading":
            if validation_result.get("sharpe", 0) < 1.0:
                feedback["immediate_actions"].append(
                    "Improve risk-adjusted returns by optimizing strategy parameters"
                )
            if validation_result.get("max_drawdown", 1.0) > 0.10:
                feedback["immediate_actions"].append(
                    "Implement stricter stop-loss mechanisms to reduce drawdown"
                )

        # Add generic improvement suggestions
        feedback["improvement_suggestions"].extend(
            [
                "Review domain-specific requirements carefully",
                "Consider edge cases and error handling",
                "Validate outputs against expected formats",
                "Test with different input scenarios",
            ]
        )

        return feedback


__all__ = ["Validator"]
