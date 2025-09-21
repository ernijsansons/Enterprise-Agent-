"""Reflector role to iterate on failures."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from .base import BaseRole


class Reflector(BaseRole):
    def _build_reflection_prompt(
        self,
        validation: Dict[str, Any],
        current_output: str,
        domain: str,
        iterations: int,
    ) -> str:
        """Build Claude-optimized reflection prompt with enhanced failure analysis."""

        # Extract specific issues for targeted analysis
        issues_analysis = self._analyze_validation_issues(validation)

        prompt_parts = [
            "You are an expert problem-solver and process improvement specialist.",
            "",
            "Your task is to analyze validation feedback and create an improved solution.",
            "Focus on ACTIONABLE fixes that directly address the root causes of failures.",
            "",
            "## Reflection Process",
            "1. **Root Cause Analysis** - Identify why each issue occurred",
            "2. **Impact Assessment** - Determine which issues are most critical",
            "3. **Solution Design** - Create targeted fixes for each issue",
            "4. **Risk Evaluation** - Assess potential side effects of fixes",
            "5. **Implementation** - Apply the safest, most effective solution",
            "6. **Confidence Assessment** - Evaluate likelihood of success",
            "",
            f"## Domain: {domain}",
            f"## Iteration: {iterations + 1}/5 (Early termination encouraged if confident)",
            "",
            "## Issue Analysis",
            issues_analysis,
            "",
            "## Validation Feedback",
            "```json",
            json.dumps(validation, indent=2),
            "```",
            "",
            "## Current Output",
            "```",
            current_output[:3000] + ("...\n[TRUNCATED]" if len(current_output) > 3000 else ""),
            "```",
            "",
            "## Fix Requirements",
            "- Address ALL critical issues identified above",
            "- Provide specific, implementable solutions",
            "- Maintain existing functionality that works correctly",
            "- Minimize changes to reduce risk of new issues",
            "- Include reasoning for each change made",
            "",
            "## Response Format",
            "Provide comprehensive reflection as JSON:",
            '{',
            '  "root_cause_analysis": {',
            '    "primary_causes": ["most critical issues"],',
            '    "secondary_causes": ["minor issues"],',
            '    "failure_patterns": "recurring issues across iterations"',
            '  },',
            '  "fixes": [',
            '    {',
            '      "issue_type": "specific issue being addressed",',
            '      "description": "detailed fix description",',
            '      "implementation": "specific changes to make",',
            '      "risks": "potential negative effects",',
            '      "priority": "high|medium|low"',
            '    }',
            '  ],',
            '  "selected_fix": 0,',
            '  "fix_reasoning": "why this fix was chosen",',
            '  "revised_output": "complete improved version",',
            '  "validation_expectations": "what should pass validation now",',
            '  "confidence": 0.0,',
            '  "confidence_reasoning": "detailed explanation of confidence level"',
            '}',
            "",
            "## Confidence Scale",
            "- 0.9-1.0: Very confident - comprehensive fix, low risk",
            "- 0.7-0.8: Confident - good fix, manageable risk",
            "- 0.5-0.6: Moderate - partial fix, some uncertainty",
            "- 0.3-0.4: Low confidence - risky changes, uncertain outcome",
            "- 0.0-0.2: Very low - major issues persist, high failure risk",
        ]

        return "\n".join(prompt_parts)

    def reflect(
        self,
        validation: Dict[str, Any],
        current_output: str,
        domain: str,
        iterations: int,
        vuln_flag: bool = False,
    ) -> Dict[str, Any]:
        """Perform reflection with enhanced failure analysis and structured error handling.

        Args:
            validation: Validation results to analyze
            current_output: Current output to improve
            domain: Domain context
            iterations: Current iteration count
            vuln_flag: Whether vulnerability scanning is enabled

        Returns:
            Reflection results with actionable analysis

        Raises:
            EnterpriseAgentError: On reflection process failures
        """
        # Check iteration limits with better error context
        max_iterations = 5  # This should be configurable
        if iterations >= max_iterations:
            from src.utils.errors import handle_error, create_orchestration_error, ErrorCode

            error = create_orchestration_error(
                f"Reflection halted: maximum iterations ({max_iterations}) reached",
                error_code=ErrorCode.REFLECTION_MAX_ITERATIONS_REACHED,
                context={
                    "iterations": iterations,
                    "max_iterations": max_iterations,
                    "domain": domain,
                    "validation_status": validation.get("passes", False)
                }
            )
            handle_error(error)

            return {
                "output": current_output,
                "halt": True,
                "iterations": iterations,
                "confidence": 0.0,
                "halt_reason": "max_iterations_reached",
                "error_details": error.details.to_dict()
            }

        # Input validation
        if not isinstance(validation, dict):
            from src.utils.errors import create_validation_error, ErrorCode
            raise create_validation_error(
                f"Validation must be a dictionary, got {type(validation).__name__}",
                validation_type="validation_input",
                error_code=ErrorCode.INVALID_PARAMETERS,
                context={"validation_type": type(validation).__name__, "domain": domain}
            )

        try:
            model = self.route_to_model(json.dumps(validation), domain, vuln_flag)
            prompt = self._build_reflection_prompt(
                validation, current_output, domain, iterations
            )

            response = self.call_model(model, prompt, "Reflector", "reflect")

            # Enhanced parsing with validation
            try:
                parsed = self.parse_json(response)
            except Exception as e:
                from src.utils.errors import create_validation_error, ErrorCode
                # Log parsing failure but continue with fallback
                error = create_validation_error(
                    f"Failed to parse reflection response: {str(e)}",
                    validation_type="json_parse",
                    error_code=ErrorCode.VALIDATION_PARSE_ERROR,
                    context={"response_length": len(response), "domain": domain, "iterations": iterations},
                    cause=e
                )
                from src.utils.errors import handle_error
                handle_error(error)
                parsed = None

            # Process reflection results with enhanced validation
            if isinstance(parsed, dict) and parsed:
                result = self._process_structured_reflection(parsed, current_output, iterations, model)
            else:
                result = self._process_fallback_reflection(response, current_output, iterations, model)

            # Add reflection metadata
            result["reflection_metadata"] = {
                "timestamp": time.time(),
                "domain": domain,
                "model_used": model,
                "iterations": iterations,
                "has_structured_analysis": isinstance(parsed, dict) and parsed,
                "validation_issues_count": len(self._extract_validation_issues(validation))
            }

            return result

        except Exception as e:
            # Handle reflection failures with structured errors
            if hasattr(e, 'details'):
                raise  # Already a structured error
            else:
                from src.utils.errors import create_orchestration_error, ErrorCode
                raise create_orchestration_error(
                    f"Reflection process failed: {str(e)}",
                    error_code=ErrorCode.REFLECTION_LOOP_FAILED,
                    context={
                        "domain": domain,
                        "iterations": iterations,
                        "output_length": len(current_output),
                        "validation_passes": validation.get("passes", False)
                    },
                    cause=e
                )

    def _analyze_validation_issues(self, validation: Dict[str, Any]) -> str:
        """Analyze validation results to extract specific actionable issues."""
        issues = []

        # Extract domain-specific issues
        if not validation.get("passes", True):
            issues.append("❌ **CRITICAL**: Validation failed")

        # Coding domain issues
        if not validation.get("tests_passed", True):
            issues.append("🧪 **Tests failing** - Check pytest output for specific errors")

        coverage = validation.get("coverage", 1.0)
        threshold = validation.get("coverage_threshold", 0.97)
        if coverage < threshold:
            missing = (threshold - coverage) * 100
            issues.append(f"📊 **Coverage insufficient** - Need {missing:.1f}% more test coverage")

        # LLM validation issues
        llm_insights = validation.get("llm_insights", {})
        if isinstance(llm_insights, dict):
            llm_issues = llm_insights.get("issues", [])
            for issue in llm_issues[:3]:  # Limit to top 3 issues
                if isinstance(issue, dict):
                    issue_type = issue.get("type", "unknown")
                    description = issue.get("description", "")
                    fix = issue.get("fix", "")
                    issues.append(f"⚠️ **{issue_type.title()}**: {description}")
                    if fix:
                        issues.append(f"   💡 *Fix*: {fix}")

        # Actionable feedback
        actionable = validation.get("actionable_feedback", {})
        if isinstance(actionable, dict):
            immediate_actions = actionable.get("immediate_actions", [])
            for action in immediate_actions[:2]:  # Top 2 immediate actions
                issues.append(f"🚨 **Immediate Action**: {action}")

        return "\n".join(issues) if issues else "✅ No specific issues identified in validation"

    def _extract_validation_issues(self, validation: Dict[str, Any]) -> List[str]:
        """Extract list of validation issues for metadata."""
        issues = []

        if not validation.get("passes", True):
            issues.append("validation_failed")
        if not validation.get("tests_passed", True):
            issues.append("tests_failed")
        if validation.get("coverage", 1.0) < validation.get("coverage_threshold", 0.97):
            issues.append("insufficient_coverage")

        # Add LLM-identified issues
        llm_insights = validation.get("llm_insights", {})
        if isinstance(llm_insights, dict):
            llm_issues = llm_insights.get("issues", [])
            for issue in llm_issues:
                if isinstance(issue, dict) and "type" in issue:
                    issues.append(f"llm_{issue['type']}")

        return issues

    def _process_structured_reflection(
        self, parsed: Dict[str, Any], current_output: str, iterations: int, model: str
    ) -> Dict[str, Any]:
        """Process structured reflection response with validation."""
        # Validate required fields
        required_fields = ["confidence", "revised_output"]
        missing_fields = [f for f in required_fields if f not in parsed]
        if missing_fields:
            from src.utils.errors import handle_error, create_validation_error, ErrorCode
            error = create_validation_error(
                f"Missing required reflection fields: {missing_fields}",
                validation_type="reflection_structure",
                error_code=ErrorCode.VALIDATION_FAILED,
                context={"missing_fields": missing_fields, "available_fields": list(parsed.keys())}
            )
            handle_error(error)
            # Set defaults for missing fields
            if "confidence" not in parsed:
                parsed["confidence"] = 0.0
            if "revised_output" not in parsed:
                parsed["revised_output"] = current_output

        confidence = float(parsed.get("confidence", 0.0))
        revised = parsed.get("revised_output", current_output)
        analysis = parsed

        # Validate confidence range
        if not (0.0 <= confidence <= 1.0):
            confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
            analysis["confidence_adjusted"] = True

        # Enhanced halt decision logic
        halt_threshold = 0.8
        should_halt = confidence >= halt_threshold

        # Additional halt conditions
        fixes = parsed.get("fixes", [])
        if len(fixes) == 0 and confidence > 0.6:
            should_halt = True  # No fixes needed and decent confidence
            analysis["halt_reason"] = "no_fixes_needed"
        elif confidence >= 0.9:
            should_halt = True  # Very high confidence
            analysis["halt_reason"] = "high_confidence"

        return {
            "output": revised,
            "halt": should_halt,
            "iterations": iterations + 1,
            "model": model,
            "confidence": confidence,
            "analysis": analysis,
            "structured_reflection": True
        }

    def _process_fallback_reflection(
        self, response: str, current_output: str, iterations: int, model: str
    ) -> Dict[str, Any]:
        """Process fallback reflection when structured parsing fails."""
        # Extract what we can from the raw response
        confidence = 0.0
        revised = response.strip() or current_output

        # Simple heuristics for fallback confidence
        if "fix" in response.lower() and "improve" in response.lower():
            confidence = 0.4  # Some improvement attempted
        elif "error" in response.lower() or "problem" in response.lower():
            confidence = 0.2  # Issues identified but unclear fixes
        elif len(response.strip()) > 100:
            confidence = 0.3  # Substantial response suggests some analysis

        analysis = {
            "raw": response,
            "parsing_failed": True,
            "fallback_processing": True,
            "confidence_heuristic": True
        }

        return {
            "output": revised,
            "halt": False,  # Conservative - don't halt on parsing failures
            "iterations": iterations + 1,
            "model": model,
            "confidence": confidence,
            "analysis": analysis,
            "structured_reflection": False
        }


__all__ = ["Reflector"]
