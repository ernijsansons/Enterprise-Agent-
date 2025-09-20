"""Reflector role to iterate on failures."""
from __future__ import annotations

import json
from typing import Any, Dict

from .base import BaseRole


class Reflector(BaseRole):
    def _build_reflection_prompt(
        self, validation: Dict[str, Any], current_output: str, domain: str, iterations: int
    ) -> str:
        """Build Claude-optimized reflection prompt."""
        prompt_parts = [
            "You are an expert problem-solver and process improvement specialist.",
            "",
            "Your task is to analyze validation feedback and create an improved solution.",
            "",
            "## Reflection Process",
            "1. **Understand** the feedback and identify root causes",
            "2. **Analyze** what went wrong and why",
            "3. **Design** targeted improvements",
            "4. **Implement** the revised solution",
            "5. **Assess** confidence in the improvement",
            "",
            f"## Domain: {domain}",
            f"## Iteration: {iterations + 1}/5",
            "",
            "## Validation Feedback",
            "```json",
            json.dumps(validation, indent=2),
            "```",
            "",
            "## Current Output",
            "```",
            current_output,
            "```",
            "",
            "## Requirements",
            "- Focus on specific issues identified in validation",
            "- Provide concrete, actionable improvements",
            "- Maintain the core functionality while fixing problems",
            "- Be conservative with changes to avoid introducing new issues",
            "",
            "## Response Format",
            "Provide your reflection as JSON:",
            '{"analysis": "detailed analysis of issues", '
            '"fixes": [{"description": "fix description", "risks": "potential risks"}], '
            '"selected_fix": 0, '
            '"revised_output": "improved version", '
            '"confidence": 0.0}',
            "",
            "Confidence scale:",
            "- 0.9-1.0: Very confident in the fix",
            "- 0.7-0.8: Confident, minor concerns remain",
            "- 0.5-0.6: Moderate confidence, some uncertainty",
            "- 0.3-0.4: Low confidence, significant concerns",
            "- 0.0-0.2: Very low confidence, major issues remain",
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
        if iterations >= 5:
            return {
                "output": current_output,
                "halt": True,
                "iterations": iterations,
                "confidence": 0.0,
            }

        model = self.route_to_model(json.dumps(validation), domain, vuln_flag)
        prompt = self._build_reflection_prompt(validation, current_output, domain, iterations)
        response = self.call_model(model, prompt, "Reflector", "reflect")
        try:
            parsed = self.parse_json(response)
        except Exception:  # pragma: no cover - stub orchestrators may raise
            parsed = None
        if isinstance(parsed, dict) and parsed:
            confidence = float(parsed.get("confidence", 0.0))
            revised = parsed.get("revised_output", current_output)
            analysis = parsed
        else:
            confidence = 0.0
            revised = response.strip() or current_output
            analysis = {"raw": response}
        halt = confidence >= 0.8
        return {
            "output": revised,
            "halt": halt,
            "iterations": iterations + 1,
            "model": model,
            "confidence": confidence,
            "analysis": analysis,
        }


__all__ = ["Reflector"]
