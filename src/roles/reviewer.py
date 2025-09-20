"""Reviewer role for scoring outputs."""
from __future__ import annotations

from typing import Dict, List

from .base import BaseRole


class Reviewer(BaseRole):
    def _build_review_prompt(self, output: str, domain: str, criteria: str) -> str:
        """Build Claude-optimized review prompt."""
        prompt_parts = [
            "You are an expert code reviewer with deep knowledge of software quality principles.",
            "",
            "Your task is to thoroughly evaluate the provided output according to the specified criteria.",
            "",
            "## Review Process",
            "1. **Analyze** the output systematically",
            "2. **Evaluate** against each criterion",
            "3. **Consider** potential issues and improvements",
            "4. **Assign** a fair, objective score",
            "",
            f"## Domain: {domain}",
            f"## Evaluation Criteria: {criteria}",
            "",
            "## Output to Review",
            "```",
            output,
            "```",
            "",
            "## Scoring Guidelines",
            "- **0.9-1.0**: Excellent - Exceeds expectations, minimal improvements needed",
            "- **0.7-0.8**: Good - Meets requirements with minor issues",
            "- **0.5-0.6**: Acceptable - Functional but needs improvements",
            "- **0.3-0.4**: Poor - Significant issues that need addressing",
            "- **0.0-0.2**: Unacceptable - Major problems, requires substantial rework",
            "",
            "## Response Format",
            "Provide your evaluation as JSON:",
            '{"score": <float between 0.0 and 1.0>, "rationale": "<detailed explanation of your score>"}',
            "",
            "In your rationale, explain:",
            "- What works well",
            "- What needs improvement",
            "- Specific recommendations for enhancement",
        ]

        return "\n".join(prompt_parts)

    def review(
        self, output: str, domain: str, vuln_flag: bool = False
    ) -> Dict[str, float | List[float] | List[str]]:
        models: List[str] = []
        primary = self.route_to_model(output, domain, vuln_flag)
        if primary:
            models.append(primary)
        if (
            getattr(self.orchestrator, "anthropic_client", None)
            and "claude_opus_4" not in models
        ):
            models.append("claude_opus_4")
        pack = self.domain_pack(domain)
        criteria = pack.get("review_criteria", "accuracy, safety, maintainability")

        scores: List[float] = []
        rationales: List[str] = []
        model_list = models or ([primary] if primary else [])
        for model in model_list:
            prompt = self._build_review_prompt(output, domain, criteria)
            try:
                result = self.call_model(model, prompt, "Reviewer", "score")
                parsed = self.parse_json(result)
                if isinstance(parsed, dict) and "score" in parsed:
                    score = float(parsed.get("score", 0.0))
                    rationale = str(parsed.get("rationale", "")).strip()
                else:
                    score = float(result.strip())
                    rationale = ""
            except Exception:
                score, rationale = 0.5, "fallback default"
            scores.append(max(0.0, min(1.0, score)))
            rationales.append(rationale)
        if not model_list:
            scores = [0.0]
            rationales = ["no reviewer models available"]
        confidence = sum(scores) / len(scores) if scores else 0.0
        return {
            "confidence": confidence,
            "scores": scores,
            "models": models,
            "rationales": rationales,
        }


__all__ = ["Reviewer"]
