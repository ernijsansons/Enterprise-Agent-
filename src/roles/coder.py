"""Coder role responsible for generation."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseRole


class Coder(BaseRole):
    def _build_prompt(self, plan_text: str, domain: str, guidelines: List[str]) -> str:
        """Build Claude-optimized coding prompt."""
        prompt_parts = [
            "You are a senior software engineer with expertise in clean, maintainable code.",
            "",
            "Your task is to implement the following plan with high-quality code.",
            "",
            "## Approach",
            "1. **Analyze** each step in the plan carefully",
            "2. **Design** the implementation approach",
            "3. **Implement** following best practices",
            "4. **Validate** the code for correctness and quality",
            "",
            f"## Domain Context: {domain}",
        ]

        if guidelines:
            prompt_parts.extend(
                [
                    "",
                    "## Guidelines",
                    "\n".join(f"- {item}" for item in guidelines),
                ]
            )

        prompt_parts.extend(
            [
                "",
                "## Implementation Plan",
                plan_text,
                "",
                "## Requirements",
                "- Write clean, readable, and maintainable code",
                "- Include appropriate error handling",
                "- Add meaningful comments for complex logic",
                "- Follow security best practices",
                "- Ensure code is production-ready",
                "",
                "## Output Format",
                "Provide your implementation with:",
                "1. Brief explanation of your approach",
                "2. The complete, working code",
                "3. Any important usage notes or considerations",
            ]
        )

        return "\n".join(prompt_parts)

    def generate(
        self, plan: str, domain: str, vuln_flag: bool = False
    ) -> Dict[str, Any]:
        plan_text = plan or "No plan available."
        pack = self.domain_pack(domain)
        guidelines = pack.get("generation_guidelines", [])
        prompt = self._build_prompt(plan_text, domain, guidelines)
        model = self.route_to_model(prompt, domain, vuln_flag)
        if model == "openai_gpt_5_codex":
            output = self.invoke_codex_cli("auto-edit", ["--prompt", prompt], domain)
            source = "codex_cli"
        else:
            output = self.call_model(model, prompt, "Coder", "generate")
            source = "model"
        self.store_memory("session", "coder_prompt", prompt)
        return {"output": output, "model": model, "source": source}


__all__ = ["Coder"]
