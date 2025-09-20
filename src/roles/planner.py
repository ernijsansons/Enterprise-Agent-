"""Planner role for decomposing tasks."""
from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseRole


class Planner(BaseRole):
    def decompose(self, task: str, domain: str) -> Dict[str, Any]:
        pack = self.domain_pack(domain)
        adapter = pack.get("prompt_adapter", domain)
        examples = pack.get("generation_guidelines", [])
        guidelines = "\n".join(f"- {item}" for item in examples) if examples else ""
        model = self.route_to_model(task, domain)

        # Claude-optimized prompt with structured reasoning
        prompt = self._build_decomposition_prompt(task, domain, adapter, guidelines)

        plan_text = self.call_model(model, prompt, "Planner", "decompose")
        epics = self._extract_actionable_epics(plan_text)
        self.store_memory("session", "plan", plan_text)
        return {"text": plan_text, "epics": epics, "model": model}

    def _build_decomposition_prompt(
        self, task: str, domain: str, adapter: str, guidelines: str
    ) -> str:
        """Build Claude-optimized decomposition prompt."""
        prompt_parts = [
            "You are an expert project planner with deep experience in software development.",
            "",
            "Your task is to decompose the following request into clear, actionable steps.",
            "Follow this structured approach:",
            "",
            "1. First, analyze the task to understand the core requirements",
            "2. Consider the domain context and any constraints",
            "3. Break down into logical, sequential steps",
            "4. Ensure each step is specific and actionable",
            "5. Consider dependencies and potential risks",
            "",
            f"Domain Context: {adapter}",
            "",
            f"Task to Decompose:\n{task}",
        ]

        if guidelines:
            prompt_parts.extend(
                [
                    "",
                    "Domain-Specific Guidelines:",
                    guidelines,
                ]
            )

        prompt_parts.extend(
            [
                "",
                "Please provide your decomposition in this format:",
                "",
                "## Analysis",
                "[Brief analysis of the task and key considerations]",
                "",
                "## Steps",
                "1. [First actionable step]",
                "2. [Second actionable step]",
                "3. [Continue with additional steps...]",
                "",
                "## Dependencies & Considerations",
                "[Any important dependencies or risk factors to consider]",
            ]
        )

        return "\n".join(prompt_parts)

    def _extract_actionable_epics(self, plan_text: str) -> List[str]:
        """Extract actionable epics from the plan text."""
        epics = []
        lines = plan_text.splitlines()
        in_steps_section = False

        for line in lines:
            line = line.strip()
            if line.lower().startswith("## steps"):
                in_steps_section = True
                continue
            elif line.startswith("## ") and in_steps_section:
                in_steps_section = False
                continue
            elif (
                in_steps_section
                and line
                and (line[0].isdigit() or line.startswith("-"))
            ):
                # Extract just the step text, removing numbering
                epic = (
                    line.split(".", 1)[-1].strip() if "." in line else line.strip("- ")
                )
                if epic:
                    epics.append(epic)

        # Fallback to original logic if no structured steps found
        if not epics:
            epics = [line.strip() for line in plan_text.splitlines() if line.strip()]

        return epics


__all__ = ["Planner"]
