"""Cost estimation utilities for agent orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostEvent:
    role: str
    operation: str
    tokens: int
    model: str
    cost: float
    extra: Dict[str, Any] = field(default_factory=dict)


class CostEstimator:
    """Track token usage and enforce per-run budgets."""

    DEFAULT_MODEL = "openai_gpt_5_codex"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.rates: Dict[str, Dict[str, float]] = {
            "openai_gpt_5_codex": {"input": 1.25 / 1_000_000, "output": 10 / 1_000_000},
            "claude_opus_4": {"input": 15 / 1_000_000, "output": 75 / 1_000_000},
            "claude_sonnet_4": {"input": 3 / 1_000_000, "output": 15 / 1_000_000},
            "gemini-2.5-pro": {"input": 0.5 / 1_000_000, "output": 5 / 1_000_000},
        }
        self.budget = float(config.get("budget_per_run", 0.40))
        self.events: List[CostEvent] = []

    # --------------------------------------------------------------------- helpers
    def _model_rates(self, model: Optional[str]) -> Dict[str, float]:
        model_name = model or self.DEFAULT_MODEL
        return self.rates.get(
            model_name, {"input": 0.001 / 1_000_000, "output": 0.001 / 1_000_000}
        )

    def _record(self, event: CostEvent) -> None:
        self.events.append(event)
        if self.total_cost > self.budget:
            logger.warning(
                "Cost budget exceeded: %.4f > %.4f", self.total_cost, self.budget
            )
            raise ValueError("Cost budget exceeded")

    # --------------------------------------------------------------------- public
    @property
    def total_cost(self) -> float:
        return sum(event.cost for event in self.events)

    @property
    def total_tokens(self) -> int:
        return sum(event.tokens for event in self.events)

    def track(
        self,
        tokens: int,
        role: str,
        operation: str,
        model: Optional[str] = None,
        direction: str = "total",
    ) -> float:
        """Track a model call. Direction can be 'input', 'output', or 'total'."""

        rates = self._model_rates(model)
        model_name = model or self.DEFAULT_MODEL
        direction = direction.lower()
        if direction not in {"input", "output", "total"}:
            raise ValueError("direction must be 'input', 'output', or 'total'")

        if direction == "total":
            # Assume 25% input / 75% output split when total tokens supplied.
            input_tokens = int(tokens * 0.25)
            output_tokens = tokens - input_tokens
            cost = input_tokens * rates.get("input", 0) + output_tokens * rates.get(
                "output", 0
            )
        else:
            rate = rates.get(direction, 0)
            cost = tokens * rate

        event = CostEvent(
            role=role, operation=operation, tokens=tokens, model=model_name, cost=cost
        )
        self._record(event)
        return cost

    def track_estimated(
        self, tokens: int, role: str, operation: str, model: Optional[str] = None
    ) -> float:
        """Track an estimated cost for non-model invocations (e.g., CLI)."""
        return self.track(tokens, role, operation, model=model, direction="total")

    def estimate(self, complexity: int, _domain: str) -> float:
        # Lightweight heuristic: 1k characters ~ 250 tokens; multiply by unit cost.
        approx_tokens = max(complexity, 1) // 4
        rates = self._model_rates(self.DEFAULT_MODEL)
        return approx_tokens * (rates["input"] * 0.25 + rates["output"] * 0.75)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_cost": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "calls": [event.__dict__ for event in self.events],
            "budget": self.budget,
        }


__all__ = ["CostEstimator", "CostEvent"]
