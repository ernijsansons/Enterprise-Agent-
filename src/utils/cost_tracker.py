"""Cost tracking utilities for Enterprise Agent."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Represents a single cost entry."""

    timestamp: float
    provider: str
    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    cost: float
    session_id: Optional[str] = None
    domain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class CostTracker:
    """Tracks costs across different providers and models."""

    def __init__(self, log_file: Optional[Path] = None):
        """Initialize cost tracker.

        Args:
            log_file: Optional path to cost log file
        """
        self.log_file = log_file or Path("logs/cost_tracking.json")
        self.entries: List[CostEntry] = []
        self.total_cost = 0.0
        self.provider_costs: Dict[str, float] = {}
        self.model_costs: Dict[str, float] = {}

    def track(
        self,
        provider: str,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        session_id: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> None:
        """Track a cost entry.

        Args:
            provider: Provider name (e.g., 'anthropic', 'openai')
            model: Model name (e.g., 'claude-3-5-sonnet-20241022')
            operation: Operation type (e.g., 'planning', 'coding')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            session_id: Optional session ID
            domain: Optional domain
        """
        entry = CostEntry(
            timestamp=time.time(),
            provider=provider,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            session_id=session_id,
            domain=domain,
        )

        self.entries.append(entry)
        self.total_cost += cost

        # Update provider costs
        if provider not in self.provider_costs:
            self.provider_costs[provider] = 0.0
        self.provider_costs[provider] += cost

        # Update model costs
        model_key = f"{provider}:{model}"
        if model_key not in self.model_costs:
            self.model_costs[model_key] = 0.0
        self.model_costs[model_key] += cost

        logger.debug(f"Tracked cost: {cost:.4f} USD for {provider}:{model}")

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary.

        Returns:
            Dictionary with cost summary
        """
        return {
            "total_cost": self.total_cost,
            "total_entries": len(self.entries),
            "provider_costs": self.provider_costs,
            "model_costs": self.model_costs,
            "average_cost_per_entry": self.total_cost / len(self.entries)
            if self.entries
            else 0.0,
        }

    def get_entries_by_provider(self, provider: str) -> List[CostEntry]:
        """Get entries for a specific provider.

        Args:
            provider: Provider name

        Returns:
            List of cost entries
        """
        return [entry for entry in self.entries if entry.provider == provider]

    def get_entries_by_session(self, session_id: str) -> List[CostEntry]:
        """Get entries for a specific session.

        Args:
            session_id: Session ID

        Returns:
            List of cost entries
        """
        return [entry for entry in self.entries if entry.session_id == session_id]

    def save_to_file(self) -> None:
        """Save entries to log file."""
        try:
            # Ensure directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert entries to dictionaries
            data = {
                "entries": [entry.to_dict() for entry in self.entries],
                "summary": self.get_summary(),
                "last_updated": time.time(),
            }

            # Write to file
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.entries)} cost entries to {self.log_file}")

        except Exception as e:
            logger.error(f"Failed to save cost tracking data: {e}")

    def load_from_file(self) -> None:
        """Load entries from log file."""
        try:
            if not self.log_file.exists():
                logger.info(f"Cost tracking file {self.log_file} does not exist")
                return

            with open(self.log_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load entries
            self.entries = []
            for entry_data in data.get("entries", []):
                entry = CostEntry(**entry_data)
                self.entries.append(entry)

            # Recalculate totals
            self.total_cost = sum(entry.cost for entry in self.entries)
            self.provider_costs = {}
            self.model_costs = {}

            for entry in self.entries:
                # Update provider costs
                if entry.provider not in self.provider_costs:
                    self.provider_costs[entry.provider] = 0.0
                self.provider_costs[entry.provider] += entry.cost

                # Update model costs
                model_key = f"{entry.provider}:{entry.model}"
                if model_key not in self.model_costs:
                    self.model_costs[model_key] = 0.0
                self.model_costs[model_key] += entry.cost

            logger.info(f"Loaded {len(self.entries)} cost entries from {self.log_file}")

        except Exception as e:
            logger.error(f"Failed to load cost tracking data: {e}")

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()
        self.total_cost = 0.0
        self.provider_costs.clear()
        self.model_costs.clear()
        logger.info("Cleared all cost tracking data")


# Global cost tracker instance
_global_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance.

    Returns:
        CostTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
        _global_tracker.load_from_file()
    return _global_tracker


def track_cost(
    provider: str,
    model: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    session_id: Optional[str] = None,
    domain: Optional[str] = None,
) -> None:
    """Track a cost entry using the global tracker.

    Args:
        provider: Provider name
        model: Model name
        operation: Operation type
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost: Cost in USD
        session_id: Optional session ID
        domain: Optional domain
    """
    tracker = get_cost_tracker()
    tracker.track(
        provider=provider,
        model=model,
        operation=operation,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        session_id=session_id,
        domain=domain,
    )


def get_cost_summary() -> Dict[str, Any]:
    """Get cost summary from global tracker.

    Returns:
        Dictionary with cost summary
    """
    tracker = get_cost_tracker()
    return tracker.get_summary()


def save_cost_data() -> None:
    """Save cost data to file."""
    tracker = get_cost_tracker()
    tracker.save_to_file()


__all__ = [
    "CostEntry",
    "CostTracker",
    "get_cost_tracker",
    "track_cost",
    "get_cost_summary",
    "save_cost_data",
]
