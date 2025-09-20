"""Usage monitoring system for Claude Max subscription limits."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.notifications import notify_usage_warning

logger = logging.getLogger(__name__)


@dataclass
class UsageWindow:
    """Represents a usage window with time boundaries."""

    start_time: float
    end_time: float
    prompt_count: int = 0
    requests: List[Dict[str, Any]] = field(default_factory=list)

    def is_active(self, current_time: float) -> bool:
        """Check if this window is currently active."""
        return self.start_time <= current_time <= self.end_time

    def add_request(self, role: str, operation: str, tokens: int = 0) -> None:
        """Add a request to this window."""
        self.prompt_count += 1
        self.requests.append({
            "timestamp": time.time(),
            "role": role,
            "operation": operation,
            "tokens": tokens
        })

    def time_remaining(self, current_time: float) -> float:
        """Get hours remaining in this window."""
        if current_time > self.end_time:
            return 0.0
        return (self.end_time - current_time) / 3600


class UsageMonitor:
    """Monitors usage against Claude Max subscription limits."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize usage monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Configuration
        self.max_prompts_per_window = self.config.get("max_prompts_per_window", 800)
        self.window_hours = self.config.get("window_hours", 5)
        self.warning_threshold = self.config.get("warning_threshold", 80)  # Percentage
        self.auto_pause_at_limit = self.config.get("auto_pause_at_limit", True)

        # State
        self.usage_file = Path.home() / ".claude" / "usage_history.json"
        self.windows: List[UsageWindow] = []
        self.paused = False
        self.pause_until = 0.0

        self._load_usage_history()

    def _load_usage_history(self) -> None:
        """Load usage history from disk."""
        try:
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)

                # Reconstruct windows from saved data
                for window_data in data.get("windows", []):
                    window = UsageWindow(
                        start_time=window_data["start_time"],
                        end_time=window_data["end_time"],
                        prompt_count=window_data.get("prompt_count", 0),
                        requests=window_data.get("requests", [])
                    )
                    self.windows.append(window)

                self.paused = data.get("paused", False)
                self.pause_until = data.get("pause_until", 0.0)

                # Clean up old windows
                self._cleanup_old_windows()

                logger.debug(f"Loaded {len(self.windows)} usage windows")

        except Exception as e:
            logger.warning(f"Failed to load usage history: {e}")
            self.windows = []

    def _save_usage_history(self) -> None:
        """Save usage history to disk."""
        try:
            # Ensure directory exists
            self.usage_file.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization
            data = {
                "windows": [
                    {
                        "start_time": window.start_time,
                        "end_time": window.end_time,
                        "prompt_count": window.prompt_count,
                        "requests": window.requests
                    }
                    for window in self.windows
                ],
                "paused": self.paused,
                "pause_until": self.pause_until,
                "last_updated": time.time()
            }

            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug("Saved usage history to disk")

        except Exception as e:
            logger.warning(f"Failed to save usage history: {e}")

    def _cleanup_old_windows(self) -> None:
        """Remove windows older than 24 hours."""
        current_time = time.time()
        cutoff_time = current_time - (24 * 3600)  # 24 hours ago

        original_count = len(self.windows)
        self.windows = [w for w in self.windows if w.end_time > cutoff_time]

        if len(self.windows) < original_count:
            logger.debug(f"Cleaned up {original_count - len(self.windows)} old usage windows")

    def _get_current_window(self) -> UsageWindow:
        """Get or create the current usage window."""
        current_time = time.time()

        # Check if we have an active window
        for window in self.windows:
            if window.is_active(current_time):
                return window

        # Create new window
        window_duration = self.window_hours * 3600
        new_window = UsageWindow(
            start_time=current_time,
            end_time=current_time + window_duration
        )

        self.windows.append(new_window)
        logger.debug(f"Created new usage window: {new_window.start_time} - {new_window.end_time}")

        return new_window

    def check_pause_status(self) -> bool:
        """Check if usage is currently paused."""
        current_time = time.time()

        if self.paused and current_time < self.pause_until:
            return True
        elif self.paused and current_time >= self.pause_until:
            # Unpause automatically
            self.paused = False
            self.pause_until = 0.0
            self._save_usage_history()
            logger.info("Usage monitoring unpaused automatically")

        return False

    def can_make_request(self) -> bool:
        """Check if a new request can be made within limits."""
        if self.check_pause_status():
            return False

        current_window = self._get_current_window()
        return current_window.prompt_count < self.max_prompts_per_window

    def record_request(self, role: str, operation: str, tokens: int = 0) -> bool:
        """Record a request and check limits.

        Args:
            role: Role making the request
            operation: Operation being performed
            tokens: Number of tokens (if available)

        Returns:
            True if request was recorded, False if blocked by limits
        """
        if self.check_pause_status():
            logger.warning("Request blocked: usage monitoring paused")
            return False

        current_window = self._get_current_window()

        # Check if we're at the limit
        if current_window.prompt_count >= self.max_prompts_per_window:
            if self.auto_pause_at_limit:
                # Pause until next window
                window_remaining = current_window.time_remaining(time.time())
                self.paused = True
                self.pause_until = current_window.end_time

                notify_usage_warning(
                    current_window.prompt_count,
                    self.max_prompts_per_window,
                    window_remaining
                )

                logger.warning(f"Usage limit reached. Paused until {self.pause_until}")
                self._save_usage_history()
                return False
            else:
                logger.warning("Usage limit reached but auto-pause disabled")
                return False

        # Record the request
        current_window.add_request(role, operation, tokens)

        # Check warning threshold
        usage_percentage = (current_window.prompt_count / self.max_prompts_per_window) * 100

        if usage_percentage >= self.warning_threshold:
            window_remaining = current_window.time_remaining(time.time())
            notify_usage_warning(
                current_window.prompt_count,
                self.max_prompts_per_window,
                window_remaining
            )

        # Save updated usage
        self._save_usage_history()

        logger.debug(f"Recorded request: {role}/{operation} ({current_window.prompt_count}/{self.max_prompts_per_window})")

        return True

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        current_time = time.time()
        current_window = self._get_current_window()

        usage_percentage = (current_window.prompt_count / self.max_prompts_per_window) * 100
        window_remaining = current_window.time_remaining(current_time)

        stats = {
            "current_usage": current_window.prompt_count,
            "limit": self.max_prompts_per_window,
            "usage_percentage": usage_percentage,
            "window_remaining_hours": window_remaining,
            "paused": self.paused,
            "can_make_request": self.can_make_request(),
            "window_start": current_window.start_time,
            "window_end": current_window.end_time
        }

        # Recent usage patterns
        if len(self.windows) > 1:
            recent_windows = self.windows[-3:]  # Last 3 windows
            stats["recent_usage"] = [
                {
                    "start": w.start_time,
                    "end": w.end_time,
                    "prompts": w.prompt_count,
                    "utilization": (w.prompt_count / self.max_prompts_per_window) * 100
                }
                for w in recent_windows
            ]

        return stats

    def reset_usage(self) -> None:
        """Reset usage tracking (for testing or manual reset)."""
        self.windows = []
        self.paused = False
        self.pause_until = 0.0
        self._save_usage_history()
        logger.info("Usage tracking reset")

    def force_pause(self, hours: float = 1.0) -> None:
        """Manually pause usage monitoring.

        Args:
            hours: How many hours to pause for
        """
        self.paused = True
        self.pause_until = time.time() + (hours * 3600)
        self._save_usage_history()
        logger.info(f"Usage monitoring manually paused for {hours} hours")

    def unpause(self) -> None:
        """Manually unpause usage monitoring."""
        self.paused = False
        self.pause_until = 0.0
        self._save_usage_history()
        logger.info("Usage monitoring manually unpaused")


# Global usage monitor instance
_usage_monitor: Optional[UsageMonitor] = None


def get_usage_monitor(config: Optional[Dict[str, Any]] = None) -> UsageMonitor:
    """Get or create the global usage monitor.

    Args:
        config: Optional configuration

    Returns:
        UsageMonitor instance
    """
    global _usage_monitor

    if _usage_monitor is None:
        _usage_monitor = UsageMonitor(config)

    return _usage_monitor


def record_claude_usage(role: str, operation: str, tokens: int = 0) -> bool:
    """Convenience function to record Claude usage.

    Args:
        role: Role making the request
        operation: Operation being performed
        tokens: Number of tokens

    Returns:
        True if request was allowed, False if blocked
    """
    return get_usage_monitor().record_request(role, operation, tokens)


def can_make_claude_request() -> bool:
    """Check if a Claude request can be made."""
    return get_usage_monitor().can_make_request()


def get_claude_usage_stats() -> Dict[str, Any]:
    """Get current Claude usage statistics."""
    return get_usage_monitor().get_usage_stats()


__all__ = [
    "UsageWindow",
    "UsageMonitor",
    "get_usage_monitor",
    "record_claude_usage",
    "can_make_claude_request",
    "get_claude_usage_stats"
]