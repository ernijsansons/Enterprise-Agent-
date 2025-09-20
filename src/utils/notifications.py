"""User notification system for Enterprise Agent events and errors."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationType(Enum):
    """Types of notifications."""

    AUTHENTICATION = "authentication"
    CLI_FAILURE = "cli_failure"
    API_FALLBACK = "api_fallback"
    USAGE_WARNING = "usage_warning"
    CONFIGURATION = "configuration"
    SESSION = "session"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class Notification:
    """Single notification with metadata."""

    type: NotificationType
    level: NotificationLevel
    title: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    action_required: bool = False
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary."""
        return {
            "type": self.type.value,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "details": self.details or {},
            "timestamp": self.timestamp,
            "action_required": self.action_required,
            "recommendations": self.recommendations,
        }


class NotificationManager:
    """Manages user notifications and feedback."""

    def __init__(self):
        """Initialize notification manager."""
        self.notifications: List[Notification] = []
        self.handlers: List[Callable[[Notification], None]] = []
        self.max_notifications = 100

        # Add default console handler
        self.add_handler(self._console_handler)

    def add_handler(self, handler: Callable[[Notification], None]) -> None:
        """Add a notification handler.

        Args:
            handler: Function that processes notifications
        """
        self.handlers.append(handler)

    def notify(
        self,
        type: NotificationType,
        level: NotificationLevel,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        action_required: bool = False,
        recommendations: Optional[List[str]] = None,
    ) -> None:
        """Send a notification.

        Args:
            type: Type of notification
            level: Severity level
            title: Short title
            message: Detailed message
            details: Additional details
            action_required: Whether user action is needed
            recommendations: List of recommended actions
        """
        notification = Notification(
            type=type,
            level=level,
            title=title,
            message=message,
            details=details,
            action_required=action_required,
            recommendations=recommendations or [],
        )

        # Store notification
        self.notifications.append(notification)

        # Trim old notifications
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[-self.max_notifications :]

        # Send to handlers
        for handler in self.handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")

    def _console_handler(self, notification: Notification) -> None:
        """Default console handler for notifications."""
        level_symbols = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌",
            NotificationLevel.CRITICAL: "🚨",
        }

        symbol = level_symbols.get(notification.level, "•")

        # Format message
        output = f"{symbol} {notification.title}"
        if notification.message:
            output += f"\n   {notification.message}"

        if notification.action_required:
            output += "\n   🔧 Action Required"

        if notification.recommendations:
            output += "\n   Recommendations:"
            for rec in notification.recommendations:
                output += f"\n   • {rec}"

        # Log with appropriate level
        if notification.level == NotificationLevel.CRITICAL:
            logger.critical(output)
        elif notification.level == NotificationLevel.ERROR:
            logger.error(output)
        elif notification.level == NotificationLevel.WARNING:
            logger.warning(output)
        else:
            logger.info(output)

    def get_notifications(
        self,
        type: Optional[NotificationType] = None,
        level: Optional[NotificationLevel] = None,
        since: Optional[float] = None,
    ) -> List[Notification]:
        """Get notifications with optional filtering.

        Args:
            type: Filter by notification type
            level: Filter by severity level
            since: Filter by timestamp (Unix time)

        Returns:
            Filtered list of notifications
        """
        filtered = self.notifications

        if type:
            filtered = [n for n in filtered if n.type == type]

        if level:
            filtered = [n for n in filtered if n.level == level]

        if since:
            filtered = [n for n in filtered if n.timestamp >= since]

        return filtered

    def clear_notifications(
        self,
        type: Optional[NotificationType] = None,
        older_than: Optional[float] = None,
    ) -> int:
        """Clear notifications with optional filtering.

        Args:
            type: Clear only this type of notification
            older_than: Clear notifications older than this timestamp

        Returns:
            Number of notifications cleared
        """
        initial_count = len(self.notifications)

        if type:
            self.notifications = [n for n in self.notifications if n.type != type]

        if older_than:
            self.notifications = [
                n for n in self.notifications if n.timestamp >= older_than
            ]

        if not type and not older_than:
            self.notifications.clear()

        return initial_count - len(self.notifications)

    def notify_cli_failure(
        self, operation: str, error: str, fallback_used: bool = False
    ) -> None:
        """Notify about Claude Code CLI failure.

        Args:
            operation: The operation that failed
            error: Error message
            fallback_used: Whether API fallback was used
        """
        recommendations = [
            "Check 'claude login' status",
            "Verify Claude Code CLI is installed and updated",
            "Check network connectivity",
        ]

        if fallback_used:
            recommendations.append(
                "⚠️  Using API fallback - this will incur additional charges!"
            )

        level = NotificationLevel.WARNING if fallback_used else NotificationLevel.ERROR

        self.notify(
            type=NotificationType.CLI_FAILURE,
            level=level,
            title=f"Claude Code CLI Failure: {operation}",
            message=f"CLI operation failed: {error}",
            details={
                "operation": operation,
                "error": error,
                "fallback_used": fallback_used,
            },
            action_required=not fallback_used,
            recommendations=recommendations,
        )

    def notify_authentication_issue(
        self, issue_type: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Notify about authentication issues.

        Args:
            issue_type: Type of authentication issue
            details: Additional details
        """
        recommendations = []

        if issue_type == "not_logged_in":
            recommendations = [
                "Run 'claude login' to authenticate with your Max subscription",
                "Ensure you have an active Anthropic Max plan",
            ]
        elif issue_type == "api_key_conflict":
            recommendations = [
                "Remove ANTHROPIC_API_KEY from environment variables",
                "Comment out ANTHROPIC_API_KEY in .env file",
                "Use Claude Code subscription instead of API",
            ]
        elif issue_type == "cli_not_found":
            recommendations = [
                "Install Claude Code CLI: npm install -g @anthropic-ai/claude-code",
                "Verify Node.js is installed and updated",
            ]

        self.notify(
            type=NotificationType.AUTHENTICATION,
            level=NotificationLevel.ERROR,
            title="Authentication Issue",
            message=f"Authentication problem detected: {issue_type}",
            details=details,
            action_required=True,
            recommendations=recommendations,
        )

    def notify_usage_warning(
        self, current_usage: int, limit: int, window_remaining: float
    ) -> None:
        """Notify about usage approaching limits.

        Args:
            current_usage: Current prompt count
            limit: Usage limit
            window_remaining: Hours remaining in window
        """
        percentage = (current_usage / limit) * 100

        recommendations = []
        if percentage >= 90:
            recommendations = [
                "Consider pausing non-critical operations",
                "Wait for usage window to reset",
                "Optimize prompts to be more efficient",
            ]
        elif percentage >= 80:
            recommendations = [
                "Monitor usage more closely",
                "Batch similar operations together",
            ]

        level = (
            NotificationLevel.CRITICAL
            if percentage >= 90
            else NotificationLevel.WARNING
        )

        self.notify(
            type=NotificationType.USAGE_WARNING,
            level=level,
            title="Usage Limit Warning",
            message=f"Usage at {percentage:.1f}% of limit ({current_usage}/{limit})",
            details={
                "current_usage": current_usage,
                "limit": limit,
                "percentage": percentage,
                "window_remaining_hours": window_remaining,
            },
            action_required=percentage >= 90,
            recommendations=recommendations,
        )

    def notify_configuration_issue(
        self, issue: str, recommendations: List[str]
    ) -> None:
        """Notify about configuration issues.

        Args:
            issue: Description of the issue
            recommendations: List of recommended fixes
        """
        self.notify(
            type=NotificationType.CONFIGURATION,
            level=NotificationLevel.WARNING,
            title="Configuration Issue",
            message=issue,
            action_required=True,
            recommendations=recommendations,
        )


# Global notification manager instance
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get or create the global notification manager.

    Returns:
        NotificationManager instance
    """
    global _notification_manager

    if _notification_manager is None:
        _notification_manager = NotificationManager()

    return _notification_manager


# Convenience functions
def notify_cli_failure(operation: str, error: str, fallback_used: bool = False) -> None:
    """Convenience function to notify about CLI failures."""
    get_notification_manager().notify_cli_failure(operation, error, fallback_used)


def notify_authentication_issue(
    issue_type: str, details: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to notify about authentication issues."""
    get_notification_manager().notify_authentication_issue(issue_type, details)


def notify_usage_warning(
    current_usage: int, limit: int, window_remaining: float
) -> None:
    """Convenience function to notify about usage warnings."""
    get_notification_manager().notify_usage_warning(
        current_usage, limit, window_remaining
    )


def notify_configuration_issue(issue: str, recommendations: List[str]) -> None:
    """Convenience function to notify about configuration issues."""
    get_notification_manager().notify_configuration_issue(issue, recommendations)


def get_notifications() -> List[Notification]:
    """Get all notifications for testing."""
    return get_notification_manager().notifications.copy()


def clear_notifications() -> None:
    """Clear all notifications for testing."""
    get_notification_manager().notifications.clear()


def reset_notification_manager() -> None:
    """Reset the global notification manager instance for testing."""
    global _notification_manager
    _notification_manager = None


__all__ = [
    "NotificationLevel",
    "NotificationType",
    "Notification",
    "NotificationManager",
    "get_notification_manager",
    "notify_cli_failure",
    "notify_authentication_issue",
    "notify_usage_warning",
    "notify_configuration_issue",
    "get_notifications",
    "clear_notifications",
    "reset_notification_manager",
]
