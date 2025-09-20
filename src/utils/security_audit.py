"""Security audit logging for Enterprise Agent."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of security audit events."""

    AUTHENTICATION = "authentication"
    API_ACCESS = "api_access"
    CLI_USAGE = "cli_usage"
    CONFIG_CHANGE = "config_change"
    CREDENTIAL_ACCESS = "credential_access"
    PERMISSION_CHANGE = "permission_change"
    DATA_ACCESS = "data_access"
    SECURITY_VIOLATION = "security_violation"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Security audit event."""

    event_type: AuditEventType
    severity: AuditSeverity
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    source_ip: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "details": self.details,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "source_ip": self.source_ip,
            "event_hash": self._generate_hash(),
        }

    def _generate_hash(self) -> str:
        """Generate hash for event integrity."""
        event_str = f"{self.event_type.value}{self.timestamp}{self.description}"
        return hashlib.sha256(event_str.encode()).hexdigest()[:16]


class SecurityAuditor:
    """Security audit logging system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security auditor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.log_file = Path(
            self.config.get(
                "log_file", Path.home() / ".claude" / "security_audit.jsonl"
            )
        )
        self.max_log_size = (
            self.config.get("max_log_size_mb", 10) * 1024 * 1024
        )  # Convert to bytes
        self.retention_days = self.config.get("retention_days", 30)

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize audit log
        self._rotate_log_if_needed()

    def audit(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        description: str,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Log a security audit event.

        Args:
            event_type: Type of event
            severity: Severity level
            description: Event description
            details: Additional event details
            session_id: Session identifier
            user_id: User identifier
        """
        if not self.enabled:
            return

        try:
            event = AuditEvent(
                event_type=event_type,
                severity=severity,
                description=description,
                details=details or {},
                session_id=session_id,
                user_id=user_id,
            )

            # Scrub sensitive data
            event = self._scrub_sensitive_data(event)

            # Write to log file
            self._write_audit_event(event)

            # Log to standard logging for immediate visibility
            log_level = self._get_log_level(severity)
            logger.log(log_level, f"AUDIT: {description}", extra=event.to_dict())

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    def _scrub_sensitive_data(self, event: AuditEvent) -> AuditEvent:
        """Scrub sensitive data from audit event.

        Args:
            event: Audit event to scrub

        Returns:
            Scrubbed audit event
        """
        # List of sensitive keys to scrub
        sensitive_keys = {
            "api_key",
            "password",
            "token",
            "secret",
            "credential",
            "authorization",
            "auth",
            "key",
            "session_token",
        }

        def scrub_dict(data: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively scrub dictionary."""
            scrubbed: Dict[str, Any] = {}
            for key, value in data.items():
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in sensitive_keys):
                    scrubbed[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    scrubbed[key] = scrub_dict(value)
                elif isinstance(value, str) and len(value) > 20:
                    # Check if string looks like an API key or token
                    if any(
                        prefix in value
                        for prefix in ["sk-", "xoxb-", "ghp_", "bearer "]
                    ):
                        scrubbed[key] = "[REDACTED]"
                    else:
                        scrubbed[key] = value
                else:
                    scrubbed[key] = value
            return scrubbed

        # Scrub event details
        event.details = scrub_dict(event.details)

        # Scrub description if it contains sensitive patterns
        description = event.description
        for pattern in ["sk-", "token:", "key:", "password:"]:
            if pattern in description.lower():
                # Replace the sensitive part
                words = description.split()
                description = " ".join(
                    "[REDACTED]" if pattern in word.lower() else word for word in words
                )
        event.description = description

        return event

    def _write_audit_event(self, event: AuditEvent) -> None:
        """Write audit event to log file.

        Args:
            event: Audit event to write
        """
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

            # Rotate log if needed
            self._rotate_log_if_needed()

        except Exception as e:
            logger.error(f"Failed to write audit event to file: {e}")

    def _rotate_log_if_needed(self) -> None:
        """Rotate log file if it exceeds maximum size."""
        try:
            if (
                self.log_file.exists()
                and self.log_file.stat().st_size > self.max_log_size
            ):
                # Create backup with timestamp
                timestamp = int(time.time())
                backup_file = self.log_file.with_suffix(f".{timestamp}.jsonl")
                self.log_file.rename(backup_file)

                logger.info(f"Rotated audit log to {backup_file}")

                # Clean up old backups
                self._cleanup_old_logs()

        except Exception as e:
            logger.error(f"Failed to rotate audit log: {e}")

    def _cleanup_old_logs(self) -> None:
        """Clean up old audit log files."""
        try:
            cutoff_time = time.time() - (self.retention_days * 24 * 3600)
            log_dir = self.log_file.parent

            for log_file in log_dir.glob(f"{self.log_file.stem}.*.jsonl"):
                try:
                    # Extract timestamp from filename
                    timestamp_str = log_file.stem.split(".")[-1]
                    timestamp = int(timestamp_str)

                    if timestamp < cutoff_time:
                        log_file.unlink()
                        logger.debug(f"Deleted old audit log: {log_file}")

                except (ValueError, OSError) as e:
                    logger.warning(f"Failed to process old audit log {log_file}: {e}")

        except Exception as e:
            logger.error(f"Failed to cleanup old audit logs: {e}")

    def _get_log_level(self, severity: AuditSeverity) -> int:
        """Get logging level for audit severity.

        Args:
            severity: Audit severity

        Returns:
            Logging level
        """
        level_map = {
            AuditSeverity.LOW: logging.INFO,
            AuditSeverity.MEDIUM: logging.WARNING,
            AuditSeverity.HIGH: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }
        return level_map.get(severity, logging.INFO)

    def get_audit_events(
        self,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditSeverity] = None,
        since: Optional[float] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Get audit events with filtering.

        Args:
            event_type: Filter by event type
            severity: Filter by severity
            since: Filter events since timestamp
            limit: Maximum number of events to return

        Returns:
            List of audit events
        """
        events: List[Dict[str, Any]] = []

        try:
            if not self.log_file.exists():
                return events

            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())

                        # Apply filters
                        if event_type and event.get("event_type") != event_type.value:
                            continue
                        if severity and event.get("severity") != severity.value:
                            continue
                        if since and event.get("timestamp", 0) < since:
                            continue

                        events.append(event)

                        if len(events) >= limit:
                            break

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Failed to read audit events: {e}")

        return events[-limit:]  # Return most recent events

    def get_security_summary(self) -> Dict[str, Any]:
        """Get security audit summary.

        Returns:
            Security summary statistics
        """
        summary: Dict[str, Any] = {
            "total_events": 0,
            "events_by_type": {},
            "events_by_severity": {},
            "recent_critical_events": [],
            "audit_health": "healthy",
        }

        try:
            # Get events from last 24 hours
            since = time.time() - (24 * 3600)
            events = self.get_audit_events(since=since)

            summary["total_events"] = len(events)

            # Count by type and severity
            for event in events:
                event_type = event.get("event_type", "unknown")
                severity = event.get("severity", "unknown")

                summary["events_by_type"][event_type] = (
                    summary["events_by_type"].get(event_type, 0) + 1
                )
                summary["events_by_severity"][severity] = (
                    summary["events_by_severity"].get(severity, 0) + 1
                )

                # Collect critical events
                if severity == "critical":
                    summary["recent_critical_events"].append(event)

            # Determine audit health
            critical_count = summary["events_by_severity"].get("critical", 0)
            high_count = summary["events_by_severity"].get("high", 0)

            if critical_count > 0:
                summary["audit_health"] = "critical"
            elif high_count > 5:
                summary["audit_health"] = "warning"
            elif summary["total_events"] > 1000:
                summary["audit_health"] = "high_activity"

        except Exception as e:
            logger.error(f"Failed to generate security summary: {e}")
            summary["audit_health"] = "error"

        return summary


# Global security auditor instance
_security_auditor: Optional[SecurityAuditor] = None


def get_security_auditor(config: Optional[Dict[str, Any]] = None) -> SecurityAuditor:
    """Get or create global security auditor.

    Args:
        config: Optional configuration

    Returns:
        SecurityAuditor instance
    """
    global _security_auditor

    if _security_auditor is None:
        _security_auditor = SecurityAuditor(config)

    return _security_auditor


# Convenience functions for common audit events
def audit_authentication(
    action: str, success: bool, details: Optional[Dict[str, Any]] = None
) -> None:
    """Audit authentication event."""
    severity = AuditSeverity.MEDIUM if success else AuditSeverity.HIGH
    description = f"Authentication {action}: {'success' if success else 'failure'}"

    get_security_auditor().audit(
        AuditEventType.AUTHENTICATION, severity, description, details or {}
    )


def audit_api_access(
    endpoint: str, method: str, success: bool, details: Optional[Dict[str, Any]] = None
) -> None:
    """Audit API access event."""
    severity = AuditSeverity.LOW if success else AuditSeverity.MEDIUM
    description = (
        f"API access: {method} {endpoint} ({'success' if success else 'failure'})"
    )

    get_security_auditor().audit(
        AuditEventType.API_ACCESS, severity, description, details or {}
    )


def audit_cli_usage(
    command: str, success: bool, details: Optional[Dict[str, Any]] = None
) -> None:
    """Audit CLI usage event."""
    severity = AuditSeverity.LOW if success else AuditSeverity.MEDIUM
    description = f"CLI command: {command} ({'success' if success else 'failure'})"

    get_security_auditor().audit(
        AuditEventType.CLI_USAGE, severity, description, details or {}
    )


def audit_security_violation(
    violation_type: str, details: Optional[Dict[str, Any]] = None
) -> None:
    """Audit security violation."""
    description = f"Security violation detected: {violation_type}"

    get_security_auditor().audit(
        AuditEventType.SECURITY_VIOLATION,
        AuditSeverity.CRITICAL,
        description,
        details or {},
    )


def audit_credential_access(
    action: str, resource: str, details: Optional[Dict[str, Any]] = None
) -> None:
    """Audit credential access."""
    description = f"Credential access: {action} for {resource}"

    get_security_auditor().audit(
        AuditEventType.CREDENTIAL_ACCESS, AuditSeverity.HIGH, description, details or {}
    )


__all__ = [
    "AuditEventType",
    "AuditSeverity",
    "AuditEvent",
    "SecurityAuditor",
    "get_security_auditor",
    "audit_authentication",
    "audit_api_access",
    "audit_cli_usage",
    "audit_security_violation",
    "audit_credential_access",
]
