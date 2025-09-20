"""Custom exception types for the Enterprise Agent."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


class AgentException(Exception):
    """Base exception for all agent-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        parts = [self.message]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)


class ModelException(AgentException):
    """Exception raised when model interaction fails."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.model = model
        self.provider = provider
        if model:
            self.details["model"] = model
        if provider:
            self.details["provider"] = provider


class ModelTimeoutException(ModelException):
    """Exception raised when model call times out."""

    pass


class ModelRateLimitException(ModelException):
    """Exception raised when hitting rate limits."""

    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after"] = retry_after


class ValidationException(AgentException):
    """Exception raised during validation."""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        failures: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        self.failures = failures or []
        if validation_type:
            self.details["validation_type"] = validation_type
        if failures:
            self.details["failures"] = failures


class ConfigurationException(AgentException):
    """Exception raised for configuration errors."""

    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        missing_keys: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.config_path = config_path
        self.missing_keys = missing_keys or []
        if config_path:
            self.details["config_path"] = config_path
        if missing_keys:
            self.details["missing_keys"] = missing_keys


class OrchestrationException(AgentException):
    """Exception raised during orchestration."""

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.stage = stage
        self.state = state
        if stage:
            self.details["stage"] = stage
        if state:
            self.details["state"] = state


class ToolException(AgentException):
    """Exception raised when tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        exit_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.exit_code = exit_code
        if tool_name:
            self.details["tool_name"] = tool_name
        if exit_code is not None:
            self.details["exit_code"] = exit_code


class SecurityException(AgentException):
    """Exception raised for security-related issues."""

    def __init__(
        self,
        message: str,
        vulnerability_type: Optional[str] = None,
        severity: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.vulnerability_type = vulnerability_type
        self.severity = severity
        if vulnerability_type:
            self.details["vulnerability_type"] = vulnerability_type
        if severity:
            self.details["severity"] = severity


class MemoryException(AgentException):
    """Exception raised for memory storage issues."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.key = key
        if operation:
            self.details["operation"] = operation
        if key:
            self.details["key"] = key


class GovernanceException(AgentException):
    """Exception raised when governance checks fail."""

    def __init__(
        self,
        message: str,
        check_type: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.check_type = check_type
        self.metrics = metrics
        if check_type:
            self.details["check_type"] = check_type
        if metrics:
            self.details["metrics"] = metrics


class RetryableException(AgentException):
    """Base class for exceptions that can be retried."""

    def __init__(
        self, message: str, max_retries: int = 3, retry_delay: float = 1.0, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.details["max_retries"] = max_retries
        self.details["retry_delay"] = retry_delay


def handle_exception(
    exc: Exception, context: Optional[Dict[str, Any]] = None
) -> AgentException:
    """Convert any exception to an AgentException with context."""
    if isinstance(exc, AgentException):
        return exc

    # Map common exceptions to specific types
    if isinstance(exc, TimeoutError):
        return ModelTimeoutException(
            f"Operation timed out: {str(exc)}", details=context, cause=exc
        )

    if isinstance(exc, FileNotFoundError):
        return ConfigurationException(
            f"File not found: {str(exc)}", details=context, cause=exc
        )

    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return ValidationException(
            f"Data validation failed: {str(exc)}",
            validation_type="json_parse"
            if isinstance(exc, json.JSONDecodeError)
            else "value",
            details=context,
            cause=exc,
        )

    # Default conversion
    return AgentException(f"Unexpected error: {str(exc)}", details=context, cause=exc)


__all__ = [
    "AgentException",
    "ModelException",
    "ModelTimeoutException",
    "ModelRateLimitException",
    "ValidationException",
    "ConfigurationException",
    "OrchestrationException",
    "ToolException",
    "SecurityException",
    "MemoryException",
    "GovernanceException",
    "RetryableException",
    "handle_exception",
]
