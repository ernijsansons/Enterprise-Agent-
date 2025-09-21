"""Structured error handling system for Enterprise Agent.

This module provides a comprehensive error classification and handling system
with error codes, severity levels, and structured logging.
"""
from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for classification."""
    ORCHESTRATION = "orchestration"
    MODEL_CALL = "model_call"
    VALIDATION = "validation"
    REFLECTION = "reflection"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PROVIDER = "provider"
    CACHE = "cache"
    MEMORY = "memory"
    GOVERNANCE = "governance"
    ASYNC_OPERATION = "async_operation"
    SECURITY = "security"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    USER_INPUT = "user_input"
    SYSTEM = "system"


class ErrorCode(Enum):
    """Structured error codes for Enterprise Agent operations."""

    # Orchestration errors (1000-1099)
    ORCHESTRATION_INIT_FAILED = 1000
    ORCHESTRATION_GRAPH_BUILD_FAILED = 1001
    ORCHESTRATION_PIPELINE_FAILED = 1002
    ORCHESTRATION_STATE_CORRUPTION = 1003
    ORCHESTRATION_ROLE_ROUTING_FAILED = 1004

    # Model call errors (1100-1199)
    MODEL_CALL_FAILED = 1100
    MODEL_CLIENT_UNAVAILABLE = 1101
    MODEL_TIMEOUT = 1102
    MODEL_RATE_LIMITED = 1103
    MODEL_AUTHENTICATION_FAILED = 1104
    MODEL_QUOTA_EXCEEDED = 1105
    MODEL_INVALID_RESPONSE = 1106

    # Validation errors (1200-1299)
    VALIDATION_FAILED = 1200
    VALIDATION_TIMEOUT = 1201
    VALIDATION_PARSE_ERROR = 1202
    VALIDATION_COVERAGE_INSUFFICIENT = 1203

    # Reflection errors (1300-1399)
    REFLECTION_LOOP_FAILED = 1300
    REFLECTION_MAX_ITERATIONS_REACHED = 1301
    REFLECTION_STAGNATION_DETECTED = 1302
    REFLECTION_CONFIDENCE_REGRESSION = 1303
    REFLECTION_EARLY_TERMINATION = 1304

    # Configuration errors (1400-1499)
    CONFIG_FILE_NOT_FOUND = 1400
    CONFIG_PARSE_ERROR = 1401
    CONFIG_VALIDATION_FAILED = 1402
    CONFIG_MISSING_REQUIRED_FIELD = 1403

    # Authentication errors (1500-1599)
    AUTH_SETUP_FAILED = 1500
    AUTH_LOGIN_REQUIRED = 1501
    AUTH_TOKEN_EXPIRED = 1502
    AUTH_PROVIDER_UNAVAILABLE = 1503

    # Provider errors (1600-1699)
    PROVIDER_INIT_FAILED = 1600
    PROVIDER_CONNECTION_FAILED = 1601
    PROVIDER_FALLBACK_FAILED = 1602
    PROVIDER_SESSION_FAILED = 1603

    # Cache errors (1700-1799)
    CACHE_OPERATION_FAILED = 1700
    CACHE_MEMORY_EXHAUSTED = 1701
    CACHE_CORRUPTION_DETECTED = 1702

    # Memory errors (1800-1899)
    MEMORY_ALLOCATION_FAILED = 1800
    MEMORY_STORE_FAILED = 1801
    MEMORY_RETRIEVAL_FAILED = 1802
    MEMORY_PRUNING_FAILED = 1803

    # Governance errors (1900-1999)
    GOVERNANCE_CHECK_FAILED = 1900
    GOVERNANCE_POLICY_VIOLATION = 1901
    GOVERNANCE_HITL_REQUIRED = 1902

    # Async operation errors (2000-2099)
    ASYNC_OPERATION_FAILED = 2000
    ASYNC_TIMEOUT = 2001
    ASYNC_CANCELLATION = 2002
    ASYNC_EXECUTOR_FAILED = 2003

    # Security errors (2100-2199)
    SECURITY_VALIDATION_FAILED = 2100
    SECURITY_PII_DETECTED = 2101
    SECURITY_COMMAND_INJECTION = 2102
    SECURITY_RATE_LIMIT_EXCEEDED = 2103

    # Timeout errors (2200-2299)
    OPERATION_TIMEOUT = 2200
    MODEL_CALL_TIMEOUT = 2201
    NETWORK_TIMEOUT = 2202

    # Resource errors (2300-2399)
    RESOURCE_EXHAUSTED = 2300
    DISK_SPACE_INSUFFICIENT = 2301
    MEMORY_LIMIT_EXCEEDED = 2302
    CPU_LIMIT_EXCEEDED = 2303

    # User input errors (2400-2499)
    INVALID_DOMAIN = 2400
    INVALID_TASK = 2401
    INVALID_PARAMETERS = 2402
    MISSING_REQUIRED_INPUT = 2403

    # System errors (2500-2599)
    SYSTEM_ERROR = 2500
    DEPENDENCY_UNAVAILABLE = 2501
    FILE_SYSTEM_ERROR = 2502
    NETWORK_ERROR = 2503


@dataclass
class ErrorDetails:
    """Detailed error information."""
    code: ErrorCode
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = None
    user_message: Optional[str] = None

    def __post_init__(self):
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        # Convert enums to values for JSON serialization
        data['code'] = self.code.value
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        return json.dumps(data, indent=2)


class EnterpriseAgentError(Exception):
    """Base exception class for Enterprise Agent with structured error handling."""

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        user_message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)

        # Auto-infer category and severity from error code if not provided
        if category is None:
            category = self._infer_category(error_code)
        if severity is None:
            severity = self._infer_severity(error_code)

        self.details = ErrorDetails(
            code=error_code,
            message=message,
            category=category,
            severity=severity,
            timestamp=time.time(),
            context=context or {},
            stack_trace=self._get_stack_trace(),
            recovery_suggestions=recovery_suggestions or [],
            user_message=user_message
        )
        self.cause = cause

    def _infer_category(self, error_code: ErrorCode) -> ErrorCategory:
        """Infer error category from error code."""
        code_value = error_code.value

        if 1000 <= code_value < 1100:
            return ErrorCategory.ORCHESTRATION
        elif 1100 <= code_value < 1200:
            return ErrorCategory.MODEL_CALL
        elif 1200 <= code_value < 1300:
            return ErrorCategory.VALIDATION
        elif 1300 <= code_value < 1400:
            return ErrorCategory.REFLECTION
        elif 1400 <= code_value < 1500:
            return ErrorCategory.CONFIGURATION
        elif 1500 <= code_value < 1600:
            return ErrorCategory.AUTHENTICATION
        elif 1600 <= code_value < 1700:
            return ErrorCategory.PROVIDER
        elif 1700 <= code_value < 1800:
            return ErrorCategory.CACHE
        elif 1800 <= code_value < 1900:
            return ErrorCategory.MEMORY
        elif 1900 <= code_value < 2000:
            return ErrorCategory.GOVERNANCE
        elif 2000 <= code_value < 2100:
            return ErrorCategory.ASYNC_OPERATION
        elif 2100 <= code_value < 2200:
            return ErrorCategory.SECURITY
        elif 2200 <= code_value < 2300:
            return ErrorCategory.TIMEOUT
        elif 2300 <= code_value < 2400:
            return ErrorCategory.RESOURCE
        elif 2400 <= code_value < 2500:
            return ErrorCategory.USER_INPUT
        else:
            return ErrorCategory.SYSTEM

    def _infer_severity(self, error_code: ErrorCode) -> ErrorSeverity:
        """Infer error severity from error code."""
        # Critical errors that halt operation
        critical_codes = {
            ErrorCode.ORCHESTRATION_INIT_FAILED,
            ErrorCode.CONFIG_FILE_NOT_FOUND,
            ErrorCode.MEMORY_ALLOCATION_FAILED,
            ErrorCode.SECURITY_COMMAND_INJECTION,
            ErrorCode.SYSTEM_ERROR
        }

        # High severity errors that significantly impact functionality
        high_severity_codes = {
            ErrorCode.ORCHESTRATION_PIPELINE_FAILED,
            ErrorCode.MODEL_CLIENT_UNAVAILABLE,
            ErrorCode.AUTH_SETUP_FAILED,
            ErrorCode.PROVIDER_INIT_FAILED,
            ErrorCode.GOVERNANCE_POLICY_VIOLATION,
            ErrorCode.SECURITY_VALIDATION_FAILED,
            ErrorCode.RESOURCE_EXHAUSTED
        }

        # Low severity errors that are mostly informational
        low_severity_codes = {
            ErrorCode.REFLECTION_EARLY_TERMINATION,
            ErrorCode.AUTH_LOGIN_REQUIRED,
            ErrorCode.CACHE_OPERATION_FAILED
        }

        if error_code in critical_codes:
            return ErrorSeverity.CRITICAL
        elif error_code in high_severity_codes:
            return ErrorSeverity.HIGH
        elif error_code in low_severity_codes:
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM

    def _get_stack_trace(self) -> Optional[str]:
        """Get current stack trace."""
        import traceback
        return traceback.format_exc()

    def add_context(self, key: str, value: Any) -> None:
        """Add context information to the error."""
        self.details.context[key] = value

    def add_recovery_suggestion(self, suggestion: str) -> None:
        """Add a recovery suggestion."""
        self.details.recovery_suggestions.append(suggestion)

    def set_user_message(self, message: str) -> None:
        """Set a user-friendly error message."""
        self.details.user_message = message


class ErrorHandler:
    """Centralized error handler for Enterprise Agent."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[ErrorDetails] = []
        self.error_counts: Dict[ErrorCode, int] = {}

    def handle_error(
        self,
        error: Union[Exception, EnterpriseAgentError],
        context: Optional[Dict[str, Any]] = None,
        log_level: Optional[int] = None
    ) -> ErrorDetails:
        """Handle an error with structured logging and tracking."""

        if isinstance(error, EnterpriseAgentError):
            details = error.details
            if context:
                details.context.update(context)
        else:
            # Convert generic exception to structured error
            details = ErrorDetails(
                code=ErrorCode.SYSTEM_ERROR,
                message=str(error),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                timestamp=time.time(),
                context=context or {},
                stack_trace=self._format_exception(error)
            )

        # Track error
        self._track_error(details)

        # Log error
        self._log_error(details, log_level)

        return details

    def _track_error(self, details: ErrorDetails) -> None:
        """Track error for analytics and monitoring."""
        self.error_history.append(details)
        self.error_counts[details.code] = self.error_counts.get(details.code, 0) + 1

        # Keep only recent errors to prevent memory growth
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]

    def _log_error(self, details: ErrorDetails, log_level: Optional[int] = None) -> None:
        """Log error with appropriate level."""
        if log_level is None:
            log_level = self._severity_to_log_level(details.severity)

        log_message = f"[{details.code.name}] {details.message}"

        # Add context information
        if details.context:
            context_str = ", ".join(f"{k}={v}" for k, v in details.context.items())
            log_message += f" | Context: {context_str}"

        # Add recovery suggestions
        if details.recovery_suggestions:
            suggestions = "; ".join(details.recovery_suggestions)
            log_message += f" | Suggestions: {suggestions}"

        self.logger.log(log_level, log_message)

        # Log stack trace for critical/high severity errors
        if details.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.HIGH) and details.stack_trace:
            self.logger.debug(f"Stack trace for {details.code.name}:\n{details.stack_trace}")

    def _severity_to_log_level(self, severity: ErrorSeverity) -> int:
        """Convert error severity to logging level."""
        severity_map = {
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.INFO: logging.INFO
        }
        return severity_map.get(severity, logging.WARNING)

    def _format_exception(self, error: Exception) -> str:
        """Format exception for logging."""
        import traceback
        return traceback.format_exc()

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {"total_errors": 0, "error_breakdown": {}}

        # Count by category and severity
        category_counts = {}
        severity_counts = {}

        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        # Recent errors (last 10)
        recent_errors = []
        for error in self.error_history[-10:]:
            recent_errors.append({
                "code": error.code.name,
                "message": error.message,
                "category": error.category.value,
                "severity": error.severity.value,
                "timestamp": error.timestamp
            })

        return {
            "total_errors": total_errors,
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "most_common_errors": dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "recent_errors": recent_errors
        }


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def handle_error(
    error: Union[Exception, EnterpriseAgentError],
    context: Optional[Dict[str, Any]] = None
) -> ErrorDetails:
    """Convenience function to handle errors using the global handler."""
    return get_error_handler().handle_error(error, context)


# Common error creation functions
def create_orchestration_error(
    message: str,
    error_code: ErrorCode = ErrorCode.ORCHESTRATION_PIPELINE_FAILED,
    context: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None
) -> EnterpriseAgentError:
    """Create an orchestration error."""
    return EnterpriseAgentError(
        error_code=error_code,
        message=message,
        context=context,
        cause=cause
    )


def create_model_error(
    message: str,
    model: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.MODEL_CALL_FAILED,
    context: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None
) -> EnterpriseAgentError:
    """Create a model call error."""
    if context is None:
        context = {}
    if model:
        context["model"] = model

    return EnterpriseAgentError(
        error_code=error_code,
        message=message,
        context=context,
        cause=cause
    )


def create_validation_error(
    message: str,
    validation_type: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.VALIDATION_FAILED,
    context: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None
) -> EnterpriseAgentError:
    """Create a validation error."""
    if context is None:
        context = {}
    if validation_type:
        context["validation_type"] = validation_type

    return EnterpriseAgentError(
        error_code=error_code,
        message=message,
        context=context,
        cause=cause
    )


def create_config_error(
    message: str,
    config_path: Optional[str] = None,
    error_code: ErrorCode = ErrorCode.CONFIG_VALIDATION_FAILED,
    context: Optional[Dict[str, Any]] = None,
    cause: Optional[Exception] = None
) -> EnterpriseAgentError:
    """Create a configuration error."""
    if context is None:
        context = {}
    if config_path:
        context["config_path"] = config_path

    return EnterpriseAgentError(
        error_code=error_code,
        message=message,
        context=context,
        cause=cause
    )