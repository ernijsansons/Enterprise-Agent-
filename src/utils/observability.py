"""Enhanced observability instrumentation for Enterprise Agent.

This module provides comprehensive observability features including:
- Detailed execution tracing
- Performance metrics collection
- Error correlation and analysis
- Agent pipeline health monitoring
- Debug and audit logging
"""
from __future__ import annotations

import json
import logging
import time
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager

from src.utils.errors import EnterpriseAgentError


class TraceLevel(Enum):
    """Trace severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(Enum):
    """System component types for tracing."""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    CODER = "coder"
    VALIDATOR = "validator"
    REFLECTOR = "reflector"
    REVIEWER = "reviewer"
    GOVERNANCE = "governance"
    PROVIDER = "provider"
    MEMORY = "memory"
    CACHE = "cache"


@dataclass
class TraceEvent:
    """Individual trace event with detailed context."""
    id: str
    timestamp: float
    component: ComponentType
    operation: str
    level: TraceLevel
    message: str
    context: Dict[str, Any]
    duration_ms: Optional[float] = None
    parent_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['component'] = self.component.value
        data['level'] = self.level.value
        return data


@dataclass
class ExecutionSpan:
    """Execution span for tracking operation duration and context."""
    id: str
    parent_id: Optional[str]
    correlation_id: str
    component: ComponentType
    operation: str
    start_time: float
    context: Dict[str, Any]
    end_time: Optional[float] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['component'] = self.component.value
        data['duration_ms'] = self.duration_ms
        return data


@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring."""
    component: ComponentType
    operation: str
    count: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    error_count: int = 0
    success_rate: float = 0.0
    last_updated: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        return self.total_duration_ms / self.count if self.count > 0 else 0.0

    def update(self, duration_ms: float, success: bool = True) -> None:
        """Update metrics with new measurement."""
        self.count += 1
        self.total_duration_ms += duration_ms
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        if not success:
            self.error_count += 1
        self.success_rate = (self.count - self.error_count) / self.count
        self.last_updated = time.time()


class ObservabilityConfig:
    """Configuration for observability features."""

    def __init__(
        self,
        enabled: bool = True,
        trace_level: TraceLevel = TraceLevel.INFO,
        max_trace_events: int = 10000,
        max_spans: int = 1000,
        retention_hours: int = 24,
        export_enabled: bool = False,
        export_path: Optional[str] = None,
        metrics_enabled: bool = True,
        health_check_enabled: bool = True,
        detailed_errors: bool = True
    ):
        self.enabled = enabled
        self.trace_level = trace_level
        self.max_trace_events = max_trace_events
        self.max_spans = max_spans
        self.retention_hours = retention_hours
        self.export_enabled = export_enabled
        self.export_path = Path(export_path) if export_path else None
        self.metrics_enabled = metrics_enabled
        self.health_check_enabled = health_check_enabled
        self.detailed_errors = detailed_errors


class ObservabilityCollector:
    """Central observability collector for Enterprise Agent."""

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        self.config = config or ObservabilityConfig()
        self.logger = logging.getLogger(__name__)

        # Thread-safe collections
        self._lock = threading.RLock()
        self._trace_events: deque = deque(maxlen=self.config.max_trace_events)
        self._active_spans: Dict[str, ExecutionSpan] = {}
        self._completed_spans: deque = deque(maxlen=self.config.max_spans)
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._correlation_contexts: Dict[str, Dict[str, Any]] = {}

        # Health monitoring
        self._health_status: Dict[ComponentType, Dict[str, Any]] = defaultdict(dict)
        self._error_patterns: Dict[str, int] = defaultdict(int)

        # Export tracking
        self._last_export = 0.0

    def start_span(
        self,
        component: ComponentType,
        operation: str,
        parent_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new execution span."""
        if not self.config.enabled:
            return ""

        span_id = str(uuid.uuid4())
        correlation_id = correlation_id or str(uuid.uuid4())

        span = ExecutionSpan(
            id=span_id,
            parent_id=parent_id,
            correlation_id=correlation_id,
            component=component,
            operation=operation,
            start_time=time.time(),
            context=context or {},
            metadata={}
        )

        with self._lock:
            self._active_spans[span_id] = span
            # Store correlation context
            if correlation_id not in self._correlation_contexts:
                self._correlation_contexts[correlation_id] = {}
            self._correlation_contexts[correlation_id].update(context or {})

        self.trace(
            component=component,
            operation=operation,
            level=TraceLevel.DEBUG,
            message=f"Started {operation}",
            context={"span_id": span_id, "correlation_id": correlation_id},
            correlation_id=correlation_id
        )

        return span_id

    def end_span(
        self,
        span_id: str,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """End an execution span."""
        if not self.config.enabled or not span_id:
            return

        with self._lock:
            span = self._active_spans.pop(span_id, None)
            if not span:
                return

            span.end_time = time.time()
            span.success = success
            span.error = error
            span.metadata = metadata or {}

            # Update metrics
            if self.config.metrics_enabled and span.duration_ms is not None:
                self._update_metrics(span)

            # Store completed span
            self._completed_spans.append(span)

            # Update health status
            self._update_health_status(span)

        self.trace(
            component=span.component,
            operation=span.operation,
            level=TraceLevel.DEBUG if success else TraceLevel.ERROR,
            message=f"Completed {span.operation}" + (f" with error: {error}" if error else ""),
            context={
                "span_id": span_id,
                "duration_ms": span.duration_ms,
                "success": success
            },
            correlation_id=span.correlation_id
        )

    @contextmanager
    def span(
        self,
        component: ComponentType,
        operation: str,
        parent_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Context manager for automatic span management."""
        span_id = self.start_span(component, operation, parent_id, correlation_id, context)
        try:
            yield span_id
            self.end_span(span_id, success=True)
        except Exception as e:
            self.end_span(span_id, success=False, error=str(e))
            raise

    def trace(
        self,
        component: ComponentType,
        operation: str,
        level: TraceLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        duration_ms: Optional[float] = None
    ) -> None:
        """Record a trace event."""
        if not self.config.enabled or level.value < self.config.trace_level.value:
            return

        event = TraceEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            component=component,
            operation=operation,
            level=level,
            message=message,
            context=context or {},
            duration_ms=duration_ms,
            correlation_id=correlation_id
        )

        with self._lock:
            self._trace_events.append(event)

        # Log to standard logger as well
        log_level = {
            TraceLevel.DEBUG: logging.DEBUG,
            TraceLevel.INFO: logging.INFO,
            TraceLevel.WARNING: logging.WARNING,
            TraceLevel.ERROR: logging.ERROR,
            TraceLevel.CRITICAL: logging.CRITICAL
        }[level]

        self.logger.log(
            log_level,
            f"[{component.value}:{operation}] {message}",
            extra={"observability_context": context, "correlation_id": correlation_id}
        )

    def trace_error(
        self,
        component: ComponentType,
        operation: str,
        error: Union[Exception, EnterpriseAgentError],
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Record an error trace with enhanced context."""
        error_context = context or {}

        if isinstance(error, EnterpriseAgentError):
            error_context.update({
                "error_code": error.details.code.name,
                "error_category": error.details.category.value,
                "error_severity": error.details.severity.value,
                "recovery_suggestions": error.details.recovery_suggestions
            })
            message = f"Structured error: {error.details.message}"
        else:
            error_context.update({
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
            message = f"Exception: {str(error)}"

        self.trace(
            component=component,
            operation=operation,
            level=TraceLevel.ERROR,
            message=message,
            context=error_context,
            correlation_id=correlation_id
        )

        # Track error patterns
        error_pattern = f"{component.value}:{operation}:{type(error).__name__}"
        with self._lock:
            self._error_patterns[error_pattern] += 1

    def _update_metrics(self, span: ExecutionSpan) -> None:
        """Update performance metrics from completed span."""
        if span.duration_ms is None:
            return

        metric_key = f"{span.component.value}:{span.operation}"

        if metric_key not in self._metrics:
            self._metrics[metric_key] = PerformanceMetrics(
                component=span.component,
                operation=span.operation
            )

        self._metrics[metric_key].update(span.duration_ms, span.success or False)

    def _update_health_status(self, span: ExecutionSpan) -> None:
        """Update component health status."""
        component_health = self._health_status[span.component]

        if span.operation not in component_health:
            component_health[span.operation] = {
                "total_calls": 0,
                "successful_calls": 0,
                "last_success": None,
                "last_failure": None,
                "avg_duration_ms": 0.0
            }

        operation_health = component_health[span.operation]
        operation_health["total_calls"] += 1

        if span.success:
            operation_health["successful_calls"] += 1
            operation_health["last_success"] = span.end_time
        else:
            operation_health["last_failure"] = span.end_time

        # Update average duration
        if span.duration_ms is not None:
            total_duration = operation_health["avg_duration_ms"] * (operation_health["total_calls"] - 1)
            operation_health["avg_duration_ms"] = (total_duration + span.duration_ms) / operation_health["total_calls"]

    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        with self._lock:
            health_summary = {}
            overall_health = "healthy"

            for component, operations in self._health_status.items():
                component_health = {}
                component_status = "healthy"

                for operation, metrics in operations.items():
                    success_rate = metrics["successful_calls"] / metrics["total_calls"] if metrics["total_calls"] > 0 else 1.0

                    operation_status = "healthy"
                    if success_rate < 0.8:
                        operation_status = "degraded"
                        component_status = "degraded"
                    if success_rate < 0.5:
                        operation_status = "critical"
                        component_status = "critical"
                        overall_health = "critical"

                    component_health[operation] = {
                        "status": operation_status,
                        "success_rate": success_rate,
                        "total_calls": metrics["total_calls"],
                        "avg_duration_ms": metrics["avg_duration_ms"],
                        "last_success": metrics["last_success"],
                        "last_failure": metrics["last_failure"]
                    }

                health_summary[component.value] = {
                    "status": component_status,
                    "operations": component_health
                }

            return {
                "overall_status": overall_health,
                "timestamp": time.time(),
                "components": health_summary
            }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        with self._lock:
            metrics_data = {}
            for key, metrics in self._metrics.items():
                metrics_data[key] = {
                    "component": metrics.component.value,
                    "operation": metrics.operation,
                    "count": metrics.count,
                    "avg_duration_ms": metrics.avg_duration_ms,
                    "min_duration_ms": metrics.min_duration_ms if metrics.min_duration_ms != float('inf') else 0,
                    "max_duration_ms": metrics.max_duration_ms,
                    "success_rate": metrics.success_rate,
                    "error_count": metrics.error_count,
                    "last_updated": metrics.last_updated
                }

            return {
                "timestamp": time.time(),
                "metrics": metrics_data,
                "total_operations": sum(m.count for m in self._metrics.values()),
                "total_errors": sum(m.error_count for m in self._metrics.values())
            }

    def get_error_analysis(self) -> Dict[str, Any]:
        """Get error pattern analysis."""
        with self._lock:
            recent_cutoff = time.time() - 3600  # Last hour
            recent_errors = [
                event for event in self._trace_events
                if event.level == TraceLevel.ERROR and event.timestamp > recent_cutoff
            ]

            error_by_component = defaultdict(int)
            error_by_operation = defaultdict(int)

            for event in recent_errors:
                error_by_component[event.component.value] += 1
                error_by_operation[f"{event.component.value}:{event.operation}"] += 1

            return {
                "timestamp": time.time(),
                "recent_error_count": len(recent_errors),
                "error_patterns": dict(self._error_patterns),
                "errors_by_component": dict(error_by_component),
                "errors_by_operation": dict(error_by_operation),
                "top_error_patterns": sorted(
                    self._error_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }

    def get_trace_summary(
        self,
        correlation_id: Optional[str] = None,
        component: Optional[ComponentType] = None,
        hours_back: int = 1
    ) -> Dict[str, Any]:
        """Get trace events summary with optional filtering."""
        cutoff_time = time.time() - (hours_back * 3600)

        with self._lock:
            # Filter events
            filtered_events = []
            for event in self._trace_events:
                if event.timestamp < cutoff_time:
                    continue
                if correlation_id and event.correlation_id != correlation_id:
                    continue
                if component and event.component != component:
                    continue
                filtered_events.append(event)

            # Filter spans
            filtered_spans = []
            for span in self._completed_spans:
                if span.start_time < cutoff_time:
                    continue
                if correlation_id and span.correlation_id != correlation_id:
                    continue
                if component and span.component != component:
                    continue
                filtered_spans.append(span)

        return {
            "timestamp": time.time(),
            "filter_criteria": {
                "correlation_id": correlation_id,
                "component": component.value if component else None,
                "hours_back": hours_back
            },
            "events": [event.to_dict() for event in filtered_events],
            "spans": [span.to_dict() for span in filtered_spans],
            "event_count": len(filtered_events),
            "span_count": len(filtered_spans)
        }

    def export_observability_data(self, file_path: Optional[str] = None) -> str:
        """Export all observability data to JSON file."""
        if not self.config.export_enabled:
            return ""

        export_path = file_path or (self.config.export_path / f"observability_{int(time.time())}.json")
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "export_timestamp": time.time(),
            "config": {
                "trace_level": self.config.trace_level.value,
                "retention_hours": self.config.retention_hours,
                "metrics_enabled": self.config.metrics_enabled
            },
            "health_status": self.get_health_status(),
            "metrics_summary": self.get_metrics_summary(),
            "error_analysis": self.get_error_analysis(),
            "trace_summary": self.get_trace_summary(hours_back=self.config.retention_hours)
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        self._last_export = time.time()
        return str(export_path)

    def cleanup_old_data(self) -> None:
        """Cleanup old observability data based on retention policy."""
        cutoff_time = time.time() - (self.config.retention_hours * 3600)

        with self._lock:
            # Clean old trace events
            while self._trace_events and self._trace_events[0].timestamp < cutoff_time:
                self._trace_events.popleft()

            # Clean old spans
            while self._completed_spans and self._completed_spans[0].start_time < cutoff_time:
                self._completed_spans.popleft()

            # Clean old correlation contexts
            old_contexts = [
                cid for cid, ctx in self._correlation_contexts.items()
                if ctx.get('timestamp', 0) < cutoff_time
            ]
            for cid in old_contexts:
                del self._correlation_contexts[cid]


# Global observability collector instance
_global_collector: Optional[ObservabilityCollector] = None


def get_observability_collector(config: Optional[ObservabilityConfig] = None) -> ObservabilityCollector:
    """Get or create the global observability collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = ObservabilityCollector(config)
    return _global_collector


def reset_observability_collector() -> None:
    """Reset the global observability collector (for testing)."""
    global _global_collector
    _global_collector = None


# Convenience functions for common observability operations
def trace_operation(
    component: ComponentType,
    operation: str,
    message: str,
    level: TraceLevel = TraceLevel.INFO,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Trace an operation with the global collector."""
    collector = get_observability_collector()
    collector.trace(component, operation, level, message, context, correlation_id)


def trace_error(
    component: ComponentType,
    operation: str,
    error: Union[Exception, EnterpriseAgentError],
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """Trace an error with the global collector."""
    collector = get_observability_collector()
    collector.trace_error(component, operation, error, context, correlation_id)


@contextmanager
def observe_operation(
    component: ComponentType,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
):
    """Context manager for observing an operation."""
    collector = get_observability_collector()
    with collector.span(component, operation, context=context, correlation_id=correlation_id) as span_id:
        yield span_id


__all__ = [
    "TraceLevel",
    "ComponentType",
    "TraceEvent",
    "ExecutionSpan",
    "PerformanceMetrics",
    "ObservabilityConfig",
    "ObservabilityCollector",
    "get_observability_collector",
    "reset_observability_collector",
    "trace_operation",
    "trace_error",
    "observe_operation"
]