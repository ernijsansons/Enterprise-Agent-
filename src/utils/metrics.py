"""Metrics collection and observability system for Enterprise Agent.

This module provides comprehensive metrics collection, performance monitoring,
and observability features for the Enterprise Agent system.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    EVENT = "event"


class MetricSeverity(Enum):
    """Severity levels for metrics and events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: Union[float, int, str]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PerformanceEvent:
    """Performance event with timing and context."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True

    def finish(self, error: Optional[str] = None) -> None:
        """Finish the performance event."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = error is None
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MetricsCollector:
    """Centralized metrics collection and aggregation."""

    def __init__(
        self,
        enabled: bool = True,
        buffer_size: int = 10000,
        flush_interval: float = 60.0,
        export_path: Optional[Path] = None
    ):
        """Initialize metrics collector.

        Args:
            enabled: Whether metrics collection is enabled
            buffer_size: Maximum number of metrics to buffer
            flush_interval: Interval to flush metrics to storage
            export_path: Path to export metrics files
        """
        self.enabled = enabled
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.export_path = export_path or Path(".metrics")

        self._metrics_buffer: deque = deque(maxlen=buffer_size)
        self._events_buffer: deque = deque(maxlen=buffer_size)
        self._lock = threading.RLock()
        self._last_flush = time.time()

        # Aggregated metrics
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)

        # Performance tracking
        self._active_events: Dict[str, PerformanceEvent] = {}
        self._completed_events: List[PerformanceEvent] = []

        # Setup export directory
        if self.enabled:
            self.export_path.mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def record_counter(
        self,
        name: str,
        value: Union[float, int] = 1,
        tags: Optional[Dict[str, str]] = None,
        **metadata: Any
    ) -> None:
        """Record a counter metric.

        Args:
            name: Metric name
            value: Increment value
            tags: Optional tags
            **metadata: Additional metadata
        """
        if not self.enabled:
            return

        with self._lock:
            self._counters[name] += value
            metric = MetricPoint(
                name=name,
                value=value,
                metric_type=MetricType.COUNTER,
                tags=tags or {},
                metadata=metadata
            )
            self._metrics_buffer.append(metric)
            self._maybe_flush()

    def record_gauge(
        self,
        name: str,
        value: Union[float, int],
        tags: Optional[Dict[str, str]] = None,
        **metadata: Any
    ) -> None:
        """Record a gauge metric.

        Args:
            name: Metric name
            value: Current value
            tags: Optional tags
            **metadata: Additional metadata
        """
        if not self.enabled:
            return

        with self._lock:
            self._gauges[name] = value
            metric = MetricPoint(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                tags=tags or {},
                metadata=metadata
            )
            self._metrics_buffer.append(metric)
            self._maybe_flush()

    def record_histogram(
        self,
        name: str,
        value: Union[float, int],
        tags: Optional[Dict[str, str]] = None,
        **metadata: Any
    ) -> None:
        """Record a histogram metric.

        Args:
            name: Metric name
            value: Value to add to histogram
            tags: Optional tags
            **metadata: Additional metadata
        """
        if not self.enabled:
            return

        with self._lock:
            self._histograms[name].append(float(value))
            # Keep only recent values to prevent memory growth
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-500:]

            metric = MetricPoint(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                tags=tags or {},
                metadata=metadata
            )
            self._metrics_buffer.append(metric)
            self._maybe_flush()

    def record_timer(
        self,
        name: str,
        duration: float,
        tags: Optional[Dict[str, str]] = None,
        **metadata: Any
    ) -> None:
        """Record a timer metric.

        Args:
            name: Metric name
            duration: Duration in seconds
            tags: Optional tags
            **metadata: Additional metadata
        """
        if not self.enabled:
            return

        with self._lock:
            self._timers[name].append(duration)
            # Keep only recent values
            if len(self._timers[name]) > 1000:
                self._timers[name] = self._timers[name][-500:]

            metric = MetricPoint(
                name=name,
                value=duration,
                metric_type=MetricType.TIMER,
                tags=tags or {},
                metadata=metadata
            )
            self._metrics_buffer.append(metric)
            self._maybe_flush()

    def record_event(
        self,
        name: str,
        severity: MetricSeverity = MetricSeverity.INFO,
        message: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **metadata: Any
    ) -> None:
        """Record an event.

        Args:
            name: Event name
            severity: Event severity
            message: Optional message
            tags: Optional tags
            **metadata: Additional metadata
        """
        if not self.enabled:
            return

        event_data = {
            "name": name,
            "severity": severity.value,
            "message": message,
            "timestamp": time.time(),
            "tags": tags or {},
            "metadata": metadata
        }

        with self._lock:
            self._events_buffer.append(event_data)
            self._maybe_flush()

    def start_performance_event(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a performance event.

        Args:
            name: Event name
            context: Optional context data

        Returns:
            Event ID for later finishing
        """
        if not self.enabled:
            return ""

        event_id = f"{name}_{int(time.time() * 1000000)}"
        event = PerformanceEvent(
            name=name,
            start_time=time.time(),
            context=context or {}
        )

        with self._lock:
            self._active_events[event_id] = event

        return event_id

    def finish_performance_event(
        self,
        event_id: str,
        error: Optional[str] = None,
        **additional_context: Any
    ) -> Optional[PerformanceEvent]:
        """Finish a performance event.

        Args:
            event_id: Event ID from start_performance_event
            error: Optional error message
            **additional_context: Additional context data

        Returns:
            Completed performance event
        """
        if not self.enabled or not event_id:
            return None

        with self._lock:
            event = self._active_events.pop(event_id, None)
            if event:
                event.context.update(additional_context)
                event.finish(error)
                self._completed_events.append(event)

                # Record as timer metric
                self.record_timer(
                    f"{event.name}_duration",
                    event.duration or 0,
                    tags={"success": str(event.success)},
                    event_name=event.name,
                    error=error
                )

                return event

        return None

    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.

        Args:
            name: Timer name
            tags: Optional tags

        Example:
            with metrics.timer("model_call"):
                result = model.call(prompt)
        """
        return TimerContext(self, name, tags)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns:
            Dictionary containing metrics summary
        """
        with self._lock:
            summary = {
                "timestamp": time.time(),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "buffer_size": len(self._metrics_buffer),
                "events_count": len(self._events_buffer),
                "active_events": len(self._active_events),
                "completed_events": len(self._completed_events)
            }

            # Add histogram statistics
            histogram_stats = {}
            for name, values in self._histograms.items():
                if values:
                    histogram_stats[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "p50": self._percentile(values, 0.5),
                        "p95": self._percentile(values, 0.95),
                        "p99": self._percentile(values, 0.99)
                    }
            summary["histograms"] = histogram_stats

            # Add timer statistics
            timer_stats = {}
            for name, values in self._timers.items():
                if values:
                    timer_stats[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "p50": self._percentile(values, 0.5),
                        "p95": self._percentile(values, 0.95),
                        "p99": self._percentile(values, 0.99)
                    }
            summary["timers"] = timer_stats

            return summary

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        with self._lock:
            events = list(self._events_buffer)[-limit:]
            return events

    def get_performance_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent performance events
        """
        with self._lock:
            events = [event.to_dict() for event in self._completed_events[-limit:]]
            return events

    def flush(self) -> None:
        """Flush metrics to storage."""
        if not self.enabled:
            return

        with self._lock:
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y%m%d")
            hour_str = timestamp.strftime("%H")

            # Export metrics
            if self._metrics_buffer:
                metrics_file = self.export_path / f"metrics_{date_str}_{hour_str}.jsonl"
                with metrics_file.open("a") as f:
                    for metric in self._metrics_buffer:
                        f.write(json.dumps(metric.to_dict()) + "\n")
                self._metrics_buffer.clear()

            # Export events
            if self._events_buffer:
                events_file = self.export_path / f"events_{date_str}_{hour_str}.jsonl"
                with events_file.open("a") as f:
                    for event in self._events_buffer:
                        f.write(json.dumps(event) + "\n")
                self._events_buffer.clear()

            # Export performance events
            if self._completed_events:
                perf_file = self.export_path / f"performance_{date_str}_{hour_str}.jsonl"
                with perf_file.open("a") as f:
                    for event in self._completed_events:
                        f.write(json.dumps(event.to_dict()) + "\n")
                self._completed_events.clear()

            # Export summary
            summary_file = self.export_path / f"summary_{date_str}_{hour_str}.json"
            with summary_file.open("w") as f:
                json.dump(self.get_summary(), f, indent=2)

            self._last_flush = time.time()

    def _maybe_flush(self) -> None:
        """Flush if enough time has passed."""
        if time.time() - self._last_flush > self.flush_interval:
            self.flush()

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        else:
            return sorted_values[f]

    def clear(self) -> None:
        """Clear all metrics and events."""
        with self._lock:
            self._metrics_buffer.clear()
            self._events_buffer.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._active_events.clear()
            self._completed_events.clear()


class TimerContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
        self.event_id = None

    def __enter__(self):
        self.start_time = time.time()
        self.event_id = self.collector.start_performance_event(
            self.name,
            {"tags": self.tags} if self.tags else None
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            error = str(exc_val) if exc_val else None

            self.collector.record_timer(
                self.name,
                duration,
                tags=self.tags,
                error=error,
                success=exc_type is None
            )

            if self.event_id:
                self.collector.finish_performance_event(
                    self.event_id,
                    error=error
                )


class MetricsConfig:
    """Configuration for metrics collection."""

    def __init__(
        self,
        enabled: bool = True,
        buffer_size: int = 10000,
        flush_interval: float = 60.0,
        export_path: Optional[str] = None,
        collect_system_metrics: bool = True,
        collect_performance_metrics: bool = True,
        collect_error_metrics: bool = True
    ):
        self.enabled = enabled
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.export_path = Path(export_path) if export_path else Path(".metrics")
        self.collect_system_metrics = collect_system_metrics
        self.collect_performance_metrics = collect_performance_metrics
        self.collect_error_metrics = collect_error_metrics

    @classmethod
    def from_env(cls) -> 'MetricsConfig':
        """Create configuration from environment variables."""
        return cls(
            enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            buffer_size=int(os.getenv("METRICS_BUFFER_SIZE", "10000")),
            flush_interval=float(os.getenv("METRICS_FLUSH_INTERVAL", "60.0")),
            export_path=os.getenv("METRICS_EXPORT_PATH"),
            collect_system_metrics=os.getenv("METRICS_COLLECT_SYSTEM", "true").lower() == "true",
            collect_performance_metrics=os.getenv("METRICS_COLLECT_PERFORMANCE", "true").lower() == "true",
            collect_error_metrics=os.getenv("METRICS_COLLECT_ERRORS", "true").lower() == "true"
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MetricsConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        config = MetricsConfig.from_env()
        _global_metrics_collector = MetricsCollector(
            enabled=config.enabled,
            buffer_size=config.buffer_size,
            flush_interval=config.flush_interval,
            export_path=config.export_path
        )
    return _global_metrics_collector


def initialize_metrics(config: Optional[MetricsConfig] = None) -> MetricsCollector:
    """Initialize the global metrics collector.

    Args:
        config: Optional metrics configuration

    Returns:
        Initialized metrics collector
    """
    global _global_metrics_collector
    config = config or MetricsConfig.from_env()
    _global_metrics_collector = MetricsCollector(
        enabled=config.enabled,
        buffer_size=config.buffer_size,
        flush_interval=config.flush_interval,
        export_path=config.export_path
    )
    return _global_metrics_collector


# Convenience functions for common metrics
def record_counter(name: str, value: Union[float, int] = 1, **kwargs) -> None:
    """Record a counter metric."""
    get_metrics_collector().record_counter(name, value, **kwargs)


def record_gauge(name: str, value: Union[float, int], **kwargs) -> None:
    """Record a gauge metric."""
    get_metrics_collector().record_gauge(name, value, **kwargs)


def record_timer(name: str, duration: float, **kwargs) -> None:
    """Record a timer metric."""
    get_metrics_collector().record_timer(name, duration, **kwargs)


def record_event(name: str, severity: MetricSeverity = MetricSeverity.INFO, **kwargs) -> None:
    """Record an event."""
    get_metrics_collector().record_event(name, severity, **kwargs)


def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for timing operations."""
    return get_metrics_collector().timer(name, tags)


__all__ = [
    "MetricType",
    "MetricSeverity",
    "MetricPoint",
    "PerformanceEvent",
    "MetricsCollector",
    "MetricsConfig",
    "TimerContext",
    "get_metrics_collector",
    "initialize_metrics",
    "record_counter",
    "record_gauge",
    "record_timer",
    "record_event",
    "timer"
]