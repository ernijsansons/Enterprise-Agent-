"""Circuit breaker pattern implementation for resilient service calls."""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, calls blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Call timeout in seconds
    expected_exceptions: tuple = (Exception,)  # Exceptions that count as failures


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitBreaker:
    """Circuit breaker implementation with thread safety."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """Initialize circuit breaker.

        Args:
            name: Unique name for this circuit breaker
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_call_time = 0.0
        self._lock = threading.Lock()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If function raises an exception
        """
        with self._lock:
            self._update_state()

            if self.state == CircuitState.OPEN:
                time_since_failure = time.time() - self.last_failure_time
                retry_after = max(0, self.config.recovery_timeout - time_since_failure)

                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Retry after {retry_after:.1f} seconds.",
                    retry_after=retry_after,
                )

            elif self.state == CircuitState.HALF_OPEN:
                # Only allow one call in half-open state
                if time.time() - self.last_call_time < 1.0:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN. "
                        "Another call is already in progress.",
                        retry_after=1.0,
                    )

        # Execute the function
        start_time = time.time()
        self.last_call_time = start_time

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except self.config.expected_exceptions:
            self._record_failure()
            raise

        except Exception as e:
            # Unexpected exceptions don't count as circuit breaker failures
            logger.warning(
                f"Unexpected exception in circuit breaker '{self.name}': {e}"
            )
            raise

    def _update_state(self) -> None:
        """Update circuit breaker state based on current conditions."""
        now = time.time()

        if self.state == CircuitState.OPEN:
            # Check if enough time has passed to try half-open
            if now - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")

        elif self.state == CircuitState.HALF_OPEN:
            # Check if we should close based on recent successes
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' moved to CLOSED")

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)

            logger.debug(f"Circuit breaker '{self.name}' recorded success")

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit breaker '{self.name}' opened after "
                        f"{self.failure_count} failures"
                    )

            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open moves back to open
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' reopened after failure")

    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics.

        Returns:
            Dictionary with current stats
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "last_call_time": self.last_call_time,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout,
                },
            }

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0.0
            self.last_call_time = 0.0
            logger.info(f"Circuit breaker '{self.name}' reset to CLOSED")

    def force_open(self) -> None:
        """Force circuit breaker to open state (for testing/maintenance)."""
        with self._lock:
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.warning(f"Circuit breaker '{self.name}' forced to OPEN")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        """Initialize circuit breaker registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration (uses default if not provided)

        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if name not in self._breakers:
                if config is None:
                    config = CircuitBreakerConfig()
                self._breakers[name] = CircuitBreaker(name, config)
                logger.info(f"Created circuit breaker '{name}'")

            return self._breakers[name]

    def remove_breaker(self, name: str) -> bool:
        """Remove a circuit breaker.

        Args:
            name: Circuit breaker name

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                logger.info(f"Removed circuit breaker '{name}'")
                return True
            return False

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers.

        Returns:
            Dictionary with stats for all breakers
        """
        with self._lock:
            return {
                name: breaker.get_stats() for name, breaker in self._breakers.items()
            }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")

    def get_failing_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers that are currently open.

        Returns:
            Dictionary of failing circuit breakers
        """
        with self._lock:
            return {
                name: breaker
                for name, breaker in self._breakers.items()
                if breaker.state == CircuitState.OPEN
            }


# Global registry instance
_global_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry.

    Returns:
        Global circuit breaker registry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: float = 30.0,
    expected_exceptions: tuple = (Exception,),
):
    """Decorator for adding circuit breaker protection to functions.

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before trying half-open
        success_threshold: Successes needed to close from half-open
        timeout: Call timeout in seconds
        expected_exceptions: Exceptions that count as failures

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exceptions=expected_exceptions,
        )

        registry = get_circuit_breaker_registry()
        breaker = registry.get_breaker(name, config)

        def wrapper(*args, **kwargs) -> T:
            return breaker.call(func, *args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._circuit_breaker = breaker  # Allow access to breaker

        return wrapper

    return decorator


def setup_default_circuit_breakers() -> None:
    """Setup default circuit breakers for common services."""
    registry = get_circuit_breaker_registry()

    # Claude Code CLI - more lenient since it's our primary service
    registry.get_breaker(
        "claude_code_cli",
        CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=60.0,
            expected_exceptions=(Exception,),
        ),
    )

    # OpenAI API - stricter to avoid costs
    registry.get_breaker(
        "openai_api",
        CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            success_threshold=2,
            timeout=30.0,
            expected_exceptions=(Exception,),
        ),
    )

    # Anthropic API - backup only, very strict
    registry.get_breaker(
        "anthropic_api",
        CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=120.0,
            success_threshold=3,
            timeout=30.0,
            expected_exceptions=(Exception,),
        ),
    )

    # Gemini API
    registry.get_breaker(
        "gemini_api",
        CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            success_threshold=2,
            timeout=30.0,
            expected_exceptions=(Exception,),
        ),
    )

    logger.info("Default circuit breakers configured")


__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    "circuit_breaker",
    "setup_default_circuit_breakers",
]
