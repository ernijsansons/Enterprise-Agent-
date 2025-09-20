"""Retry utilities with exponential backoff."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type

from src.exceptions import (
    ModelRateLimitException,
    ModelTimeoutException,
    RetryableException,
)

logger = logging.getLogger(__name__)


class RetryStrategy:
    """Base class for retry strategies."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt."""
        raise NotImplementedError

    def should_retry(
        self, exception: Exception, attempt: int
    ) -> Tuple[bool, Optional[float]]:
        """Determine if we should retry and the delay."""
        if attempt >= self.max_retries:
            return False, None

        if isinstance(exception, RetryableException):
            delay = exception.retry_delay
        elif isinstance(exception, ModelRateLimitException):
            delay = exception.retry_after or self.calculate_delay(attempt)
        elif isinstance(exception, (ModelTimeoutException, TimeoutError)):
            delay = self.calculate_delay(attempt)
        elif isinstance(exception, (ConnectionError, OSError)):
            delay = self.calculate_delay(attempt)
        else:
            # Don't retry unknown exceptions
            return False, None

        return True, min(delay, self.max_delay)


class ExponentialBackoff(RetryStrategy):
    """Exponential backoff retry strategy."""

    def __init__(self, exponential_base: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.exponential_base = exponential_base

    def calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (self.exponential_base**attempt)

        if self.jitter:
            # Add random jitter (0.5x to 1.5x the delay)
            jitter_factor = 0.5 + random.random()  # nosec B311
            delay *= jitter_factor

        return min(delay, self.max_delay)


class LinearBackoff(RetryStrategy):
    """Linear backoff retry strategy."""

    def calculate_delay(self, attempt: int) -> float:
        """Calculate linear backoff delay."""
        delay = self.base_delay * (attempt + 1)

        if self.jitter:
            # Add small random jitter
            delay += random.uniform(0, self.base_delay * 0.5)  # nosec B311

        return min(delay, self.max_delay)


class ConstantBackoff(RetryStrategy):
    """Constant delay retry strategy."""

    def calculate_delay(self, attempt: int) -> float:
        """Return constant delay."""
        delay = self.base_delay

        if self.jitter:
            # Add small random jitter
            delay += random.uniform(-0.1 * delay, 0.1 * delay)  # nosec B311

        return delay


def retry_with_backoff(
    strategy: Optional[RetryStrategy] = None,
    max_retries: Optional[int] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """Decorator for retrying functions with backoff."""
    if strategy is None:
        strategy = ExponentialBackoff(max_retries=max_retries or 3)
    elif max_retries is not None:
        strategy.max_retries = max_retries

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            attempt = 0
            last_exception = None

            while attempt <= strategy.max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    should_retry, delay = strategy.should_retry(exc, attempt)

                    if not should_retry:
                        break

                    if on_retry:
                        on_retry(exc, attempt)

                    logger.warning(
                        f"Retry {attempt + 1}/{strategy.max_retries} for {func.__name__} "
                        f"after {type(exc).__name__}: {exc}. "
                        f"Waiting {delay:.2f}s before retry."
                    )

                    time.sleep(delay)
                    attempt += 1

            # Exhausted retries, raise last exception
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Failed after {strategy.max_retries} retries")

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            attempt = 0
            last_exception = None

            while attempt <= strategy.max_retries:
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    should_retry, delay = strategy.should_retry(exc, attempt)

                    if not should_retry:
                        break

                    if on_retry:
                        on_retry(exc, attempt)

                    logger.warning(
                        f"Retry {attempt + 1}/{strategy.max_retries} for {func.__name__} "
                        f"after {type(exc).__name__}: {exc}. "
                        f"Waiting {delay:.2f}s before retry."
                    )

                    await asyncio.sleep(delay)
                    attempt += 1

            # Exhausted retries, raise last exception
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Failed after {strategy.max_retries} retries")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise RuntimeError(
                    f"Circuit breaker is open. Service unavailable for "
                    f"{self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as exc:
            self._on_failure()
            raise exc

    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Call async function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise RuntimeError(
                    f"Circuit breaker is open. Service unavailable for "
                    f"{self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as exc:
            self._on_failure()
            raise exc

    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit."""
        return (
            self.last_failure_time
            and time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == "half_open":
            logger.info("Circuit breaker reset to closed state")
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry in {self.recovery_timeout}s"
            )
        elif self.state == "half_open":
            self.state = "open"
            logger.warning("Circuit breaker reopened after failure in half_open state")

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None


def create_retry_decorator(
    retry_type: str = "exponential",
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
) -> Callable:
    """Factory function to create retry decorators."""
    strategies = {
        "exponential": ExponentialBackoff,
        "linear": LinearBackoff,
        "constant": ConstantBackoff,
    }

    strategy_class = strategies.get(retry_type, ExponentialBackoff)
    strategy = strategy_class(max_retries=max_retries, base_delay=base_delay, **kwargs)

    return lambda func: retry_with_backoff(strategy=strategy)(func)


# Convenience decorators
retry_on_failure = retry_with_backoff(
    strategy=ExponentialBackoff(max_retries=3, base_delay=1.0)
)

retry_on_timeout = retry_with_backoff(
    strategy=ExponentialBackoff(max_retries=3, base_delay=2.0),
    exceptions=(TimeoutError, ModelTimeoutException),
)

retry_on_rate_limit = retry_with_backoff(
    strategy=ExponentialBackoff(max_retries=5, base_delay=5.0),
    exceptions=(ModelRateLimitException,),
)


__all__ = [
    "RetryStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    "ConstantBackoff",
    "retry_with_backoff",
    "CircuitBreaker",
    "create_retry_decorator",
    "retry_on_failure",
    "retry_on_timeout",
    "retry_on_rate_limit",
]
