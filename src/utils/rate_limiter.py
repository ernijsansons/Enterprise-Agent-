"""Rate limiter implementation using token bucket algorithm."""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    max_tokens: int = 100  # Maximum tokens in bucket
    refill_rate: float = 10.0  # Tokens per second
    burst_allowance: int = 20  # Extra tokens for burst traffic
    window_seconds: int = 60  # Time window for rate limiting


class TokenBucket:
    """Thread-safe token bucket for rate limiting."""

    def __init__(self, config: RateLimitConfig):
        """Initialize token bucket.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.tokens = float(config.max_tokens)
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False if rate limited
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add tokens based on refill rate
            self.tokens = min(
                self.config.max_tokens + self.config.burst_allowance,
                self.tokens + elapsed * self.config.refill_rate,
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False

    def tokens_available(self) -> float:
        """Get current number of tokens available.

        Returns:
            Current token count
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            return min(
                self.config.max_tokens + self.config.burst_allowance,
                self.tokens + elapsed * self.config.refill_rate,
            )

    def time_until_tokens(self, tokens: int = 1) -> float:
        """Calculate time until specified tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Time in seconds until tokens are available
        """
        available = self.tokens_available()
        if available >= tokens:
            return 0.0

        needed = tokens - available
        return needed / self.config.refill_rate


class RateLimiter:
    """Multi-key rate limiter with different limits per key."""

    def __init__(self):
        """Initialize rate limiter."""
        self._buckets: Dict[str, TokenBucket] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        self._lock = threading.Lock()

    def add_limit(self, key: str, config: RateLimitConfig) -> None:
        """Add a rate limit for a specific key.

        Args:
            key: Unique identifier for the rate limit
            config: Rate limit configuration
        """
        with self._lock:
            self._configs[key] = config
            self._buckets[key] = TokenBucket(config)

    def acquire(self, key: str, tokens: int = 1) -> bool:
        """Attempt to acquire tokens for a specific key.

        Args:
            key: Rate limit key
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False if rate limited
        """
        with self._lock:
            if key not in self._buckets:
                # No limit configured, allow by default
                return True

            bucket = self._buckets[key]
            acquired = bucket.acquire(tokens)

            if not acquired:
                logger.warning(
                    f"Rate limit exceeded for key '{key}', "
                    f"requested {tokens} tokens, "
                    f"available: {bucket.tokens_available():.1f}"
                )

            return acquired

    def get_status(self, key: str) -> Dict[str, float]:
        """Get current rate limit status for a key.

        Args:
            key: Rate limit key

        Returns:
            Dictionary with rate limit status
        """
        with self._lock:
            if key not in self._buckets:
                return {"error": "No rate limit configured for key"}

            bucket = self._buckets[key]
            config = self._configs[key]

            return {
                "key": key,
                "tokens_available": bucket.tokens_available(),
                "max_tokens": config.max_tokens,
                "refill_rate": config.refill_rate,
                "time_until_token": bucket.time_until_tokens(1),
                "burst_allowance": config.burst_allowance,
            }

    def reset(self, key: str) -> bool:
        """Reset rate limit for a specific key.

        Args:
            key: Rate limit key

        Returns:
            True if reset successful, False if key not found
        """
        with self._lock:
            if key not in self._buckets:
                return False

            config = self._configs[key]
            self._buckets[key] = TokenBucket(config)
            logger.info(f"Reset rate limit for key '{key}'")
            return True

    def remove_limit(self, key: str) -> bool:
        """Remove rate limit for a specific key.

        Args:
            key: Rate limit key

        Returns:
            True if removed, False if key not found
        """
        with self._lock:
            if key not in self._buckets:
                return False

            del self._buckets[key]
            del self._configs[key]
            logger.info(f"Removed rate limit for key '{key}'")
            return True

    def get_all_status(self) -> Dict[str, Dict[str, float]]:
        """Get status for all configured rate limits.

        Returns:
            Dictionary with status for all keys
        """
        return {key: self.get_status(key) for key in self._buckets.keys()}


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts limits based on success/failure rates."""

    def __init__(self):
        """Initialize adaptive rate limiter."""
        super().__init__()
        self._success_counts: Dict[str, int] = {}
        self._failure_counts: Dict[str, int] = {}
        self._last_adaptation: Dict[str, float] = {}

    def record_success(self, key: str) -> None:
        """Record a successful operation.

        Args:
            key: Rate limit key
        """
        with self._lock:
            self._success_counts[key] = self._success_counts.get(key, 0) + 1
            self._adapt_if_needed(key)

    def record_failure(self, key: str) -> None:
        """Record a failed operation.

        Args:
            key: Rate limit key
        """
        with self._lock:
            self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
            self._adapt_if_needed(key)

    def _adapt_if_needed(self, key: str) -> None:
        """Adapt rate limits based on success/failure ratio.

        Args:
            key: Rate limit key
        """
        now = time.time()
        last_adapt = self._last_adaptation.get(key, 0)

        # Only adapt every 60 seconds
        if now - last_adapt < 60:
            return

        if key not in self._buckets:
            return

        successes = self._success_counts.get(key, 0)
        failures = self._failure_counts.get(key, 0)
        total = successes + failures

        if total < 10:  # Need minimum samples
            return

        success_rate = successes / total
        config = self._configs[key]

        # Adapt refill rate based on success rate
        if success_rate > 0.9:  # High success rate, increase limit
            new_rate = min(config.refill_rate * 1.2, config.refill_rate * 2)
            logger.info(
                f"Increasing rate limit for '{key}': {config.refill_rate} -> {new_rate}"
            )
            config.refill_rate = new_rate
        elif success_rate < 0.5:  # Low success rate, decrease limit
            new_rate = max(config.refill_rate * 0.8, config.refill_rate * 0.5)
            logger.info(
                f"Decreasing rate limit for '{key}': {config.refill_rate} -> {new_rate}"
            )
            config.refill_rate = new_rate

        # Reset counters and update timestamp
        self._success_counts[key] = 0
        self._failure_counts[key] = 0
        self._last_adaptation[key] = now

        # Create new bucket with updated config
        self._buckets[key] = TokenBucket(config)


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance.

    Returns:
        Global rate limiter
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = AdaptiveRateLimiter()
    return _global_rate_limiter


def setup_default_limits() -> None:
    """Setup default rate limits for common operations."""
    limiter = get_rate_limiter()

    # Claude Code CLI limits (conservative based on Max plan)
    limiter.add_limit(
        "claude_code",
        RateLimitConfig(
            max_tokens=20,
            refill_rate=2.0,  # 2 requests per second
            burst_allowance=10,
            window_seconds=60,
        ),
    )

    # OpenAI API limits (more restrictive to save costs)
    limiter.add_limit(
        "openai_api",
        RateLimitConfig(
            max_tokens=10,
            refill_rate=1.0,  # 1 request per second
            burst_allowance=5,
            window_seconds=60,
        ),
    )

    # Anthropic API limits (backup only)
    limiter.add_limit(
        "anthropic_api",
        RateLimitConfig(
            max_tokens=5,
            refill_rate=0.5,  # 0.5 requests per second
            burst_allowance=2,
            window_seconds=60,
        ),
    )

    # Gemini API limits
    limiter.add_limit(
        "gemini_api",
        RateLimitConfig(
            max_tokens=15,
            refill_rate=1.5,  # 1.5 requests per second
            burst_allowance=5,
            window_seconds=60,
        ),
    )

    logger.info("Default rate limits configured")


def rate_limited(key: str, tokens: int = 1):
    """Decorator for rate limiting function calls.

    Args:
        key: Rate limit key
        tokens: Number of tokens to consume

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()

            if not limiter.acquire(key, tokens):
                from src.exceptions import RateLimitExceeded

                status = limiter.get_status(key)
                wait_time = status.get("time_until_token", 1.0)

                raise RateLimitExceeded(
                    f"Rate limit exceeded for '{key}'. "
                    f"Wait {wait_time:.1f} seconds before retrying.",
                    retry_after=wait_time,
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "RateLimitConfig",
    "TokenBucket",
    "RateLimiter",
    "AdaptiveRateLimiter",
    "get_rate_limiter",
    "setup_default_limits",
    "rate_limited",
]
