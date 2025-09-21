"""Tests for rate limiter functionality."""
import pytest
import time
import threading
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.rate_limiter import (
    RateLimitConfig,
    TokenBucket,
    RateLimiter,
    AdaptiveRateLimiter,
    get_rate_limiter,
    setup_default_limits,
    rate_limited,
)
from src.exceptions import RateLimitExceeded


class TestTokenBucket:
    """Test token bucket implementation."""

    def test_token_bucket_initialization(self):
        """Test token bucket initializes correctly."""
        config = RateLimitConfig(max_tokens=10, refill_rate=5.0)
        bucket = TokenBucket(config)

        assert bucket.tokens == 10.0
        assert bucket.config.max_tokens == 10
        assert bucket.config.refill_rate == 5.0

    def test_token_acquisition(self):
        """Test token acquisition works correctly."""
        config = RateLimitConfig(max_tokens=10, refill_rate=5.0)
        bucket = TokenBucket(config)

        # Should be able to acquire tokens
        assert bucket.acquire(5) == True
        assert bucket.tokens == 5.0

        # Should be able to acquire remaining tokens
        assert bucket.acquire(5) == True
        assert bucket.tokens == 0.0

        # Should not be able to acquire more tokens
        assert bucket.acquire(1) == False

    def test_token_refill(self):
        """Test token refill over time."""
        config = RateLimitConfig(max_tokens=10, refill_rate=10.0)  # 10 tokens per second
        bucket = TokenBucket(config)

        # Use all tokens
        bucket.acquire(10)
        assert bucket.tokens == 0.0

        # Wait and check refill (simulate 0.5 seconds)
        bucket.last_update = time.time() - 0.5
        available = bucket.tokens_available()
        assert available >= 4.0  # Should have refilled ~5 tokens

    def test_burst_allowance(self):
        """Test burst allowance allows temporary exceeding of max tokens."""
        config = RateLimitConfig(max_tokens=10, refill_rate=5.0, burst_allowance=5)
        bucket = TokenBucket(config)

        # Should start with max + burst tokens available
        assert bucket.tokens_available() >= 10.0


class TestRateLimiter:
    """Test rate limiter implementation."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter()
        assert len(limiter._buckets) == 0

    def test_add_limit(self):
        """Test adding rate limits."""
        limiter = RateLimiter()
        config = RateLimitConfig(max_tokens=5, refill_rate=2.0)

        limiter.add_limit("test_key", config)
        assert "test_key" in limiter._buckets
        assert "test_key" in limiter._configs

    def test_acquire_with_limit(self):
        """Test acquiring tokens with rate limit."""
        limiter = RateLimiter()
        config = RateLimitConfig(max_tokens=5, refill_rate=2.0)
        limiter.add_limit("test_key", config)

        # Should be able to acquire tokens
        assert limiter.acquire("test_key", 3) == True
        assert limiter.acquire("test_key", 2) == True

        # Should be rate limited now
        assert limiter.acquire("test_key", 1) == False

    def test_acquire_without_limit(self):
        """Test acquiring tokens without configured limit."""
        limiter = RateLimiter()

        # Should allow by default
        assert limiter.acquire("unknown_key", 100) == True

    def test_get_status(self):
        """Test getting rate limit status."""
        limiter = RateLimiter()
        config = RateLimitConfig(max_tokens=10, refill_rate=5.0)
        limiter.add_limit("test_key", config)

        status = limiter.get_status("test_key")
        assert "tokens_available" in status
        assert "max_tokens" in status
        assert status["max_tokens"] == 10
        assert status["refill_rate"] == 5.0

    def test_reset_limit(self):
        """Test resetting rate limit."""
        limiter = RateLimiter()
        config = RateLimitConfig(max_tokens=5, refill_rate=2.0)
        limiter.add_limit("test_key", config)

        # Use up tokens
        limiter.acquire("test_key", 5)

        # Reset should restore tokens
        assert limiter.reset("test_key") == True
        assert limiter.acquire("test_key", 5) == True

    def test_thread_safety(self):
        """Test rate limiter is thread safe."""
        limiter = RateLimiter()
        config = RateLimitConfig(max_tokens=100, refill_rate=10.0)
        limiter.add_limit("test_key", config)

        results = []

        def worker():
            for _ in range(10):
                result = limiter.acquire("test_key", 1)
                results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have some successful acquisitions
        successful = sum(results)
        assert successful > 0
        assert successful <= 100  # Should not exceed bucket capacity


class TestAdaptiveRateLimiter:
    """Test adaptive rate limiter."""

    def test_adaptive_initialization(self):
        """Test adaptive rate limiter initializes correctly."""
        limiter = AdaptiveRateLimiter()
        assert hasattr(limiter, '_success_counts')
        assert hasattr(limiter, '_failure_counts')

    def test_record_success_failure(self):
        """Test recording success and failure."""
        limiter = AdaptiveRateLimiter()
        config = RateLimitConfig(max_tokens=10, refill_rate=5.0)
        limiter.add_limit("test_key", config)

        limiter.record_success("test_key")
        limiter.record_failure("test_key")

        assert limiter._success_counts.get("test_key", 0) == 1
        assert limiter._failure_counts.get("test_key", 0) == 1


class TestRateLimitedDecorator:
    """Test rate limited decorator."""

    def test_decorator_basic(self):
        """Test basic decorator functionality."""
        limiter = get_rate_limiter()
        config = RateLimitConfig(max_tokens=2, refill_rate=1.0)
        limiter.add_limit("test_func", config)

        @rate_limited("test_func", tokens=1)
        def test_function():
            return "success"

        # Should work for first two calls
        assert test_function() == "success"
        assert test_function() == "success"

        # Third call should be rate limited
        with pytest.raises(RateLimitExceeded):
            test_function()

    def test_decorator_retry_after(self):
        """Test decorator includes retry_after in exception."""
        limiter = get_rate_limiter()
        limiter.remove_limit("test_func_2")  # Clean up if exists

        config = RateLimitConfig(max_tokens=1, refill_rate=1.0)
        limiter.add_limit("test_func_2", config)

        @rate_limited("test_func_2", tokens=1)
        def test_function():
            return "success"

        # Use up the token
        test_function()

        # Next call should be rate limited with retry_after
        with pytest.raises(RateLimitExceeded) as exc_info:
            test_function()

        assert exc_info.value.retry_after is not None
        assert exc_info.value.retry_after > 0


class TestDefaultLimits:
    """Test default rate limits setup."""

    def test_setup_default_limits(self):
        """Test setting up default limits."""
        # This should not raise an exception
        setup_default_limits()

        limiter = get_rate_limiter()

        # Check that default limits were created
        claude_status = limiter.get_status("claude_code")
        assert "tokens_available" in claude_status

        openai_status = limiter.get_status("openai_api")
        assert "tokens_available" in openai_status


class TestIntegration:
    """Integration tests for rate limiting."""

    def test_realistic_usage_pattern(self):
        """Test realistic usage pattern."""
        limiter = RateLimiter()
        config = RateLimitConfig(max_tokens=10, refill_rate=5.0, burst_allowance=5)
        limiter.add_limit("api_calls", config)

        # Burst of calls should work initially
        successful_calls = 0
        for _ in range(15):  # Try more than bucket capacity
            if limiter.acquire("api_calls", 1):
                successful_calls += 1

        # Should handle some burst but not all
        assert successful_calls > 10
        assert successful_calls <= 15

    def test_rate_limit_recovery(self):
        """Test that rate limits recover over time."""
        limiter = RateLimiter()
        config = RateLimitConfig(max_tokens=5, refill_rate=10.0)  # Fast refill for testing
        limiter.add_limit("recovery_test", config)

        # Use all tokens
        assert limiter.acquire("recovery_test", 5) == True
        assert limiter.acquire("recovery_test", 1) == False

        # Simulate time passing (0.1 second should refill ~1 token)
        bucket = limiter._buckets["recovery_test"]
        bucket.last_update = time.time() - 0.1

        # Should be able to acquire a token now
        assert limiter.acquire("recovery_test", 1) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])