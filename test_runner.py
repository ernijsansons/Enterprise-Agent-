"""Simple test runner for validating our implementations."""
import sys
import traceback

# Add src to path
sys.path.insert(0, "src")


def run_test(test_name, test_func):
    """Run a single test and report results."""
    try:
        print(f"Running {test_name}...", end=" ")
        test_func()
        print("PASSED")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        print(f"  {traceback.format_exc()}")
        return False


def test_rate_limiter_basic():
    """Test basic rate limiter functionality."""
    from src.utils.rate_limiter import RateLimitConfig, RateLimiter, TokenBucket

    # Test token bucket
    config = RateLimitConfig(max_tokens=5, refill_rate=2.0)
    bucket = TokenBucket(config)

    assert bucket.acquire(3) is True, "Should acquire 3 tokens"
    assert bucket.acquire(3) is False, "Should not acquire 3 more tokens"
    assert bucket.acquire(2) is True, "Should acquire remaining 2 tokens"

    # Test rate limiter
    limiter = RateLimiter()
    limiter.add_limit("test", config)

    assert limiter.acquire("test", 2) is True, "Should acquire tokens"
    status = limiter.get_status("test")
    assert "tokens_available" in status, "Should have status"


def test_circuit_breaker_basic():
    """Test basic circuit breaker functionality."""
    from src.utils.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerError,
        CircuitState,
    )

    config = CircuitBreakerConfig(failure_threshold=2)
    breaker = CircuitBreaker("test", config)

    # Test successful call
    def success_func():
        return "ok"

    result = breaker.call(success_func)
    assert result == "ok", "Should execute successfully"
    assert breaker.state == CircuitState.CLOSED, "Should remain closed"

    # Test failure
    def fail_func():
        raise RuntimeError("test error")

    # First failure
    try:
        breaker.call(fail_func)
        assert False, "Should have raised exception"
    except RuntimeError:
        pass

    assert (
        breaker.state == CircuitState.CLOSED
    ), "Should still be closed after 1 failure"

    # Second failure should open circuit
    try:
        breaker.call(fail_func)
        assert False, "Should have raised exception"
    except RuntimeError:
        pass

    assert breaker.state == CircuitState.OPEN, "Should be open after 2 failures"

    # Now calls should be blocked
    try:
        breaker.call(success_func)
        assert False, "Should have raised CircuitBreakerError"
    except CircuitBreakerError:
        pass


def test_security_imports():
    """Test that security modules can be imported."""
    from src.exceptions import RateLimitExceeded
    from src.providers.auth_manager import ClaudeAuthManager
    from src.providers.claude_code_provider import ClaudeCodeProvider

    # Just test imports work
    assert ClaudeCodeProvider is not None
    assert ClaudeAuthManager is not None
    assert RateLimitExceeded is not None


def test_provider_resilience_setup():
    """Test that provider can setup resilience patterns."""
    from unittest.mock import patch

    from src.providers.claude_code_provider import ClaudeCodeProvider

    # Mock the CLI verification to avoid actual CLI calls
    with patch.object(
        ClaudeCodeProvider, "verify_cli_available", return_value=True
    ), patch.object(
        ClaudeCodeProvider, "verify_subscription_auth", return_value=True
    ), patch.object(
        ClaudeCodeProvider, "_load_persistent_sessions"
    ):
        provider = ClaudeCodeProvider()

        # Check that resilience patterns were setup
        assert hasattr(provider, "circuit_breaker"), "Should have circuit breaker"

        # Test status method
        status = provider.get_resilience_status()
        assert "rate_limiter" in status, "Should have rate limiter status"
        assert "circuit_breaker" in status, "Should have circuit breaker status"


def test_config_loading():
    """Test that configuration files exist."""
    from pathlib import Path

    # Test rate limits config exists
    config_path = Path("configs/rate_limits.yaml")
    assert config_path.exists(), "Rate limits config should exist"

    # Test Claude Code config exists
    claude_config_path = Path("configs/claude_code_config.yaml")
    assert claude_config_path.exists(), "Claude Code config should exist"

    # Just verify they're readable text files
    with open(config_path) as f:
        content = f.read()
        assert "providers:" in content, "Should have providers section"
        assert "claude_code:" in content, "Should have Claude Code config"


def main():
    """Run all tests."""
    print("Running Enterprise Agent Phase 1 Tests")
    print("=" * 50)

    tests = [
        ("Rate Limiter Basic", test_rate_limiter_basic),
        ("Circuit Breaker Basic", test_circuit_breaker_basic),
        ("Security Imports", test_security_imports),
        ("Provider Resilience Setup", test_provider_resilience_setup),
        ("Config Loading", test_config_loading),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed! Phase 1 implementation is working.")
        return 0
    else:
        print("Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
