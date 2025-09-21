"""Tests for circuit breaker functionality."""
import os
import sys
import threading
import time

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_breaker,
    get_circuit_breaker_registry,
    setup_default_circuit_breakers,
)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0
        assert config.expected_exceptions == (Exception,)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=15.0,
        )
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.success_threshold == 2
        assert config.timeout == 15.0


class TestCircuitBreaker:
    """Test circuit breaker implementation."""

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test", config)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0

    def test_successful_call(self):
        """Test successful function call."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test", config)

        def successful_function():
            return "success"

        result = breaker.call(successful_function)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_failing_call(self):
        """Test failing function call."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config)

        def failing_function():
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            breaker.call(failing_function)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 1

        # Second failure should open the circuit
        with pytest.raises(ValueError):
            breaker.call(failing_function)
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 2

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        def failing_function():
            raise RuntimeError("Service unavailable")

        # Fail up to threshold
        for i in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

    def test_open_circuit_blocks_calls(self):
        """Test open circuit blocks function calls."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        def failing_function():
            raise RuntimeError("Service down")

        # Trigger circuit opening
        with pytest.raises(RuntimeError):
            breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

        # Now calls should be blocked
        def any_function():
            return "should not execute"

        with pytest.raises(CircuitBreakerError):
            breaker.call(any_function)

    def test_half_open_state(self):
        """Test circuit moves to half-open after recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.1  # Very short for testing
        )
        breaker = CircuitBreaker("test", config)

        def failing_function():
            raise RuntimeError("Service down")

        # Open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(failing_function)
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.2)

        # Force state update by attempting a call
        def test_function():
            return "recovery test"

        # First call in half-open should work
        # Wait a bit to avoid the 1-second throttle in half-open state
        time.sleep(1.1)
        result = breaker.call(test_function)
        assert result == "recovery test"

    def test_half_open_to_closed(self):
        """Test circuit moves from half-open to closed after successful calls."""
        config = CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.1, success_threshold=2
        )
        breaker = CircuitBreaker("test", config)

        def failing_function():
            raise RuntimeError("Service down")

        def working_function():
            return "working"

        # Open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(failing_function)

        # Wait for recovery
        time.sleep(0.2)

        # Successful calls should close circuit
        # Wait to avoid throttle
        time.sleep(1.1)
        breaker.call(working_function)  # First success
        time.sleep(1.1)
        breaker.call(working_function)  # Second success, should close

        assert breaker.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Test circuit moves from half-open back to open on failure."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = CircuitBreaker("test", config)

        def failing_function():
            raise RuntimeError("Still failing")

        # Open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(failing_function)

        # Wait for recovery
        time.sleep(0.2)

        # Failure in half-open should reopen circuit
        # Wait to avoid throttle
        time.sleep(1.1)
        with pytest.raises(RuntimeError):
            breaker.call(failing_function)

        assert breaker.state == CircuitState.OPEN

    def test_get_stats(self):
        """Test getting circuit breaker statistics."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test", config)

        stats = breaker.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == CircuitState.CLOSED.value
        assert stats["failure_count"] == 0
        assert "config" in stats

    def test_reset(self):
        """Test resetting circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        def failing_function():
            raise RuntimeError("Error")

        # Open circuit
        with pytest.raises(RuntimeError):
            breaker.call(failing_function)
        assert breaker.state == CircuitState.OPEN

        # Reset should close circuit
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_force_open(self):
        """Test forcing circuit breaker open."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test", config)

        assert breaker.state == CircuitState.CLOSED

        breaker.force_open()
        assert breaker.state == CircuitState.OPEN

        # Calls should be blocked
        def test_function():
            return "blocked"

        with pytest.raises(CircuitBreakerError):
            breaker.call(test_function)

    def test_thread_safety(self):
        """Test circuit breaker is thread safe."""
        config = CircuitBreakerConfig(failure_threshold=10)
        breaker = CircuitBreaker("test", config)

        results = []
        errors = []

        def worker(worker_id):
            def test_function():
                return f"worker_{worker_id}"

            try:
                result = breaker.call(test_function)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Multiple threads calling simultaneously
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All calls should succeed (circuit starts closed)
        assert len(results) == 10
        assert len(errors) == 0


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry."""

    def test_registry_initialization(self):
        """Test registry initializes empty."""
        registry = CircuitBreakerRegistry()
        assert len(registry._breakers) == 0

    def test_get_breaker(self):
        """Test getting circuit breaker from registry."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig()

        breaker = registry.get_breaker("test_service", config)
        assert breaker.name == "test_service"
        assert "test_service" in registry._breakers

    def test_get_breaker_default_config(self):
        """Test getting circuit breaker with default config."""
        registry = CircuitBreakerRegistry()

        breaker = registry.get_breaker("test_service")
        assert breaker.name == "test_service"

    def test_remove_breaker(self):
        """Test removing circuit breaker."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig()

        registry.get_breaker("test_service", config)
        assert registry.remove_breaker("test_service") is True
        assert "test_service" not in registry._breakers

        # Removing non-existent breaker should return False
        assert registry.remove_breaker("nonexistent") is False

    def test_get_all_stats(self):
        """Test getting all circuit breaker stats."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig()

        registry.get_breaker("service1", config)
        registry.get_breaker("service2", config)

        stats = registry.get_all_stats()
        assert "service1" in stats
        assert "service2" in stats

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=1)

        breaker1 = registry.get_breaker("service1", config)
        breaker2 = registry.get_breaker("service2", config)

        # Open both circuits
        def failing_function():
            raise RuntimeError("Error")

        with pytest.raises(RuntimeError):
            breaker1.call(failing_function)
        with pytest.raises(RuntimeError):
            breaker2.call(failing_function)

        assert breaker1.state == CircuitState.OPEN
        assert breaker2.state == CircuitState.OPEN

        # Reset all
        registry.reset_all()

        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED

    def test_get_failing_breakers(self):
        """Test getting failing circuit breakers."""
        registry = CircuitBreakerRegistry()
        config = CircuitBreakerConfig(failure_threshold=1)

        breaker1 = registry.get_breaker("service1", config)
        registry.get_breaker("service2", config)

        # Open one circuit
        def failing_function():
            raise RuntimeError("Error")

        with pytest.raises(RuntimeError):
            breaker1.call(failing_function)

        failing = registry.get_failing_breakers()
        assert "service1" in failing
        assert "service2" not in failing


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator."""

    def test_decorator_basic(self):
        """Test basic decorator functionality."""

        @circuit_breaker("test_service", failure_threshold=2)
        def test_function(should_fail=False):
            if should_fail:
                raise RuntimeError("Service error")
            return "success"

        # Successful calls should work
        assert test_function() == "success"
        assert test_function() == "success"

        # Failures should eventually open circuit
        with pytest.raises(RuntimeError):
            test_function(should_fail=True)
        with pytest.raises(RuntimeError):
            test_function(should_fail=True)

        # Now circuit should be open
        with pytest.raises(CircuitBreakerError):
            test_function()

    def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""

        @circuit_breaker("metadata_test")
        def test_function():
            """Test docstring."""
            return "test"

        assert test_function.__name__ == "test_function"
        assert test_function.__doc__ == "Test docstring."

    def test_decorator_access_to_breaker(self):
        """Test decorator provides access to underlying circuit breaker."""

        @circuit_breaker("breaker_access_test")
        def test_function():
            return "test"

        assert hasattr(test_function, "_circuit_breaker")
        assert test_function._circuit_breaker.name == "breaker_access_test"


class TestDefaultCircuitBreakers:
    """Test default circuit breakers setup."""

    def test_setup_default_circuit_breakers(self):
        """Test setting up default circuit breakers."""
        # This should not raise an exception
        setup_default_circuit_breakers()

        registry = get_circuit_breaker_registry()
        stats = registry.get_all_stats()

        # Check that default breakers were created
        expected_breakers = [
            "claude_code_cli",
            "openai_api",
            "anthropic_api",
            "gemini_api",
        ]

        for breaker_name in expected_breakers:
            assert breaker_name in stats


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker."""

    def test_realistic_service_failure_scenario(self):
        """Test realistic service failure and recovery scenario."""
        config = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=0.1, success_threshold=2
        )
        breaker = CircuitBreaker("integration_test", config)

        # Simulate service going down
        def unreliable_service(fail_count=[0]):
            fail_count[0] += 1
            if fail_count[0] <= 5:  # Fail first 5 calls
                raise RuntimeError("Service temporarily unavailable")
            return "Service recovered"

        # Initial failures should open circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(unreliable_service)

        assert breaker.state == CircuitState.OPEN

        # Calls should be blocked while circuit is open
        with pytest.raises(CircuitBreakerError):
            breaker.call(unreliable_service)

        # Wait for recovery timeout
        time.sleep(0.2)

        # Service should be working now, circuit should close
        result1 = breaker.call(unreliable_service)
        result2 = breaker.call(unreliable_service)

        assert result1 == "Service recovered"
        assert result2 == "Service recovered"
        assert breaker.state == CircuitState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
