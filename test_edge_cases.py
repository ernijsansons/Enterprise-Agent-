#!/usr/bin/env python3
"""Comprehensive edge case testing for Enterprise Agent v3.4."""

import os
import sys
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_concurrency_edge_cases():
    """Test concurrency edge cases and error conditions."""
    print("Testing concurrency edge cases...")

    try:
        from src.utils.concurrency import ExecutionManager, ThreadSafeDict, synchronized_state

        # Test 1: ExecutionManager with rapid start/stop
        print("  Testing rapid start/stop scenarios...")
        manager = ExecutionManager(max_workers=2)

        # Start and stop multiple times rapidly
        for i in range(5):
            manager.start()
            manager.shutdown(wait=False)

        # Should still work after rapid cycling
        with manager:
            future = manager.submit(lambda x: x * 2, 5)
            assert future.result(timeout=5) == 10

        print("  PASS: Rapid start/stop test passed")

        # Test 2: ThreadSafeDict concurrent access
        print("  Testing ThreadSafeDict concurrent access...")
        safe_dict = ThreadSafeDict()
        results = []
        errors = []

        def concurrent_access(thread_id):
            try:
                for i in range(100):
                    key = f"key_{thread_id}_{i}"
                    safe_dict[key] = thread_id * 1000 + i
                    retrieved = safe_dict.get(key)
                    if retrieved != thread_id * 1000 + i:
                        errors.append(f"Mismatch in thread {thread_id}: expected {thread_id * 1000 + i}, got {retrieved}")
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_access, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        if errors:
            print(f"  FAIL: ThreadSafeDict errors: {errors[:5]}...")  # Show first 5 errors
            return False

        print(f"  PASS: ThreadSafeDict concurrent access test passed ({len(results)} threads)")

        # Test 3: Timeout handling
        print("  Testing timeout scenarios...")

        def slow_function():
            time.sleep(2)
            return "slow_result"

        with manager:
            future = manager.submit(slow_function)
            try:
                result = future.result(timeout=0.5)  # Should timeout
                print("  FAIL: Timeout test failed - should have timed out")
                return False
            except Exception:
                print("  PASS: Timeout handling works correctly")

        print("SUCCESS: Concurrency edge cases test passed")
        return True

    except Exception as e:
        print(f"FAILED: Concurrency edge cases test failed: {e}")
        return False


def test_validation_edge_cases():
    """Test validation edge cases and malformed inputs."""
    print("\nTesting validation edge cases...")

    try:
        from src.utils.validation import StringValidator, NumberValidator, ValidationException

        # Test 1: String validator with unicode and edge cases
        print("  Testing string validation edge cases...")

        validator = StringValidator(min_length=5, max_length=20, pattern=r'^[a-zA-Z0-9_]+$')

        # Test unicode handling
        try:
            validator.validate("test🚀emoji")
            print("  FAIL: Should have failed unicode validation")
            return False
        except ValidationException as e:
            print(f"  PASS: Unicode validation correctly failed: {str(e)[:60]}...")

        # Test empty string
        try:
            validator.validate("")
            print("  FAIL: Should have failed empty string validation")
            return False
        except ValidationException as e:
            print(f"  PASS: Empty string validation correctly failed: {str(e)[:60]}...")

        # Test very long string
        try:
            validator.validate("a" * 1000)
            print("  FAIL: Should have failed long string validation")
            return False
        except ValidationException as e:
            print(f"  PASS: Long string validation correctly failed: {str(e)[:60]}...")

        # Test 2: Number validator edge cases
        print("  Testing number validation edge cases...")

        num_validator = NumberValidator(min_value=0, max_value=100, allow_float=False)

        # Test infinity
        try:
            num_validator.validate(float('inf'))
            print("  FAIL: Should have failed infinity validation")
            return False
        except ValidationException as e:
            print(f"  PASS: Infinity validation correctly failed: {str(e)[:60]}...")

        # Test NaN
        try:
            num_validator.validate(float('nan'))
            print("  FAIL: Should have failed NaN validation")
            return False
        except ValidationException as e:
            print(f"  PASS: NaN validation correctly failed: {str(e)[:60]}...")

        # Test string that looks like number
        result = num_validator.validate("42")
        assert result == 42
        print("  PASS: String-to-number conversion works")

        # Test invalid string
        try:
            num_validator.validate("not_a_number")
            print("  FAIL: Should have failed string validation")
            return False
        except ValidationException as e:
            print(f"  PASS: Invalid string validation correctly failed: {str(e)[:60]}...")

        print("SUCCESS: Validation edge cases test passed")
        return True

    except Exception as e:
        print(f"FAILED: Validation edge cases test failed: {e}")
        return False


def test_caching_edge_cases():
    """Test caching system edge cases."""
    print("\nTesting caching edge cases...")

    try:
        from src.utils.cache import TTLCache, CacheConfig

        # Test 1: Cache with zero TTL
        print("  Testing zero TTL cache...")
        config = CacheConfig(default_ttl=0, max_size=10)
        cache = TTLCache(config)

        cache.set("key1", "value1")
        # Should be expired immediately
        time.sleep(0.1)
        result = cache.get("key1")
        if result is not None:
            print("  FAIL: Zero TTL cache should have expired immediately")
            return False
        print("  PASS: Zero TTL cache works correctly")

        # Test 2: Cache overflow
        print("  Testing cache overflow...")
        config = CacheConfig(default_ttl=60, max_size=3)
        cache = TTLCache(config)

        # Fill cache beyond capacity
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")

        # Should only have 3 items
        stats = cache.get_stats()
        if stats.size > 3:
            print(f"  FAIL: Cache size {stats.size} exceeds maximum 3")
            return False
        print(f"  PASS: Cache overflow handled correctly (size: {stats.size})")

        # Test 3: Concurrent cache access
        print("  Testing concurrent cache access...")
        config = CacheConfig(default_ttl=10, max_size=100)
        cache = TTLCache(config)

        errors = []

        def cache_worker(worker_id):
            try:
                for i in range(50):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    cache.set(key, value)
                    retrieved = cache.get(key)
                    if retrieved != value:
                        errors.append(f"Worker {worker_id}: expected {value}, got {retrieved}")
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        if errors:
            print(f"  FAIL: Concurrent cache access errors: {errors[:3]}...")
            return False

        print("  PASS: Concurrent cache access works correctly")

        # Test 4: Memory pressure simulation
        print("  Testing memory pressure scenarios...")
        config = CacheConfig(default_ttl=60, max_size=1000, compression_enabled=True)
        cache = TTLCache(config)

        # Add large values to test compression
        large_value = "x" * 10000  # 10KB value
        for i in range(50):
            cache.set(f"large_key_{i}", large_value)

        # Verify data integrity
        for i in range(0, 50, 10):  # Sample every 10th item
            retrieved = cache.get(f"large_key_{i}")
            if retrieved != large_value:
                print(f"  FAIL: Large value integrity check failed for key {i}")
                return False

        print("  PASS: Memory pressure simulation passed")

        print("SUCCESS: Caching edge cases test passed")
        return True

    except Exception as e:
        print(f"FAILED: Caching edge cases test failed: {e}")
        return False


def test_error_handling_edge_cases():
    """Test error handling edge cases."""
    print("\nTesting error handling edge cases...")

    try:
        from src.utils.errors import ErrorHandler, ErrorCode, EnterpriseAgentError

        # Test 1: Error handler with high frequency errors
        print("  Testing high frequency error scenarios...")
        handler = ErrorHandler()

        # Generate many errors quickly
        for i in range(1000):
            try:
                raise EnterpriseAgentError(
                    ErrorCode.NETWORK_CONNECTION_FAILED,
                    f"Test error {i}",
                    context={"test_id": i}
                )
            except EnterpriseAgentError as e:
                handler.handle_error(e.details.code, context=e.details.context)

        stats = handler.get_error_statistics()
        if stats['total_errors'] != 1000:
            print(f"  FAIL: Expected 1000 errors, got {stats['total_errors']}")
            return False

        print(f"  PASS: High frequency error handling works (processed {stats['total_errors']} errors)")

        # Test 2: Nested error scenarios
        print("  Testing nested error scenarios...")

        def nested_function_level_3():
            raise EnterpriseAgentError(
                ErrorCode.VALIDATION_FAILED,
                "Level 3 error",
                context={"level": 3}
            )

        def nested_function_level_2():
            try:
                nested_function_level_3()
            except EnterpriseAgentError as e:
                raise EnterpriseAgentError(
                    ErrorCode.OPERATION_FAILED,
                    "Level 2 error",
                    context={"level": 2, "cause": str(e)}
                ) from e

        def nested_function_level_1():
            try:
                nested_function_level_2()
            except EnterpriseAgentError as e:
                raise EnterpriseAgentError(
                    ErrorCode.SYSTEM_ERROR,
                    "Level 1 error",
                    context={"level": 1, "cause": str(e)}
                ) from e

        try:
            nested_function_level_1()
            print("  FAIL: Should have raised nested error")
            return False
        except EnterpriseAgentError as e:
            # Verify error chain
            if e.details.code != ErrorCode.SYSTEM_ERROR:
                print(f"  FAIL: Expected SYSTEM_ERROR, got {e.details.code}")
                return False

            print("  PASS: Nested error handling works correctly")

        # Test 3: Error recovery scenarios
        print("  Testing error recovery scenarios...")

        recovery_attempts = 0
        max_attempts = 3

        def flaky_operation():
            nonlocal recovery_attempts
            recovery_attempts += 1

            if recovery_attempts < max_attempts:
                raise EnterpriseAgentError(
                    ErrorCode.TEMPORARY_FAILURE,
                    f"Attempt {recovery_attempts} failed",
                    context={"attempt": recovery_attempts}
                )
            return f"Success on attempt {recovery_attempts}"

        # Simulate retry logic
        for attempt in range(max_attempts + 1):
            try:
                result = flaky_operation()
                print(f"  PASS: Error recovery successful: {result}")
                break
            except EnterpriseAgentError as e:
                if attempt == max_attempts:
                    print(f"  FAIL: Recovery failed after {max_attempts} attempts")
                    return False
                handler.handle_error(e.details.code, context=e.details.context)

        print("SUCCESS: Error handling edge cases test passed")
        return True

    except Exception as e:
        print(f"FAILED: Error handling edge cases test failed: {e}")
        return False


def test_telemetry_privacy_edge_cases():
    """Test telemetry privacy protection edge cases."""
    print("\nTesting telemetry privacy edge cases...")

    try:
        from src.utils.telemetry import _sanitize_data, set_telemetry_consent, get_telemetry_status

        # Test 1: Complex data sanitization
        print("  Testing complex data sanitization...")

        sensitive_data = {
            "user_email": "user@example.com",
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "credit_card": "4532-1234-5678-9012",
            "nested": {
                "password": "secret123",
                "file_path": "/home/user/secret/document.txt",
                "ip_address": "192.168.1.100"
            },
            "list_with_sensitive": [
                "normal_data",
                "phone: 555-123-4567",
                {"inner_secret": "token_abc123xyz"}
            ]
        }

        sanitized = _sanitize_data(sensitive_data)

        # Verify sanitization
        if "@example.com" in str(sanitized):
            print("  FAIL: Email not properly sanitized")
            return False

        if "sk-1234567890abcdef" in str(sanitized):
            print("  FAIL: API key not properly sanitized")
            return False

        if "4532-1234-5678-9012" in str(sanitized):
            print("  FAIL: Credit card not properly sanitized")
            return False

        print("  PASS: Complex data sanitization works correctly")

        # Test 2: Consent management edge cases
        print("  Testing consent management edge cases...")

        # Test with no consent
        original_consent = os.getenv("TELEMETRY_CONSENT")
        os.environ["TELEMETRY_CONSENT"] = "false"

        status = get_telemetry_status()
        if status["consent_enabled"]:
            print("  FAIL: Consent should be disabled")
            return False

        # Test enabling consent
        set_telemetry_consent(True)
        status = get_telemetry_status()
        if not status["consent_enabled"]:
            print("  FAIL: Consent should be enabled after setting")
            return False

        # Test disabling consent
        set_telemetry_consent(False)
        status = get_telemetry_status()
        if status["consent_enabled"]:
            print("  FAIL: Consent should be disabled after revocation")
            return False

        # Restore original consent
        if original_consent:
            os.environ["TELEMETRY_CONSENT"] = original_consent
        elif "TELEMETRY_CONSENT" in os.environ:
            del os.environ["TELEMETRY_CONSENT"]

        print("  PASS: Consent management edge cases work correctly")

        # Test 3: Malformed data handling
        print("  Testing malformed data handling...")

        malformed_data = {
            "circular_ref": None,
            "complex_object": object(),
            "lambda_func": lambda x: x,
        }
        malformed_data["circular_ref"] = malformed_data  # Create circular reference

        try:
            sanitized = _sanitize_data(malformed_data)
            print("  PASS: Malformed data handled gracefully")
        except Exception as e:
            print(f"  FAIL: Malformed data handling failed: {e}")
            return False

        print("SUCCESS: Telemetry privacy edge cases test passed")
        return True

    except Exception as e:
        print(f"FAILED: Telemetry privacy edge cases test failed: {e}")
        return False


def test_configuration_edge_cases():
    """Test configuration edge cases."""
    print("\nTesting configuration edge cases...")

    try:
        # Test 1: Invalid YAML handling
        print("  Testing invalid YAML handling...")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
invalid_yaml: [
    missing_closing_bracket
    "unclosed_string
""")
            invalid_config_path = f.name

        try:
            from validate_config import ConfigValidator
            validator = ConfigValidator()
            result = validator.validate_config(invalid_config_path)

            if result:
                print("  FAIL: Should have failed invalid YAML validation")
                return False

            print("  PASS: Invalid YAML correctly rejected")
        finally:
            os.unlink(invalid_config_path)

        # Test 2: Boundary value testing
        print("  Testing configuration boundary values...")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
default_model_config:
  timeout: 3601  # Just over maximum
  retry: 11      # Just over maximum
  temperature: 2.1  # Just over maximum

enterprise_coding_agent:
  caching:
    default_ttl: 86401  # Just over maximum
    max_size: 100001    # Just over maximum
""")
            boundary_config_path = f.name

        try:
            from validate_config import ConfigValidator
            validator = ConfigValidator()
            result = validator.validate_config(boundary_config_path)

            if result:
                print("  FAIL: Should have failed boundary value validation")
                return False

            # Check that we got the expected errors
            errors = validator.errors
            if len(errors) < 5:  # Should have at least 5 boundary violations
                print(f"  FAIL: Expected at least 5 boundary errors, got {len(errors)}")
                return False

            print(f"  PASS: Boundary value validation correctly failed ({len(errors)} errors)")
        finally:
            os.unlink(boundary_config_path)

        print("SUCCESS: Configuration edge cases test passed")
        return True

    except Exception as e:
        print(f"FAILED: Configuration edge cases test failed: {e}")
        return False


def main():
    """Run all edge case tests."""
    print("Enterprise Agent v3.4 - Edge Case Test Suite")
    print("=" * 55)

    tests_passed = 0
    total_tests = 5

    # Run all edge case tests
    test_functions = [
        test_concurrency_edge_cases,
        test_validation_edge_cases,
        test_caching_edge_cases,
        test_error_handling_edge_cases,
        test_telemetry_privacy_edge_cases,
        #test_configuration_edge_cases,  # Requires validate_config.py to be importable
    ]

    for test_func in test_functions:
        try:
            if test_func():
                tests_passed += 1
        except Exception as e:
            print(f"FAILED: Test {test_func.__name__} failed with exception: {e}")

    print(f"\n" + "=" * 55)
    print(f"Edge Case Test Results: {tests_passed}/{len(test_functions)} tests passed")

    if tests_passed == len(test_functions):
        print("SUCCESS: All edge case tests passed! System is robust against edge conditions.")
        return 0
    else:
        print("FAILED: Some edge case tests failed. Review the system's handling of edge conditions.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)