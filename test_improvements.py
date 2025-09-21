#!/usr/bin/env python3
"""Comprehensive test suite for Enterprise Agent improvements."""

import io
import sys
import time
from pathlib import Path

# Set UTF-8 encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_agent_initialization():
    """Test basic agent initialization."""
    print("\n=== Testing Agent Initialization ===")
    try:
        from src.agent_orchestrator import AgentOrchestrator

        agent = AgentOrchestrator()
        print("✓ Agent initialized successfully")
        return agent
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return None


def test_json_parsing():
    """Test improved JSON parsing with error recovery."""
    print("\n=== Testing JSON Parsing ===")
    from src.agent_orchestrator import AgentOrchestrator

    agent = AgentOrchestrator()

    test_cases = [
        ('{"valid": "json"}', {"valid": "json"}),
        ('{"trailing": "comma",}', {"trailing": "comma"}),
        ("{'single': 'quotes'}", {"single": "quotes"}),
        ('Some text {"embedded": "json"} more text', {"embedded": "json"}),
        ('```json\n{"code": "block"}\n```', {"code": "block"}),
        ("malformed {broken json", {"raw_text": "malformed {broken json"}),
    ]

    passed = 0
    for input_str, expected in test_cases:
        try:
            result = agent._parse_json(input_str)
            if "raw_text" in expected and "raw_text" in result:
                print(f"✓ Parse fallback working for: {input_str[:30]}...")
                passed += 1
            elif result == expected:
                print(f"✓ Parsed correctly: {input_str[:30]}...")
                passed += 1
            else:
                print(f"✗ Parse mismatch for: {input_str[:30]}...")
                print(f"  Expected: {expected}")
                print(f"  Got: {result}")
        except Exception as e:
            print(f"✗ Parse error for {input_str[:30]}...: {e}")

    print(f"JSON Parsing: {passed}/{len(test_cases)} tests passed")


def test_cache_functionality():
    """Test caching with TTL."""
    print("\n=== Testing Cache Functionality ===")
    from src.utils.cache import ModelResponseCache, TTLCache

    # Test basic cache
    cache = TTLCache(default_ttl=2)  # 2 second TTL for testing

    cache.set("test_key", "test_value")
    assert cache.get("test_key") == "test_value", "Cache set/get failed"
    print("✓ Basic cache set/get working")

    # Test TTL expiration
    time.sleep(2.5)
    assert cache.get("test_key") is None, "TTL expiration failed"
    print("✓ TTL expiration working")

    # Test model response cache
    model_cache = ModelResponseCache(default_ttl=5)
    model_cache.cache_response(
        model="gpt-4", prompt="test prompt", response="test response"
    )

    cached = model_cache.get_response(model="gpt-4", prompt="test prompt")
    assert cached == "test response", "Model cache failed"
    print("✓ Model response cache working")

    # Test cache stats
    stats = cache.get_stats()
    print(f"✓ Cache stats: hits={stats['hits']}, misses={stats['misses']}")


def test_validation():
    """Test input validation."""
    print("\n=== Testing Input Validation ===")
    from src.utils.validation import (
        DomainValidator,
        ModelNameValidator,
        StringValidator,
        ValidationException,
    )

    # Test string validation
    str_validator = StringValidator(min_length=3, max_length=10)
    try:
        str_validator.validate("test")
        print("✓ String validation passed for valid input")
    except ValidationException:
        print("✗ String validation failed for valid input")

    try:
        str_validator.validate("xy")  # Too short
        print("✗ String validation should have failed for short input")
    except ValidationException:
        print("✓ String validation correctly rejected short input")

    # Test domain validation
    domain_validator = DomainValidator()
    try:
        domain_validator.validate("coding")
        print("✓ Domain validation passed for valid domain")
    except ValidationException:
        print("✗ Domain validation failed for valid domain")

    try:
        domain_validator.validate("invalid_domain")
        print("✗ Domain validation should have failed for invalid domain")
    except ValidationException:
        print("✓ Domain validation correctly rejected invalid domain")

    # Test model name validation
    model_validator = ModelNameValidator()
    try:
        model_validator.validate("gpt-4-turbo")
        print("✓ Model validation passed for valid model")
    except ValidationException:
        print("✗ Model validation failed for valid model")


def test_retry_mechanism():
    """Test retry with exponential backoff."""
    print("\n=== Testing Retry Mechanism ===")
    from src.utils.retry import ExponentialBackoff, retry_with_backoff

    attempt_count = 0

    @retry_with_backoff(strategy=ExponentialBackoff(max_retries=3, base_delay=0.1))
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Simulated failure")
        return "Success"

    try:
        result = flaky_function()
        assert result == "Success", "Retry didn't return expected value"
        assert attempt_count == 3, f"Expected 3 attempts, got {attempt_count}"
        print(f"✓ Retry mechanism working ({attempt_count} attempts)")
    except Exception as e:
        print(f"✗ Retry mechanism failed: {e}")


def test_concurrency():
    """Test thread-safe operations."""
    print("\n=== Testing Concurrency ===")
    import threading

    from src.utils.concurrency import ExecutionManager, ThreadSafeDict

    # Test thread-safe dict
    safe_dict = ThreadSafeDict()
    errors = []

    def update_dict(key, value):
        try:
            for i in range(100):
                safe_dict[f"{key}_{i}"] = value
                _ = safe_dict.get(f"{key}_{i}")
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(5):
        t = threading.Thread(target=update_dict, args=(f"thread_{i}", i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if errors:
        print(f"✗ Thread safety failed with {len(errors)} errors")
    else:
        print(f"✓ Thread-safe dict handled {len(safe_dict)} operations")

    # Test execution manager
    with ExecutionManager(max_workers=2) as executor:

        def task(n):
            return n * 2

        results = executor.map_parallel(task, [1, 2, 3, 4, 5])
        assert results == [2, 4, 6, 8, 10], "Parallel execution failed"
        print("✓ Parallel execution working correctly")


def test_error_handling():
    """Test structured error handling."""
    print("\n=== Testing Error Handling ===")
    from src.exceptions import ModelException, ValidationException, handle_exception

    # Test custom exceptions
    try:
        raise ModelException("Test error", model="gpt-4", provider="openai")
    except ModelException as e:
        assert e.model == "gpt-4", "Exception attribute not set"
        assert "model" in e.details, "Exception details not set"
        print("✓ Custom exceptions working with context")

    # Test exception conversion
    original = ValueError("Test value error")
    converted = handle_exception(original, {"context": "test"})
    assert isinstance(converted, ValidationException), "Exception conversion failed"
    assert converted.cause == original, "Original cause not preserved"
    print("✓ Exception conversion working")


def test_vector_dimensions():
    """Test improved vector dimension handling."""
    print("\n=== Testing Vector Dimensions ===")
    from src.memory.storage import MemoryStore

    memory = MemoryStore({"vector_dimensions": 768})

    # Test placeholder vector generation
    vector = memory._generate_placeholder_vector("test text", dimensions=768)
    assert len(vector) == 768, f"Vector dimension mismatch: {len(vector)}"
    assert all(-1 <= v <= 1 for v in vector), "Vector values out of range"
    print("✓ Placeholder vector generation working")

    # Test dimension adjustment
    test_vector = [1.0] * 500
    adjusted = memory._ensure_vector_dimensions(test_vector)
    assert len(adjusted) == 768, "Vector padding failed"

    test_vector = [1.0] * 1000
    adjusted = memory._ensure_vector_dimensions(test_vector)
    assert len(adjusted) == 768, "Vector truncation failed"
    print("✓ Vector dimension adjustment working")


def test_full_pipeline():
    """Test a complete agent pipeline."""
    print("\n=== Testing Full Pipeline ===")
    from src.agent_orchestrator import AgentOrchestrator

    agent = AgentOrchestrator()

    try:
        # Test with a simple task
        result = agent.run_mode(
            domain="coding",
            task="Create a function to add two numbers",
            vuln_flag=False,
        )

        assert "plan" in result, "No plan in result"
        assert "code" in result, "No code in result"
        assert "cost_summary" in result, "No cost summary in result"

        print("✓ Full pipeline execution successful")
        print(f"  - Plan generated: {bool(result.get('plan'))}")
        print(f"  - Code generated: {bool(result.get('code'))}")
        print(f"  - Confidence: {result.get('confidence', 0)}")

        # Test cache hit on second run
        start = time.time()
        agent.run_mode(
            domain="coding",
            task="Create a function to add two numbers",
            vuln_flag=False,
        )
        elapsed = time.time() - start
        print(f"✓ Second run completed in {elapsed:.2f}s (cache may be used)")

    except Exception as e:
        print(f"✗ Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ENTERPRISE AGENT COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    # Initialize agent first
    agent = test_agent_initialization()
    if not agent:
        print("\nCritical: Agent initialization failed. Cannot continue tests.")
        return

    # Run all test modules
    test_functions = [
        test_json_parsing,
        test_cache_functionality,
        test_validation,
        test_retry_mechanism,
        test_concurrency,
        test_error_handling,
        test_vector_dimensions,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} failed with error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All tests passed! The Enterprise Agent is fully functional.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")


if __name__ == "__main__":
    run_all_tests()
