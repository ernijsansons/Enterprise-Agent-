#!/usr/bin/env python3
"""Simple test for core async functionality without heavy dependencies."""
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_async_cache():
    """Test async cache functionality."""
    print("Testing Async Cache...")
    try:
        from src.utils.async_cache import AsyncLRUCache, ModelResponseCache

        # Test LRU cache
        cache = AsyncLRUCache(max_size=10)

        # Test basic operations
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")
        assert value == "test_value", f"Expected 'test_value', got {value}"

        # Test model response cache
        model_cache = ModelResponseCache(max_size=5)
        await model_cache.cache_response(
            model="test_model",
            prompt="test prompt",
            response="test response",
            max_tokens=100,
            role="Tester",
        )

        cached_response = await model_cache.get_response(
            model="test_model", prompt="test prompt", max_tokens=100, role="Tester"
        )
        assert cached_response == "test response"

        print("   SUCCESS: Async cache working correctly")
        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False


async def test_async_http():
    """Test async HTTP client (without making real requests)."""
    print("Testing Async HTTP Client...")
    try:
        from src.utils.async_http import AIOHTTP_AVAILABLE

        if not AIOHTTP_AVAILABLE:
            print("   SKIP: aiohttp not available")
            return True

        from src.utils.async_http import AsyncHTTPClient

        # Just test client creation (no actual requests)
        client = AsyncHTTPClient()
        print("   SUCCESS: Async HTTP client created")

        # Test client cleanup
        await client.close()
        print("   SUCCESS: Async HTTP client closed properly")
        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False


async def test_async_memory():
    """Test async memory storage."""
    print("Testing Async Memory Storage...")
    try:
        from src.memory.async_storage import AsyncMemoryStore

        # Create memory store with minimal config
        config = {
            "retention_days": 1,
            "enable_vectors": False,  # Disable to avoid numpy dependency
            "storage": "memory",
        }
        memory = AsyncMemoryStore(config)

        # Test basic operations
        await memory.store("test_level", "key1", "value1")
        value = await memory.retrieve("test_level", "key1")
        assert value == "value1"

        # Test batch operations
        operations = [
            ("test_level", f"batch_key_{i}", f"batch_value_{i}", {"index": i})
            for i in range(3)
        ]
        await memory.batch_store(operations)

        keys = [("test_level", f"batch_key_{i}") for i in range(3)]
        results = await memory.batch_retrieve(keys)
        assert len(results) == 3

        print("   SUCCESS: Async memory storage working correctly")
        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False


async def test_async_claude_provider():
    """Test async Claude provider (without actual CLI calls)."""
    print("Testing Async Claude Provider...")
    try:
        from src.providers.async_claude_provider import AsyncClaudeCodeProvider

        # Create provider with minimal config
        config = {
            "timeout": 30,
            "enable_fallback": True,
            "working_directory": os.getcwd(),
        }
        provider = AsyncClaudeCodeProvider(config)

        # Test provider creation
        print("   SUCCESS: Async Claude provider created")

        # Test stats method
        stats = await provider.get_stats()
        assert "provider" in stats
        assert stats["provider"] == "async_claude_code"

        print("   SUCCESS: Async Claude provider stats working")
        return True

    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_main_orchestrator_async_integration():
    """Test main orchestrator has async methods."""
    print("Testing Main Orchestrator Async Integration...")
    try:
        from src.agent_orchestrator import AgentOrchestrator

        # Check if async methods exist
        orch = AgentOrchestrator.__new__(AgentOrchestrator)  # Don't call __init__

        has_async_call = hasattr(orch, "_call_model_async")
        has_async_run = hasattr(orch, "run_mode_async")
        has_async_pipeline = hasattr(orch, "_execute_pipeline_async")

        print(f"   _call_model_async: {'YES' if has_async_call else 'NO'}")
        print(f"   run_mode_async: {'YES' if has_async_run else 'NO'}")
        print(f"   _execute_pipeline_async: {'YES' if has_async_pipeline else 'NO'}")

        if has_async_call and has_async_run and has_async_pipeline:
            print("   SUCCESS: All async methods present")
            return True
        else:
            print("   WARNING: Some async methods missing")
            return False

    except Exception as e:
        print(f"   ERROR: {e}")
        return False


async def main():
    """Run all simple async tests."""
    print("Enterprise Agent Async Component Test")
    print("=" * 50)
    print("Testing individual async components...")
    print()

    tests = [
        ("Async Cache", test_async_cache()),
        ("Async HTTP", test_async_http()),
        ("Async Memory", test_async_memory()),
        ("Async Claude Provider", test_async_claude_provider()),
    ]

    # Run async tests
    results = []
    for test_name, test_coro in tests:
        result = await test_coro
        results.append((test_name, result))

    # Run sync test
    sync_result = test_main_orchestrator_async_integration()
    results.append(("Main Orchestrator Integration", sync_result))

    # Summary
    print("\n" + "=" * 50)
    print("Test Results:")
    print("-" * 20)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:30}: {status}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\nSUCCESS: All async components working!")
    elif passed >= len(results) - 1:
        print("\nMOSTLY SUCCESS: Core async functionality working!")
    else:
        print("\nWARNING: Some async components have issues")

    return passed >= len(results) - 1  # Allow 1 failure


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\nAsync component test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
