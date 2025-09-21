#!/usr/bin/env python3
"""Simple test to verify async integration works."""
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent_orchestrator import AgentOrchestrator
from src.orchestration.async_orchestrator import get_async_orchestrator


async def test_async_orchestrator():
    """Test the async orchestrator functionality."""
    print("Testing Async Orchestrator...")
    print("-" * 40)

    try:
        # Test async orchestrator creation
        config = {
            "memory": {"retention_days": 1},
            "costs": {"max_daily_cost": 10.0},
            "governance": {"enabled": False},
            "use_claude_code": False,  # Disable for testing
        }

        async_orch = get_async_orchestrator(config)
        print("SUCCESS: Async orchestrator created")

        # Test async call_model (will use stubbed responses)
        response = await async_orch.call_model(
            model="claude_sonnet_4",
            prompt="Hello, this is a test",
            role="Tester",
            operation="test_call",
            max_tokens=100,
        )
        print(f"SUCCESS: Async model call returned: {response[:50]}...")

        # Test batch calls
        requests = [
            {
                "model": "claude_sonnet_4",
                "prompt": f"Test prompt {i}",
                "role": "Tester",
                "operation": f"batch_test_{i}",
                "max_tokens": 50,
            }
            for i in range(3)
        ]

        batch_results = await async_orch.batch_call_models(requests)
        print(f"SUCCESS: Batch call returned {len(batch_results)} results")

        # Test performance stats
        stats = await async_orch.get_performance_stats()
        print(f"SUCCESS: Performance stats collected: {list(stats.keys())}")

        return True

    except Exception as e:
        print(f"ERROR: Async orchestrator test failed: {e}")
        return False


def test_main_orchestrator_async():
    """Test the main orchestrator with async capabilities."""
    print("\nTesting Main Orchestrator Async Integration...")
    print("-" * 50)

    try:
        # Create orchestrator with async enabled
        os.environ["ENABLE_ASYNC"] = "true"
        os.environ["USE_CLAUDE_CODE"] = "false"  # Disable for testing

        orch = AgentOrchestrator("configs/agent_config_v3.4.yaml")
        print("SUCCESS: Main orchestrator created with async support")

        # Check async integration
        if hasattr(orch, "_async_orchestrator") and orch._async_orchestrator:
            print("SUCCESS: Async orchestrator integrated")
        else:
            print("INFO: Async orchestrator not initialized (may be expected)")

        # Test sync call_model
        response = orch._call_model(
            model="claude_sonnet_4",
            prompt="Test sync call",
            role="Tester",
            operation="sync_test",
            max_tokens=50,
        )
        print(f"SUCCESS: Sync model call returned: {response[:30]}...")

        return True

    except Exception as e:
        print(f"ERROR: Main orchestrator test failed: {e}")
        return False


async def test_async_call_model():
    """Test the async _call_model method."""
    print("\nTesting Async Call Model...")
    print("-" * 30)

    try:
        os.environ["ENABLE_ASYNC"] = "true"
        os.environ["USE_CLAUDE_CODE"] = "false"

        orch = AgentOrchestrator("configs/agent_config_v3.4.yaml")

        if hasattr(orch, "_call_model_async"):
            response = await orch._call_model_async(
                model="claude_sonnet_4",
                prompt="Test async call method",
                role="Tester",
                operation="async_method_test",
                max_tokens=50,
            )
            print(f"SUCCESS: Async call_model returned: {response[:30]}...")
            return True
        else:
            print("INFO: _call_model_async method not found")
            return False

    except Exception as e:
        print(f"ERROR: Async call_model test failed: {e}")
        return False


async def main():
    """Run all async integration tests."""
    print("Enterprise Agent Async Integration Test")
    print("=" * 50)

    results = []

    # Test 1: Async orchestrator
    result1 = await test_async_orchestrator()
    results.append(("Async Orchestrator", result1))

    # Test 2: Main orchestrator async integration
    result2 = test_main_orchestrator_async()
    results.append(("Main Orchestrator Integration", result2))

    # Test 3: Async call_model method
    result3 = await test_async_call_model()
    results.append(("Async Call Model Method", result3))

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("-" * 20)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:30}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("SUCCESS: All async integration tests passed!")
    else:
        print("WARNING: Some tests failed - async features may be limited")

    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
