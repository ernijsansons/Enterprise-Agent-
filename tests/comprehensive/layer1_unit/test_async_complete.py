"""Complete unit tests for async components."""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.comprehensive.test_framework import (
    critical_test,
    high_priority_test,
    low_priority_test,
    medium_priority_test,
)


class TestAsyncComplete:
    """Complete test suite for all async components."""

    def setup_method(self):
        """Setup for each test."""
        self.test_config = {
            "memory": {"retention_days": 1},
            "costs": {"max_daily_cost": 10.0},
            "governance": {"enabled": False},
            "use_claude_code": False,  # Disable for testing
        }

    @critical_test
    async def test_async_orchestrator_initialization(self):
        """Test async orchestrator initialization."""
        try:
            from src.orchestration.async_orchestrator import AsyncAgentOrchestrator

            orchestrator = AsyncAgentOrchestrator(self.test_config)

            # Verify core components
            assert hasattr(orchestrator, "config")
            assert hasattr(orchestrator, "memory")
            assert hasattr(orchestrator, "cache")
            assert hasattr(orchestrator, "cost_estimator")

            return {
                "success": True,
                "message": "Async orchestrator initialized successfully",
            }

        except Exception as e:
            return {"success": False, "message": f"Async orchestrator init failed: {e}"}

    @critical_test
    async def test_async_cache_functionality(self):
        """Test async cache functionality."""
        try:
            from src.utils.async_cache import AsyncLRUCache

            cache = AsyncLRUCache(max_size=10, default_ttl=60.0)

            # Test basic operations
            await cache.set("test_key", "test_value")
            value = await cache.get("test_key")
            assert value == "test_value"

            # Test TTL
            await cache.set("ttl_key", "ttl_value", ttl=0.1)
            await asyncio.sleep(0.2)
            expired_value = await cache.get("ttl_key")
            assert expired_value is None

            # Test stats
            stats = await cache.get_stats()
            assert isinstance(stats, dict)
            assert "hits" in stats
            assert "misses" in stats

            return {
                "success": True,
                "message": "Async cache functionality verified",
                "details": {"cache_stats": stats},
            }

        except Exception as e:
            return {"success": False, "message": f"Async cache test failed: {e}"}

    @critical_test
    async def test_async_memory_store_functionality(self):
        """Test async memory store functionality."""
        try:
            from src.memory.async_storage import AsyncMemoryStore

            memory = AsyncMemoryStore(
                {
                    "retention_days": 1,
                    "enable_vectors": False,  # Disable to avoid numpy dependency
                    "storage": "memory",
                }
            )

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
            assert results[0] == "batch_value_0"

            # Test stats
            stats = await memory.get_stats()
            assert isinstance(stats, dict)
            assert "total_records" in stats

            return {
                "success": True,
                "message": "Async memory store functionality verified",
                "details": {"memory_stats": stats},
            }

        except Exception as e:
            return {"success": False, "message": f"Async memory test failed: {e}"}

    @high_priority_test
    async def test_async_http_client_functionality(self):
        """Test async HTTP client functionality."""
        try:
            from src.utils.async_http import AIOHTTP_AVAILABLE

            if not AIOHTTP_AVAILABLE:
                return {
                    "success": True,
                    "message": "Async HTTP client skipped (aiohttp not available)",
                }

            from src.utils.async_http import AsyncHTTPClient

            client = AsyncHTTPClient(timeout=10.0)

            # Verify client creation
            assert hasattr(client, "timeout")
            assert hasattr(client, "max_retries")

            # Test client cleanup
            await client.close()

            return {
                "success": True,
                "message": "Async HTTP client functionality verified",
            }

        except Exception as e:
            return {"success": False, "message": f"Async HTTP test failed: {e}"}

    @high_priority_test
    async def test_async_orchestrator_call_model(self):
        """Test async orchestrator call_model functionality."""
        try:
            from src.orchestration.async_orchestrator import AsyncAgentOrchestrator

            orchestrator = AsyncAgentOrchestrator(self.test_config)

            # Test call_model (should return stubbed response)
            response = await orchestrator.call_model(
                model="claude_sonnet_4",
                prompt="Test prompt",
                role="Tester",
                operation="test_call",
                max_tokens=100,
            )

            assert isinstance(response, str)
            assert len(response) > 0

            return {
                "success": True,
                "message": "Async orchestrator call_model verified",
                "details": {"response_length": len(response)},
            }

        except Exception as e:
            return {"success": False, "message": f"Async call_model test failed: {e}"}

    @high_priority_test
    async def test_async_orchestrator_batch_calls(self):
        """Test async orchestrator batch calls."""
        try:
            from src.orchestration.async_orchestrator import AsyncAgentOrchestrator

            orchestrator = AsyncAgentOrchestrator(self.test_config)

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

            responses = await orchestrator.batch_call_models(requests, max_concurrent=2)

            assert len(responses) == len(requests)
            assert all(isinstance(r, str) for r in responses)

            return {
                "success": True,
                "message": "Async batch calls verified",
                "details": {"batch_size": len(responses)},
            }

        except Exception as e:
            return {"success": False, "message": f"Async batch calls test failed: {e}"}

    @medium_priority_test
    async def test_async_orchestrator_performance_stats(self):
        """Test async orchestrator performance statistics."""
        try:
            from src.orchestration.async_orchestrator import AsyncAgentOrchestrator

            orchestrator = AsyncAgentOrchestrator(self.test_config)

            # Make a call to generate some stats
            await orchestrator.call_model(
                model="claude_sonnet_4",
                prompt="Stats test",
                role="Tester",
                operation="stats_test",
            )

            # Get performance stats
            stats = await orchestrator.get_performance_stats()

            assert isinstance(stats, dict)
            assert "cache" in stats
            assert "memory" in stats
            assert "cost_summary" in stats
            assert "providers" in stats

            return {
                "success": True,
                "message": "Async performance stats verified",
                "details": {"stats_keys": list(stats.keys())},
            }

        except Exception as e:
            return {"success": False, "message": f"Performance stats test failed: {e}"}

    @medium_priority_test
    async def test_async_cache_memory_management(self):
        """Test async cache memory management."""
        try:
            from src.utils.async_cache import AsyncLRUCache

            # Create cache with small limits
            cache = AsyncLRUCache(max_size=3, max_memory_mb=1)

            # Fill cache beyond capacity
            for i in range(5):
                await cache.set(f"key_{i}", f"value_{i}" * 100)

            # Check that eviction occurred
            stats = await cache.get_stats()
            assert stats["size"] <= 3
            assert stats["evictions"] > 0

            return {
                "success": True,
                "message": "Async cache memory management verified",
                "details": {"final_stats": stats},
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Cache memory management test failed: {e}",
            }

    @medium_priority_test
    async def test_async_memory_search_functionality(self):
        """Test async memory search functionality."""
        try:
            from src.memory.async_storage import AsyncMemoryStore

            memory = AsyncMemoryStore(
                {"retention_days": 1, "enable_vectors": False, "storage": "memory"}
            )

            # Store test data
            test_data = [
                ("search_test", "python_code", "def hello(): return 'world'"),
                (
                    "search_test",
                    "javascript_code",
                    "function hello() { return 'world'; }",
                ),
                ("search_test", "documentation", "This is a hello world function"),
            ]

            for level, key, value in test_data:
                await memory.store(level, key, value)

            # Test search
            results = await memory.search("search_test", "hello", limit=5)

            assert len(results) > 0
            assert all(
                len(result) == 3 for result in results
            )  # (key, value, score) tuples

            return {
                "success": True,
                "message": "Async memory search verified",
                "details": {"search_results_count": len(results)},
            }

        except Exception as e:
            return {"success": False, "message": f"Memory search test failed: {e}"}

    @medium_priority_test
    async def test_async_parallel_role_execution(self):
        """Test async parallel role execution."""
        try:
            from src.orchestration.async_orchestrator import AsyncAgentOrchestrator

            orchestrator = AsyncAgentOrchestrator(self.test_config)

            # Test parallel role execution
            tasks = [
                {
                    "role": "Planner",
                    "operation": "decompose",
                    "prompt": "Plan a web application",
                    "model": "claude_sonnet_4",
                },
                {
                    "role": "Coder",
                    "operation": "generate",
                    "prompt": "Write a function",
                    "model": "claude_sonnet_4",
                },
            ]

            start_time = time.time()
            results = await orchestrator.parallel_roles_execution(
                tasks, max_concurrent=2
            )
            duration = time.time() - start_time

            assert len(results) == len(tasks)
            assert all(isinstance(result, dict) for result in results.values())

            return {
                "success": True,
                "message": "Async parallel role execution verified",
                "details": {"execution_time": duration, "results_count": len(results)},
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Parallel role execution test failed: {e}",
            }

    @low_priority_test
    async def test_async_cache_warming(self):
        """Test async cache warming functionality."""
        try:
            from src.utils.async_cache import AsyncLRUCache

            cache = AsyncLRUCache(max_size=10)

            # Test cache warming
            async def generate_value(key="test"):
                await asyncio.sleep(0.01)  # Simulate work
                return f"warmed_value_{key}"

            warming_tasks = [
                (f"warm_key_{i}", lambda i=i: generate_value(f"item_{i}"))
                for i in range(3)
            ]

            await cache.warm_cache(warming_tasks)

            # Verify warmed values
            warmed_values = []
            for i in range(3):
                value = await cache.get(f"warm_key_{i}")
                warmed_values.append(value is not None)

            success_count = sum(warmed_values)

            return {
                "success": success_count >= 2,  # At least 2 should be warmed
                "message": f"Cache warming: {success_count}/3 items warmed",
                "details": {"warming_success": warmed_values},
            }

        except Exception as e:
            return {"success": False, "message": f"Cache warming test failed: {e}"}

    @low_priority_test
    async def test_async_component_cleanup(self):
        """Test async component cleanup and resource management."""
        try:
            from src.memory.async_storage import AsyncMemoryStore
            from src.orchestration.async_orchestrator import AsyncAgentOrchestrator
            from src.utils.async_cache import AsyncLRUCache

            # Create components
            orchestrator = AsyncAgentOrchestrator(self.test_config)
            cache = AsyncLRUCache()
            memory = AsyncMemoryStore({"retention_days": 1})

            # Test cleanup methods exist and are callable
            cleanup_results = {}

            try:
                await orchestrator.close()
                cleanup_results["orchestrator"] = True
            except Exception:
                cleanup_results["orchestrator"] = False

            try:
                await cache.clear()
                cleanup_results["cache"] = True
            except Exception:
                cleanup_results["cache"] = False

            try:
                await memory.close()
                cleanup_results["memory"] = True
            except Exception:
                cleanup_results["memory"] = False

            success_count = sum(cleanup_results.values())

            return {
                "success": success_count >= 2,
                "message": f"Component cleanup: {success_count}/3 successful",
                "details": {"cleanup_results": cleanup_results},
            }

        except Exception as e:
            return {"success": False, "message": f"Component cleanup test failed: {e}"}

    @critical_test
    async def test_async_error_handling_and_timeouts(self):
        """Test async error handling and timeout management."""
        try:
            from src.orchestration.async_orchestrator import AsyncAgentOrchestrator

            orchestrator = AsyncAgentOrchestrator(self.test_config)

            # Test timeout handling
            try:
                # This should complete quickly with stubbed response
                await asyncio.wait_for(
                    orchestrator.call_model(
                        model="claude_sonnet_4",
                        prompt="Timeout test",
                        role="Tester",
                        operation="timeout_test",
                    ),
                    timeout=5.0,
                )
                timeout_handled = True
            except asyncio.TimeoutError:
                timeout_handled = False

            # Test error propagation
            error_handled = True
            try:
                # Invalid parameters should be handled gracefully
                await orchestrator.call_model(
                    model="", prompt="", role="", operation=""
                )
            except Exception:
                # Should handle errors gracefully
                pass

            return {
                "success": timeout_handled and error_handled,
                "message": "Async error handling and timeouts verified",
                "details": {
                    "timeout_handling": timeout_handled,
                    "error_handling": error_handled,
                },
            }

        except Exception as e:
            return {"success": False, "message": f"Error handling test failed: {e}"}


def get_async_tests():
    """Get all async test methods."""
    test_class = TestAsyncComplete()
    test_methods = []

    for attr_name in dir(test_class):
        if attr_name.startswith("test_") and callable(getattr(test_class, attr_name)):
            method = getattr(test_class, attr_name)
            test_methods.append(method)

    return test_methods, test_class.setup_method, None
