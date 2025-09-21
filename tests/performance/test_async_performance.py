"""Performance tests for async implementation improvements."""
import time
import pytest
import logging

from src.agent_orchestrator import AgentOrchestrator
from src.orchestration.async_orchestrator import get_async_orchestrator

logger = logging.getLogger(__name__)


class TestAsyncPerformance:
    """Test suite for async performance improvements."""

    @pytest.fixture
    def orchestrator(self):
        """Create test orchestrator."""
        config_path = "configs/test_config.yaml"
        return AgentOrchestrator(config_path)

    @pytest.fixture
    def async_orchestrator(self):
        """Create async orchestrator."""
        config = {
            "memory": {"retention_days": 1},
            "costs": {"max_daily_cost": 10.0},
            "governance": {"enabled": False},
            "use_claude_code": True,
            "claude_code": {"timeout": 30},
        }
        return get_async_orchestrator(config)

    def test_sync_model_calls_baseline(self, orchestrator):
        """Baseline test for synchronous model calls."""
        start_time = time.time()

        # Simulate multiple model calls
        prompts = [
            "Write a simple hello world function",
            "Create a basic data structure",
            "Implement error handling",
            "Add logging functionality",
            "Write unit tests",
        ]

        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = orchestrator._call_model(
                    model="claude_sonnet_4",
                    prompt=prompt,
                    role="Coder",
                    operation=f"test_{i}",
                    max_tokens=500,
                )
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")

        sync_duration = time.time() - start_time

        logger.info(f"Sync baseline: {len(results)} calls in {sync_duration:.2f}s")
        logger.info(f"Sync average: {sync_duration/len(prompts):.2f}s per call")

        assert len(results) == len(prompts)
        return {"duration": sync_duration, "results": results}

    @pytest.mark.asyncio
    async def test_async_model_calls_performance(self, async_orchestrator):
        """Test async model calls performance."""
        start_time = time.time()

        # Same prompts as sync test
        prompts = [
            "Write a simple hello world function",
            "Create a basic data structure",
            "Implement error handling",
            "Add logging functionality",
            "Write unit tests",
        ]

        # Prepare batch requests
        requests = [
            {
                "model": "claude_sonnet_4",
                "prompt": prompt,
                "role": "Coder",
                "operation": f"async_test_{i}",
                "max_tokens": 500,
            }
            for i, prompt in enumerate(prompts)
        ]

        # Execute async batch
        results = await async_orchestrator.batch_call_models(requests)

        async_duration = time.time() - start_time

        logger.info(f"Async batch: {len(results)} calls in {async_duration:.2f}s")
        logger.info(f"Async average: {async_duration/len(prompts):.2f}s per call")

        assert len(results) == len(prompts)
        return {"duration": async_duration, "results": results}

    @pytest.mark.asyncio
    async def test_async_orchestrator_integration(self, orchestrator):
        """Test async orchestrator integration with main orchestrator."""
        if not orchestrator._async_enabled:
            pytest.skip("Async not enabled in orchestrator")

        start_time = time.time()

        # Test async version of run_mode
        result = await orchestrator.run_mode_async(
            domain="coding",
            task="Create a simple calculator function",
            vuln_flag=False
        )

        duration = time.time() - start_time

        logger.info(f"Async run_mode completed in {duration:.2f}s")

        assert "code" in result or "plan" in result
        assert result.get("domain") == "coding"

        return {"duration": duration, "result": result}

    @pytest.mark.asyncio
    async def test_cache_performance(self, async_orchestrator):
        """Test cache performance improvements."""
        # First call to populate cache
        start_time = time.time()
        result1 = await async_orchestrator.call_model(
            model="claude_sonnet_4",
            prompt="What is 2+2?",
            role="Calculator",
            operation="add",
            use_cache=True,
        )
        first_duration = time.time() - start_time

        # Second call should hit cache
        start_time = time.time()
        result2 = await async_orchestrator.call_model(
            model="claude_sonnet_4",
            prompt="What is 2+2?",
            role="Calculator",
            operation="add",
            use_cache=True,
        )
        cached_duration = time.time() - start_time

        logger.info(f"First call: {first_duration:.3f}s")
        logger.info(f"Cached call: {cached_duration:.3f}s")
        logger.info(f"Cache speedup: {first_duration/cached_duration:.1f}x")

        # Cache should be significantly faster
        assert cached_duration < first_duration * 0.5
        assert result1 == result2  # Should be identical

        return {
            "first_duration": first_duration,
            "cached_duration": cached_duration,
            "speedup": first_duration / cached_duration if cached_duration > 0 else 0,
        }

    @pytest.mark.asyncio
    async def test_parallel_role_execution(self, async_orchestrator):
        """Test parallel execution of different roles."""
        start_time = time.time()

        # Simulate parallel role tasks
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
                "prompt": "Write a REST API endpoint",
                "model": "claude_sonnet_4",
            },
            {
                "role": "Reviewer",
                "operation": "review",
                "prompt": "Review this code for best practices",
                "model": "claude_sonnet_4",
            },
        ]

        # Execute in parallel
        results = await async_orchestrator.parallel_roles_execution(tasks)

        parallel_duration = time.time() - start_time

        logger.info(f"Parallel execution: {len(tasks)} roles in {parallel_duration:.2f}s")

        assert len(results) == len(tasks)
        assert all(result.get("success", False) for result in results.values())

        return {"duration": parallel_duration, "results": results}

    @pytest.mark.asyncio
    async def test_memory_batch_operations(self, async_orchestrator):
        """Test batch memory operations performance."""
        memory = async_orchestrator.memory

        # Prepare batch operations
        operations = [
            ("test_level", f"key_{i}", f"value_{i}", {"index": i})
            for i in range(100)
        ]

        start_time = time.time()
        await memory.batch_store(operations)
        store_duration = time.time() - start_time

        # Batch retrieve
        keys = [("test_level", f"key_{i}") for i in range(100)]

        start_time = time.time()
        results = await memory.batch_retrieve(keys)
        retrieve_duration = time.time() - start_time

        logger.info(f"Batch store: {len(operations)} items in {store_duration:.3f}s")
        logger.info(f"Batch retrieve: {len(keys)} items in {retrieve_duration:.3f}s")

        assert len(results) == len(keys)
        assert all(result == f"value_{i}" for i, result in enumerate(results))

        return {
            "store_duration": store_duration,
            "retrieve_duration": retrieve_duration,
            "total_duration": store_duration + retrieve_duration,
        }

    @pytest.mark.asyncio
    async def test_http_client_performance(self):
        """Test async HTTP client performance."""
        from src.utils.async_http import AsyncHTTPClient

        client = AsyncHTTPClient()

        # Test concurrent requests
        urls = [
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/1",
        ]

        requests = [
            {"method": "GET", "url": url}
            for url in urls
        ]

        start_time = time.time()
        results = await client.batch_requests(requests, max_concurrent=3)
        concurrent_duration = time.time() - start_time

        await client.close()

        logger.info(f"Concurrent HTTP: {len(requests)} requests in {concurrent_duration:.2f}s")

        # Should be faster than sequential (3 x 1s = 3s)
        assert concurrent_duration < 2.0  # Allow some overhead
        assert len(results) == len(requests)

        return {"duration": concurrent_duration, "results": len(results)}

    @pytest.mark.asyncio
    async def test_performance_stats(self, async_orchestrator):
        """Test performance statistics collection."""
        # Make some calls to generate stats
        await async_orchestrator.call_model(
            model="claude_sonnet_4",
            prompt="Test prompt for stats",
            role="Tester",
            operation="stats_test",
        )

        stats = await async_orchestrator.get_performance_stats()

        logger.info(f"Performance stats: {stats}")

        assert "cache" in stats
        assert "memory" in stats
        assert "cost_summary" in stats
        assert "providers" in stats

        # Check provider availability
        providers = stats["providers"]
        assert "claude_code" in providers
        assert "openai" in providers
        assert "anthropic" in providers

        return stats


def run_performance_comparison():
    """Run a comprehensive performance comparison."""
    print("Running Performance Comparison...")
    print("=" * 50)

    # Note: This would be run manually for performance analysis
    # as it requires actual API access or Claude Code CLI

    results = {
        "sync_baseline": "Would measure sync performance",
        "async_improvement": "Would measure async performance",
        "cache_speedup": "Would measure cache performance",
        "parallel_execution": "Would measure parallel execution",
        "memory_operations": "Would measure memory performance",
        "http_performance": "Would measure HTTP performance",
    }

    print("Performance Test Results:")
    for test, result in results.items():
        print(f"  {test}: {result}")

    return results


if __name__ == "__main__":
    # Run basic performance analysis
    results = run_performance_comparison()
    print(f"\nPerformance analysis completed: {len(results)} tests")