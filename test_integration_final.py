#!/usr/bin/env python
"""Final integration test for Enterprise Agent with Claude Code."""
import os
import sys
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Fix Windows encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Test results tracking
test_results = {"passed": 0, "failed": 0, "errors": []}


def test_step(description):
    """Decorator for test steps."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                print(f"📝 Testing: {description}...", end=" ")
                result = func(*args, **kwargs)
                test_results["passed"] += 1
                print("✅ PASS")
                return result
            except Exception as e:
                test_results["failed"] += 1
                test_results["errors"].append(f"{description}: {str(e)}")
                print(f"❌ FAIL: {e}")
                return None
        return wrapper
    return decorator


@test_step("Import core modules")
def test_imports():
    """Test all critical imports."""
    global AgentOrchestrator, ClaudeCodeProvider, ClaudeAuthManager
    global CostEstimator, MemoryStore, TTLCache

    from src.agent_orchestrator import AgentOrchestrator
    from src.providers.claude_code_provider import ClaudeCodeProvider
    from src.providers.auth_manager import ClaudeAuthManager
    from src.utils.costs import CostEstimator
    from src.memory import MemoryStore
    from src.utils.cache import TTLCache
    from src.governance import Governance
    from src.utils.hitl import HITLManager
    return True


@test_step("Initialize orchestrator with mocked components")
def test_orchestrator_init():
    """Test orchestrator initialization."""
    with patch.dict(os.environ, {"USE_CLAUDE_CODE": "false"}):
        orchestrator = AgentOrchestrator()
        assert orchestrator is not None
        assert orchestrator.cost_estimator is not None
        assert orchestrator.memory_store is not None
        return orchestrator


@test_step("Test Claude Code provider with mocked subprocess")
def test_claude_provider():
    """Test Claude Code provider functionality."""
    with patch("subprocess.run") as mock_run:
        # Mock successful CLI responses
        mock_run.return_value = Mock(
            returncode=0,
            stdout="claude version 1.0.0",
            stderr=""
        )

        provider = ClaudeCodeProvider({"timeout": 30})

        # Test model mapping
        assert provider._map_model_to_cli("claude_sonnet_4") == "sonnet"
        assert provider._map_model_to_cli("claude_opus_4") == "opus"

        # Test session management
        session_id = provider.create_session("test-session")
        assert "test-session" in provider.sessions

        provider.clear_session("test-session")
        assert "test-session" not in provider.sessions

        return provider


@test_step("Test authentication manager")
def test_auth_manager():
    """Test authentication management."""
    auth = ClaudeAuthManager()

    # Test API key handling
    original_key = os.environ.get("ANTHROPIC_API_KEY")
    try:
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        result = auth.ensure_subscription_mode()
        assert "ANTHROPIC_API_KEY" not in os.environ

        # Test plan verification
        plan_info = auth.verify_subscription_plan()
        assert "authenticated" in plan_info

        return auth
    finally:
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]


@test_step("Test cost tracking with zero-cost Claude Code")
def test_cost_tracking():
    """Test cost estimation and tracking."""
    estimator = CostEstimator({})

    # Track regular API call (should have cost)
    estimator.track(100, "Planner", "test", model="gpt-4")

    # Track Claude Code call (should be zero cost)
    estimator.track(0, "Coder", "claude_code", model="claude_code")

    summary = estimator.get_summary()
    assert "total_cost" in summary

    # Verify zero cost for Claude Code
    events = summary.get("events", [])
    for event in events:
        if "claude_code" in event.get("model", ""):
            assert event.get("cost", 1) == 0

    return estimator


@test_step("Test memory store operations")
def test_memory_store():
    """Test memory storage and retrieval."""
    memory = MemoryStore()

    # Store memories
    memory.store("test_key", "Test content", "project")
    memory.store("another_key", "Another content", "workflow")

    # Retrieve memories
    memories = memory.retrieve("project", top_k=5)
    assert isinstance(memories, list)

    # Test pruning
    initial_count = len(memory._memories)
    memory.prune(max_age_hours=24)
    assert len(memory._memories) <= initial_count

    return memory


@test_step("Test TTL cache functionality")
def test_cache():
    """Test caching with TTL."""
    cache = TTLCache(default_ttl=60)

    # Set and get
    cache.set("test_key", "test_value", ttl=300)
    value = cache.get("test_key")
    assert value == "test_value"

    # Test expiration
    cache.set("expire_key", "data", ttl=0.1)
    import time
    time.sleep(0.2)
    expired = cache.get("expire_key")
    assert expired is None

    return cache


@test_step("Test environment configuration detection")
def test_environment_config():
    """Test environment variable configuration."""
    # Test USE_CLAUDE_CODE
    os.environ["USE_CLAUDE_CODE"] = "true"
    orch = AgentOrchestrator()
    assert orch._use_claude_code == True

    os.environ["USE_CLAUDE_CODE"] = "false"
    orch = AgentOrchestrator()
    assert orch._use_claude_code == False

    return True


@test_step("Test all agent roles initialization")
def test_roles():
    """Test agent role initialization."""
    from src.roles import Planner, Coder, Validator, Reviewer, Reflector

    mock_orchestrator = MagicMock()

    roles = [
        Planner(mock_orchestrator),
        Coder(mock_orchestrator),
        Validator(mock_orchestrator),
        Reviewer(mock_orchestrator),
        Reflector(mock_orchestrator),
    ]

    for role in roles:
        assert role is not None

    return roles


@test_step("Test thread safety utilities")
def test_concurrency():
    """Test concurrency utilities."""
    from src.utils.concurrency import thread_safe_operation

    counter = {"value": 0}

    @thread_safe_operation
    def increment():
        current = counter["value"]
        counter["value"] = current + 1

    # Run multiple times
    for _ in range(10):
        increment()

    assert counter["value"] == 10
    return True


@test_step("Test retry with exponential backoff")
def test_retry():
    """Test retry logic."""
    from src.utils.retry import retry_with_exponential_backoff

    attempt_count = {"value": 0}

    @retry_with_exponential_backoff(max_retries=3, base_delay=0.01)
    def flaky_function():
        attempt_count["value"] += 1
        if attempt_count["value"] < 3:
            raise Exception("Temporary failure")
        return "Success"

    result = flaky_function()
    assert result == "Success"
    assert attempt_count["value"] == 3

    return True


@test_step("Test input validation")
def test_validation():
    """Test input validation."""
    from src.utils.validation import validate_input
    from src.exceptions import ValidationException

    # Valid input
    valid = validate_input("Test prompt", max_length=1000)
    assert valid == "Test prompt"

    # Test length limit
    try:
        validate_input("x" * 2000, max_length=1000)
        raise AssertionError("Should have raised ValidationException")
    except ValidationException:
        pass  # Expected

    return True


@test_step("Test exception handling")
def test_exceptions():
    """Test custom exceptions."""
    from src.exceptions import ModelException, ValidationException

    # Test ModelException
    try:
        raise ModelException("Test", provider="claude", model="sonnet")
    except ModelException as e:
        assert e.provider == "claude"
        assert e.model == "sonnet"

    # Test ValidationException
    try:
        raise ValidationException("Invalid", field="test")
    except ValidationException as e:
        assert e.field == "test"

    return True


@test_step("Test governance system")
def test_governance():
    """Test governance checks."""
    from src.governance import Governance

    gov = Governance({
        "max_validations": 3,
        "max_reflections": 2,
        "min_confidence": 0.7
    })

    # Test validation count
    for i in range(3):
        gov.check({"validation_count": i, "confidence": 0.8})

    # Should flag after max validations
    result = gov.check({"validation_count": 4, "confidence": 0.8})
    assert result.get("action") is not None

    return gov


@test_step("Test HITL manager")
def test_hitl():
    """Test Human-in-the-loop manager."""
    from src.utils.hitl import HITLManager

    hitl = HITLManager({
        "threshold": 0.8,
        "auto_approve_low_risk": True
    })

    # Test low risk auto-approval
    result = hitl.check_approval("low_risk_task", risk_level="low")
    assert result["approved"] == True

    # Test high risk flagging
    result = hitl.check_approval("high_risk_task", risk_level="high")
    assert result["requires_human"] == True

    return hitl


@test_step("Test model routing with Claude Code priority")
def test_model_routing():
    """Test that Claude models are properly routed."""
    orchestrator = AgentOrchestrator()

    # Check model aliases
    assert "claude_sonnet_4" in orchestrator.model_aliases
    assert "claude_opus_4" in orchestrator.model_aliases

    # Check routing configuration
    model = orchestrator._get_model_for_role("Planner")
    assert "claude" in model.lower() or "gpt" in model.lower()

    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("🔧 ENTERPRISE AGENT FINAL INTEGRATION TEST")
    print("="*60 + "\n")

    # Run all tests
    test_imports()
    test_orchestrator_init()
    test_claude_provider()
    test_auth_manager()
    test_cost_tracking()
    test_memory_store()
    test_cache()
    test_environment_config()
    test_roles()
    test_concurrency()
    test_retry()
    test_validation()
    test_exceptions()
    test_governance()
    test_hitl()
    test_model_routing()

    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {test_results['passed']}")
    print(f"❌ Failed: {test_results['failed']}")

    if test_results["errors"]:
        print("\n⚠️ Errors:")
        for error in test_results["errors"]:
            print(f"  - {error}")

    if test_results["failed"] == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n😞 {test_results['failed']} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())