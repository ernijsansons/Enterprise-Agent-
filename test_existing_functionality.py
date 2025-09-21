#!/usr/bin/env python
"""Test existing functionality in Enterprise Agent."""
import io
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "=" * 60)
print("TESTING EXISTING ENTERPRISE AGENT FUNCTIONALITY")
print("=" * 60 + "\n")


def test_with_status(name):
    """Decorator for test functions."""

    def decorator(func):
        def wrapper():
            try:
                print(f"Testing {name}...", end=" ")
                func()
                print("✅ PASS")
                return True
            except Exception as e:
                print(f"❌ FAIL: {e}")
                return False

        return wrapper

    return decorator


@test_with_status("Core imports")
def test_imports():
    """Test core imports."""


@test_with_status("Agent Orchestrator")
def test_orchestrator():
    """Test orchestrator initialization."""
    os.environ["USE_CLAUDE_CODE"] = "false"
    from src.agent_orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    assert orchestrator.cost_estimator is not None


@test_with_status("Claude Code Provider")
def test_claude_provider():
    """Test Claude Code provider."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(
            returncode=0, stdout="claude version 1.0.0", stderr=""
        )
        from src.providers.claude_code_provider import ClaudeCodeProvider

        provider = ClaudeCodeProvider({"timeout": 30})
        assert provider._map_model_to_cli("claude_sonnet_4") == "sonnet"


@test_with_status("Authentication Manager")
def test_auth_manager():
    """Test auth manager."""
    from src.providers.auth_manager import ClaudeAuthManager

    auth = ClaudeAuthManager()
    config = auth.get_config()
    assert isinstance(config, dict)


@test_with_status("Cost Estimator")
def test_cost_estimator():
    """Test cost tracking."""
    from src.utils.costs import CostEstimator

    estimator = CostEstimator({})
    estimator.track(100, "Planner", "test", model="gpt-4")
    estimator.track(0, "Coder", "test", model="claude_code")
    # Get current cost
    assert estimator.total_cost >= 0


@test_with_status("Memory Store")
def test_memory():
    """Test memory store."""
    from src.memory import MemoryStore

    memory = MemoryStore({})  # Pass empty config
    memory.store("key", "value", "project")
    memories = memory.retrieve("project", top_k=5)
    assert isinstance(memories, list)


@test_with_status("TTL Cache")
def test_cache():
    """Test caching."""
    from src.utils.cache import TTLCache

    cache = TTLCache()
    cache.set("key", "value")
    assert cache.get("key") == "value"


@test_with_status("Agent Roles")
def test_roles():
    """Test all roles."""
    from unittest.mock import MagicMock

    from src.roles import Coder, Planner, Reflector, Reviewer, Validator

    mock_orch = MagicMock()

    roles = [
        Planner(mock_orch),
        Coder(mock_orch),
        Validator(mock_orch),
        Reviewer(mock_orch),
        Reflector(mock_orch),
    ]

    for role in roles:
        assert role is not None


@test_with_status("Environment Variables")
def test_env_vars():
    """Test environment configuration."""
    os.environ["USE_CLAUDE_CODE"] = "true"
    from src.agent_orchestrator import AgentOrchestrator

    orch = AgentOrchestrator()
    assert orch._use_claude_code is True

    os.environ["USE_CLAUDE_CODE"] = "false"
    orch = AgentOrchestrator()
    assert orch._use_claude_code is False


@test_with_status("Claude CLI wrapper")
def test_cli_wrapper():
    """Test CLI wrapper utilities."""
    from src.utils.claude_cli import ClaudeCommand, OutputFormat

    cmd = ClaudeCommand(prompt="test", model="sonnet", output_format=OutputFormat.JSON)

    command_list = cmd.build_command()
    assert "claude" in command_list
    assert "--model" in command_list
    assert "sonnet" in command_list


@test_with_status("Exception classes")
def test_exceptions():
    """Test custom exceptions."""
    from src.exceptions import AgentException

    try:
        raise AgentException("Test error", stage="testing")
    except AgentException as e:
        assert e.stage == "testing"
        assert str(e) == "Test error"


@test_with_status("Concurrency utilities")
def test_concurrency():
    """Test concurrency helpers."""
    from src.utils.concurrency import ThreadSafeDict

    safe_dict = ThreadSafeDict()
    safe_dict["key"] = "value"
    assert safe_dict["key"] == "value"


@test_with_status("Retry utilities")
def test_retry():
    """Test retry functionality."""
    from src.utils.retry import ExponentialBackoff

    backoff = ExponentialBackoff(base_delay=0.1, max_delay=10, factor=2)

    delay1 = backoff.get_delay(1)
    delay2 = backoff.get_delay(2)
    assert delay2 > delay1


@test_with_status("Validation utilities")
def test_validation():
    """Test input validation."""
    from src.utils.validation import ValidationResult

    result = ValidationResult(
        is_valid=True, message="Valid input", sanitized_value="test"
    )
    assert result.is_valid is True


@test_with_status("HITL Orchestrator")
def test_hitl():
    """Test HITL functionality."""
    from src.utils.hitl import HITLOrchestrator

    hitl = HITLOrchestrator({"threshold": 0.8, "auto_approve_low_risk": True})

    decision = hitl.check("test_task", confidence=0.9)
    assert decision is not None


# Run all tests
if __name__ == "__main__":
    tests = [
        test_imports,
        test_orchestrator,
        test_claude_provider,
        test_auth_manager,
        test_cost_estimator,
        test_memory,
        test_cache,
        test_roles,
        test_env_vars,
        test_cli_wrapper,
        test_exceptions,
        test_concurrency,
        test_retry,
        test_validation,
        test_hitl,
    ]

    passed = 0
    failed = 0

    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {failed} tests failed")
        sys.exit(1)
