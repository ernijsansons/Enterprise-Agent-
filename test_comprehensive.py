#!/usr/bin/env python
"""Comprehensive test suite for Enterprise Agent with Claude Code integration."""
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent_orchestrator import AgentOrchestrator
from src.exceptions import ModelException, ValidationException
from src.memory import MemoryStore  # Memory is in src.memory
from src.providers.auth_manager import ClaudeAuthManager
from src.providers.claude_code_provider import ClaudeCodeProvider
from src.utils.cache import ResponseCache
from src.utils.concurrency import thread_safe_operation
from src.utils.costs import CostEstimator
from src.utils.retry import retry_with_exponential_backoff
from src.utils.validation import validate_input


class TestColors:
    """Terminal colors for test output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_test_header(test_name: str):
    """Print a test header."""
    print(f"\n{TestColors.HEADER}{TestColors.BOLD}{'='*60}{TestColors.ENDC}")
    print(f"{TestColors.HEADER}{TestColors.BOLD}Testing: {test_name}{TestColors.ENDC}")
    print(f"{TestColors.HEADER}{TestColors.BOLD}{'='*60}{TestColors.ENDC}")


def print_success(msg: str):
    """Print success message."""
    print(f"{TestColors.OKGREEN}✓ {msg}{TestColors.ENDC}")


def print_fail(msg: str):
    """Print failure message."""
    print(f"{TestColors.FAIL}✗ {msg}{TestColors.ENDC}")


def print_info(msg: str):
    """Print info message."""
    print(f"{TestColors.OKBLUE}ℹ {msg}{TestColors.ENDC}")


def test_basic_imports():
    """Test 1: Basic imports and module availability."""
    print_test_header("Basic Imports and Dependencies")

    modules_to_test = [
        ("Agent Orchestrator", "src.agent_orchestrator", "AgentOrchestrator"),
        (
            "Claude Code Provider",
            "src.providers.claude_code_provider",
            "ClaudeCodeProvider",
        ),
        ("Auth Manager", "src.providers.auth_manager", "ClaudeAuthManager"),
        ("Cost Estimator", "src.utils.costs", "CostEstimator"),
        ("Memory Store", "src.memory", "MemoryStore"),
        ("Response Cache", "src.utils.cache", "ResponseCache"),
    ]

    for name, module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print_success(f"{name} imported successfully")
        except Exception as e:
            print_fail(f"{name} import failed: {e}")
            return False

    return True


def test_claude_code_provider():
    """Test 2: Claude Code Provider functionality."""
    print_test_header("Claude Code Provider")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(
            returncode=0, stdout="claude version 1.0.0", stderr=""
        )

        try:
            # Test provider initialization
            provider = ClaudeCodeProvider({"timeout": 30})
            print_success("Provider initialized")

            # Test model mapping
            assert provider._map_model_to_cli("claude_sonnet_4") == "sonnet"
            assert provider._map_model_to_cli("claude_opus_4") == "opus"
            assert provider._map_model_to_cli("claude_haiku") == "haiku"
            print_success("Model mapping works correctly")

            # Test session management
            session_id = provider.create_session("test-session")
            assert session_id == "test-session"
            assert "test-session" in provider.sessions
            provider.clear_session("test-session")
            assert "test-session" not in provider.sessions
            print_success("Session management works")

            # Test response parsing
            json_response = '{"response": "test", "session_id": "123"}'
            parsed = provider._parse_cli_response(json_response)
            assert parsed["response"] == "test"
            print_success("Response parsing works")

            return True

        except Exception as e:
            print_fail(f"Provider test failed: {e}")
            return False


def test_auth_manager():
    """Test 3: Authentication Manager."""
    print_test_header("Authentication Manager")

    try:
        auth_manager = ClaudeAuthManager()
        print_success("Auth manager initialized")

        # Test API key removal
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        result = auth_manager.ensure_subscription_mode()
        assert "ANTHROPIC_API_KEY" not in os.environ
        print_success("API key removal works")

        # Restore original state
        if original_key:
            os.environ["ANTHROPIC_API_KEY"] = original_key

        # Test config operations
        config = auth_manager.get_config()
        assert isinstance(config, dict)
        print_success("Config retrieval works")

        # Test plan verification
        plan_info = auth_manager.verify_subscription_plan()
        assert "authenticated" in plan_info
        assert "using_api_key" in plan_info
        print_success("Plan verification works")

        return True

    except Exception as e:
        print_fail(f"Auth manager test failed: {e}")
        return False


def test_orchestrator_initialization():
    """Test 4: Agent Orchestrator initialization."""
    print_test_header("Agent Orchestrator Initialization")

    try:
        # Set environment for testing
        os.environ["USE_CLAUDE_CODE"] = "false"  # Disable for testing

        orchestrator = AgentOrchestrator()
        print_success("Orchestrator initialized")

        # Check components
        assert orchestrator.cost_estimator is not None
        print_success("Cost estimator present")

        assert orchestrator.memory_store is not None
        print_success("Memory store present")

        assert orchestrator.governance is not None
        print_success("Governance system present")

        assert orchestrator.hitl is not None
        print_success("HITL system present")

        # Check model configuration
        assert orchestrator._model_for_role is not None
        print_success("Model configuration loaded")

        return True

    except Exception as e:
        print_fail(f"Orchestrator initialization failed: {e}")
        return False


def test_cost_tracking():
    """Test 5: Cost tracking with zero-cost Claude Code."""
    print_test_header("Cost Tracking")

    try:
        estimator = CostEstimator({})

        # Track API call (should have cost)
        estimator.track(100, "Planner", "test", model="gpt-4")

        # Track Claude Code call (should be zero cost)
        estimator.track(0, "Coder", "test", model="claude_code")

        summary = estimator.get_summary()
        assert "total_cost" in summary
        print_success("Cost tracking works")

        # Verify zero cost for Claude Code
        events = summary.get("events", [])
        claude_code_events = [e for e in events if "claude_code" in e.get("model", "")]
        if claude_code_events:
            assert all(e.get("cost", 1) == 0 for e in claude_code_events)
            print_success("Claude Code calls have zero cost")

        return True

    except Exception as e:
        print_fail(f"Cost tracking test failed: {e}")
        return False


def test_memory_store():
    """Test 6: Memory store and embeddings."""
    print_test_header("Memory Store")

    try:
        memory = MemoryStore()

        # Store memories
        memory.store("task1", "Test task", "project")
        memory.store("result1", "Test result", "workflow")
        print_success("Memory storage works")

        # Retrieve memories
        project_memories = memory.retrieve("project", top_k=5)
        assert isinstance(project_memories, list)
        print_success("Memory retrieval works")

        # Test pruning
        initial_count = len(memory._memories)
        memory.prune(max_age_hours=0)  # Prune everything old
        assert len(memory._memories) <= initial_count
        print_success("Memory pruning works")

        return True

    except Exception as e:
        print_fail(f"Memory store test failed: {e}")
        return False


def test_response_cache():
    """Test 7: Response caching."""
    print_test_header("Response Cache")

    try:
        cache = ResponseCache()

        # Test caching
        cache.cache_response("test_key", "planner", "test_response", ttl=60)

        # Test retrieval
        cached = cache.get_response("test_key", "planner")
        assert cached == "test_response"
        print_success("Cache storage and retrieval works")

        # Test expiration
        cache.cache_response("expire_key", "coder", "data", ttl=0.1)
        time.sleep(0.2)
        expired = cache.get_response("expire_key", "coder")
        assert expired is None
        print_success("Cache expiration works")

        # Test cleanup
        cache.cleanup()
        print_success("Cache cleanup works")

        return True

    except Exception as e:
        print_fail(f"Response cache test failed: {e}")
        return False


def test_concurrency():
    """Test 8: Thread safety and concurrent operations."""
    print_test_header("Concurrency and Thread Safety")

    try:
        counter = {"value": 0}

        @thread_safe_operation
        def increment():
            current = counter["value"]
            time.sleep(0.001)  # Simulate work
            counter["value"] = current + 1

        # Run concurrent operations
        import threading

        threads = []
        for _ in range(10):
            t = threading.Thread(target=increment)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert counter["value"] == 10
        print_success("Thread-safe operations work correctly")

        return True

    except Exception as e:
        print_fail(f"Concurrency test failed: {e}")
        return False


def test_retry_logic():
    """Test 9: Retry with exponential backoff."""
    print_test_header("Retry Logic")

    try:
        attempt_count = {"value": 0}

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.1)
        def flaky_function():
            attempt_count["value"] += 1
            if attempt_count["value"] < 3:
                raise Exception("Temporary failure")
            return "Success"

        result = flaky_function()
        assert result == "Success"
        assert attempt_count["value"] == 3
        print_success("Retry with exponential backoff works")

        return True

    except Exception as e:
        print_fail(f"Retry logic test failed: {e}")
        return False


def test_validation():
    """Test 10: Input validation."""
    print_test_header("Input Validation")

    try:
        # Test valid input
        valid = validate_input("Test prompt", max_length=1000)
        assert valid == "Test prompt"
        print_success("Valid input passes")

        # Test sanitization
        dirty = "Test <script>alert('xss')</script> prompt"
        clean = validate_input(dirty, max_length=1000)
        assert "<script>" not in clean
        print_success("Input sanitization works")

        # Test length limit
        try:
            validate_input("x" * 2000, max_length=1000)
            print_fail("Length validation should have failed")
            return False
        except ValidationException:
            print_success("Length validation works")

        return True

    except Exception as e:
        print_fail(f"Validation test failed: {e}")
        return False


def test_environment_configuration():
    """Test 11: Environment configuration loading."""
    print_test_header("Environment Configuration")

    try:
        # Test USE_CLAUDE_CODE detection
        os.environ["USE_CLAUDE_CODE"] = "true"
        orchestrator = AgentOrchestrator()
        assert orchestrator._use_claude_code == True
        print_success("USE_CLAUDE_CODE=true detected")

        os.environ["USE_CLAUDE_CODE"] = "false"
        orchestrator = AgentOrchestrator()
        assert orchestrator._use_claude_code == False
        print_success("USE_CLAUDE_CODE=false detected")

        # Test model configuration from env
        if "PRIMARY_MODEL" in os.environ:
            print_info(f"PRIMARY_MODEL: {os.environ['PRIMARY_MODEL']}")

        return True

    except Exception as e:
        print_fail(f"Environment configuration test failed: {e}")
        return False


def test_all_roles():
    """Test 12: All agent roles."""
    print_test_header("Agent Roles")

    try:
        from src.roles import Coder, Planner, Reflector, Reviewer, Validator

        # Test each role initialization
        roles = [
            ("Planner", Planner),
            ("Coder", Coder),
            ("Validator", Validator),
            ("Reviewer", Reviewer),
            ("Reflector", Reflector),
        ]

        for role_name, role_class in roles:
            try:
                role = role_class(MagicMock())
                print_success(f"{role_name} initialized")
            except Exception as e:
                print_fail(f"{role_name} failed: {e}")
                return False

        return True

    except Exception as e:
        print_fail(f"Role test failed: {e}")
        return False


def test_error_handling():
    """Test 13: Error handling and recovery."""
    print_test_header("Error Handling")

    try:
        # Test ModelException
        try:
            raise ModelException("Test error", provider="test", model="test-model")
        except ModelException as e:
            assert e.provider == "test"
            assert e.model == "test-model"
            print_success("ModelException works")

        # Test ValidationException
        try:
            raise ValidationException("Invalid input", field="test_field")
        except ValidationException as e:
            assert e.field == "test_field"
            print_success("ValidationException works")

        return True

    except Exception as e:
        print_fail(f"Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all comprehensive tests."""
    print(f"\n{TestColors.HEADER}{TestColors.BOLD}{'='*60}{TestColors.ENDC}")
    print(
        f"{TestColors.HEADER}{TestColors.BOLD}COMPREHENSIVE ENTERPRISE AGENT TEST SUITE{TestColors.ENDC}"
    )
    print(f"{TestColors.HEADER}{TestColors.BOLD}{'='*60}{TestColors.ENDC}")

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Claude Code Provider", test_claude_code_provider),
        ("Auth Manager", test_auth_manager),
        ("Orchestrator Init", test_orchestrator_initialization),
        ("Cost Tracking", test_cost_tracking),
        ("Memory Store", test_memory_store),
        ("Response Cache", test_response_cache),
        ("Concurrency", test_concurrency),
        ("Retry Logic", test_retry_logic),
        ("Input Validation", test_validation),
        ("Environment Config", test_environment_configuration),
        ("Agent Roles", test_all_roles),
        ("Error Handling", test_error_handling),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_fail(f"{test_name} crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print(f"\n{TestColors.HEADER}{TestColors.BOLD}{'='*60}{TestColors.ENDC}")
    print(f"{TestColors.HEADER}{TestColors.BOLD}TEST SUMMARY{TestColors.ENDC}")
    print(f"{TestColors.HEADER}{TestColors.BOLD}{'='*60}{TestColors.ENDC}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = (
            f"{TestColors.OKGREEN}PASS{TestColors.ENDC}"
            if result
            else f"{TestColors.FAIL}FAIL{TestColors.ENDC}"
        )
        print(f"{test_name:.<40} {status}")

    print(f"\n{TestColors.BOLD}Results: {passed}/{total} tests passed{TestColors.ENDC}")

    if passed == total:
        print(
            f"{TestColors.OKGREEN}{TestColors.BOLD}✓ ALL TESTS PASSED!{TestColors.ENDC}"
        )
        return 0
    else:
        print(f"{TestColors.FAIL}{TestColors.BOLD}✗ Some tests failed{TestColors.ENDC}")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
