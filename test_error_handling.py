#!/usr/bin/env python3
"""Test script to validate structured error handling system."""
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_error_code_classification():
    """Test that error codes are properly classified."""
    print("Testing error code classification...")

    try:
        from src.utils.errors import (
            EnterpriseAgentError,
            ErrorCategory,
            ErrorCode,
            ErrorSeverity,
        )

        # Test orchestration error classification
        orch_error = EnterpriseAgentError(
            ErrorCode.ORCHESTRATION_INIT_FAILED, "Test orchestration error"
        )

        assert orch_error.details.category == ErrorCategory.ORCHESTRATION
        assert orch_error.details.severity == ErrorSeverity.CRITICAL
        assert orch_error.details.code == ErrorCode.ORCHESTRATION_INIT_FAILED

        # Test model error classification
        model_error = EnterpriseAgentError(
            ErrorCode.MODEL_CALL_FAILED, "Test model error"
        )

        assert model_error.details.category == ErrorCategory.MODEL_CALL
        assert model_error.details.severity == ErrorSeverity.MEDIUM

        # Test validation error classification
        validation_error = EnterpriseAgentError(
            ErrorCode.VALIDATION_FAILED, "Test validation error"
        )

        assert validation_error.details.category == ErrorCategory.VALIDATION
        assert validation_error.details.severity == ErrorSeverity.MEDIUM

        print("✅ Error code classification test passed")
        return True

    except Exception as e:
        print(f"❌ Error code classification test failed: {e}")
        return False


def test_error_handler():
    """Test the error handler functionality."""
    print("\nTesting error handler...")

    try:
        from src.utils.errors import EnterpriseAgentError, ErrorCode, ErrorHandler

        handler = ErrorHandler()

        # Create test errors
        test_error1 = EnterpriseAgentError(
            ErrorCode.MODEL_CALL_FAILED,
            "Test model error",
            context={"model": "claude-3", "role": "coder"},
        )

        test_error2 = EnterpriseAgentError(
            ErrorCode.VALIDATION_FAILED,
            "Test validation error",
            context={"coverage": 0.5},
        )

        # Handle errors
        handler.handle_error(test_error1)
        handler.handle_error(test_error2)

        # Check that errors are tracked
        assert len(handler.error_history) == 2
        assert handler.error_counts[ErrorCode.MODEL_CALL_FAILED] == 1
        assert handler.error_counts[ErrorCode.VALIDATION_FAILED] == 1

        # Test error summary
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 2
        assert "model_call" in summary["category_breakdown"]
        assert "validation" in summary["category_breakdown"]

        print("✅ Error handler test passed")
        return True

    except Exception as e:
        print(f"❌ Error handler test failed: {e}")
        return False


def test_orchestrator_error_integration():
    """Test error handling integration with orchestrator."""
    print("\nTesting orchestrator error integration...")

    try:
        # Create a test config file that will cause a configuration error
        test_config = """
invalid_yaml_structure:
  - missing enterprise_coding_agent section
  - this should cause: a configuration error
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(test_config)
            config_path = f.name

        from src.agent_orchestrator import AgentOrchestrator
        from src.utils.errors import EnterpriseAgentError, ErrorCode

        # This should raise a configuration error
        try:
            orchestrator = AgentOrchestrator(config_path)
            print("❌ Expected configuration error but none occurred")
            return False
        except EnterpriseAgentError as e:
            assert e.details.code in [
                ErrorCode.CONFIG_MISSING_REQUIRED_FIELD,
                ErrorCode.CONFIG_VALIDATION_FAILED,
                ErrorCode.CONFIG_FILE_NOT_FOUND,
            ]
            print(f"✅ Configuration error properly caught: {e.details.code.name}")

        print("✅ Orchestrator error integration test passed")
        return True

    except Exception as e:
        print(f"❌ Orchestrator error integration test failed: {e}")
        return False

    finally:
        # Cleanup
        if "config_path" in locals():
            try:
                os.unlink(config_path)
            except OSError:
                pass


def test_error_recovery_suggestions():
    """Test that error recovery suggestions are properly provided."""
    print("\nTesting error recovery suggestions...")

    try:
        from src.utils.errors import ErrorCode, create_model_error

        # Create model error with recovery suggestions
        model_error = create_model_error(
            "Rate limit exceeded",
            model="claude-3",
            error_code=ErrorCode.MODEL_RATE_LIMITED,
            context={"requests_per_minute": 100},
        )

        model_error.add_recovery_suggestion("Reduce request frequency")
        model_error.add_recovery_suggestion("Implement exponential backoff")

        assert len(model_error.details.recovery_suggestions) >= 2
        assert "Reduce request frequency" in model_error.details.recovery_suggestions
        assert "exponential backoff" in model_error.details.recovery_suggestions[1]

        # Test JSON serialization
        json_str = model_error.details.to_json()
        assert "recovery_suggestions" in json_str
        assert "rate_limited" in json_str.lower()

        print("✅ Error recovery suggestions test passed")
        return True

    except Exception as e:
        print(f"❌ Error recovery suggestions test failed: {e}")
        return False


def test_error_context_and_metadata():
    """Test error context and metadata handling."""
    print("\nTesting error context and metadata...")

    try:
        from src.utils.errors import EnterpriseAgentError, ErrorCode

        # Create error with rich context
        error = EnterpriseAgentError(
            ErrorCode.REFLECTION_LOOP_FAILED,
            "Reflection loop stagnated",
            context={
                "iteration": 3,
                "confidence": 0.45,
                "max_iterations": 5,
                "stagnation_count": 3,
            },
        )

        error.add_context("domain", "web_development")
        error.add_context("task_complexity", "high")
        error.set_user_message(
            "The reflection process encountered issues. Please review the task requirements."
        )

        # Verify context is properly stored
        assert error.details.context["iteration"] == 3
        assert error.details.context["confidence"] == 0.45
        assert error.details.context["domain"] == "web_development"
        assert error.details.context["task_complexity"] == "high"

        # Verify user message
        assert (
            "reflection process encountered issues"
            in error.details.user_message.lower()
        )

        # Verify metadata
        assert error.details.timestamp > 0
        assert error.details.severity in [
            s.value for s in error.details.severity.__class__
        ]

        print("✅ Error context and metadata test passed")
        return True

    except Exception as e:
        print(f"❌ Error context and metadata test failed: {e}")
        return False


def main():
    """Run all error handling tests."""
    print("Structured Error Handling Test Suite")
    print("=" * 50)

    tests_passed = 0
    total_tests = 5

    if test_error_code_classification():
        tests_passed += 1

    if test_error_handler():
        tests_passed += 1

    if test_orchestrator_error_integration():
        tests_passed += 1

    if test_error_recovery_suggestions():
        tests_passed += 1

    if test_error_context_and_metadata():
        tests_passed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print(
            "🎉 All error handling tests passed! Structured error handling is working correctly."
        )
        return 0
    else:
        print("❌ Some tests failed. Please check the error handling implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
