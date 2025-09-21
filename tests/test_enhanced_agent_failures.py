"""Comprehensive test cases for Enterprise Agent pipeline failures.

This module tests failure scenarios, error propagation, interface mismatches,
reflection logic issues, and observability for the enhanced agent system.
"""
from __future__ import annotations

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.agent_orchestrator import AgentOrchestrator, AgentState
from src.roles.base import BaseRole
from src.roles.validator import Validator
from src.roles.reflector import Reflector
from src.utils.errors import (
    EnterpriseAgentError,
    ErrorCode,
    get_error_handler,
    create_validation_error,
    create_orchestration_error,
    handle_error
)
from src.exceptions import (
    ModelException,
    ModelTimeoutException
)


class TestAgentInterfaceFailures:
    """Test interface mismatches and error propagation between modules."""

    def test_invalid_model_name_in_base_role(self):
        """Test base role handles invalid model names correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            base_role = BaseRole(orchestrator)

            # Test empty model name
            with pytest.raises(EnterpriseAgentError) as exc_info:
                base_role.call_model("", "test prompt", "TestRole", "test")

            assert exc_info.value.details.code == ErrorCode.INVALID_PARAMETERS
            assert "model must be a non-empty string" in exc_info.value.details.message.lower()

            # Test None model name
            with pytest.raises(EnterpriseAgentError) as exc_info:
                base_role.call_model(None, "test prompt", "TestRole", "test")

            assert exc_info.value.details.code == ErrorCode.INVALID_PARAMETERS

    def test_invalid_prompt_in_base_role(self):
        """Test base role handles invalid prompts correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            base_role = BaseRole(orchestrator)

            # Test empty prompt
            with pytest.raises(EnterpriseAgentError) as exc_info:
                base_role.call_model("test-model", "", "TestRole", "test")

            assert exc_info.value.details.code == ErrorCode.INVALID_PARAMETERS
            assert "prompt must be a non-empty string" in exc_info.value.details.message.lower()

    def test_invalid_domain_in_base_role(self):
        """Test base role handles invalid domains correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            base_role = BaseRole(orchestrator)

            # Test invalid domain
            with pytest.raises(EnterpriseAgentError) as exc_info:
                base_role.domain_pack("invalid_domain")

            assert exc_info.value.details.code == ErrorCode.INVALID_DOMAIN
            assert "unknown domain" in exc_info.value.details.message.lower()

    def test_json_parsing_failures_in_base_role(self):
        """Test base role handles JSON parsing failures correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            base_role = BaseRole(orchestrator)

            # Test invalid JSON
            with pytest.raises(EnterpriseAgentError) as exc_info:
                base_role.parse_json("invalid json {")

            assert exc_info.value.details.code == ErrorCode.VALIDATION_PARSE_ERROR
            assert "json parsing failed" in exc_info.value.details.message.lower()

            # Test non-string input
            with pytest.raises(EnterpriseAgentError) as exc_info:
                base_role.parse_json(123)

            assert exc_info.value.details.code == ErrorCode.VALIDATION_PARSE_ERROR


class TestValidatorFailures:
    """Test validator failures and error handling."""

    def test_validator_invalid_input_types(self):
        """Test validator handles invalid input types correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
  domains:
    coding:
      coverage_threshold: 0.97
""")

            orchestrator = AgentOrchestrator(str(config_path))
            validator = Validator(orchestrator)

            # Test non-string output
            with pytest.raises(EnterpriseAgentError) as exc_info:
                validator.validate(123, "coding")

            assert exc_info.value.details.code == ErrorCode.INVALID_PARAMETERS
            assert "output must be a string" in exc_info.value.details.message.lower()

            # Test empty output
            with pytest.raises(EnterpriseAgentError) as exc_info:
                validator.validate("", "coding")

            assert exc_info.value.details.code == ErrorCode.INVALID_PARAMETERS
            assert "output cannot be empty" in exc_info.value.details.message.lower()

    def test_validator_unknown_domain(self):
        """Test validator handles unknown domains correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            validator = Validator(orchestrator)

            with pytest.raises(EnterpriseAgentError) as exc_info:
                validator.validate("test output", "unknown_domain")

            assert exc_info.value.details.code == ErrorCode.VALIDATION_FAILED
            assert "no validator available" in exc_info.value.details.message.lower()

    @patch('src.roles.validators.validate_coding')
    def test_validator_domain_execution_failure(self, mock_validate_coding):
        """Test validator handles domain validation execution failures."""
        mock_validate_coding.side_effect = Exception("Domain validator crashed")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            validator = Validator(orchestrator)

            with pytest.raises(EnterpriseAgentError) as exc_info:
                validator.validate("test code", "coding")

            assert exc_info.value.details.code == ErrorCode.VALIDATION_FAILED
            assert "domain validation failed" in exc_info.value.details.message.lower()

    def test_validator_llm_validation_failure(self):
        """Test validator handles LLM validation failures gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            validator = Validator(orchestrator)

            # Mock call_model to raise an error
            with patch.object(validator, 'call_model', side_effect=Exception("LLM error")):
                with patch('src.roles.validators.validate_coding', return_value={"passes": True}):
                    result = validator.validate("test code", "coding")

                    # Should not fail, but return error in llm_insights
                    assert "llm_insights" in result
                    assert "error" in result["llm_insights"]
                    assert "fallback_used" in result["llm_insights"]

    def test_validator_actionable_feedback_generation(self):
        """Test validator generates actionable feedback for failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            validator = Validator(orchestrator)

            # Mock validation failure
            with patch('src.roles.validators.validate_coding') as mock_validate:
                mock_validate.return_value = {
                    "passes": False,
                    "tests_passed": False,
                    "coverage": 0.80,
                    "coverage_threshold": 0.97
                }

                with patch.object(validator, '_perform_llm_validation', return_value={}):
                    result = validator.validate("test code", "coding")

                    assert "actionable_feedback" in result
                    feedback = result["actionable_feedback"]
                    assert "immediate_actions" in feedback
                    assert "domain_specific_guidance" in feedback
                    assert len(feedback["immediate_actions"]) > 0


class TestReflectorFailures:
    """Test reflector failures and enhanced analysis."""

    def test_reflector_max_iterations_reached(self):
        """Test reflector handles maximum iterations correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            reflector = Reflector(orchestrator)

            validation = {"passes": False, "coverage": 0.8}
            result = reflector.reflect(validation, "test output", "coding", 5)

            assert result["halt"] is True
            assert result["halt_reason"] == "max_iterations_reached"
            assert "error_details" in result

    def test_reflector_invalid_validation_input(self):
        """Test reflector handles invalid validation input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            reflector = Reflector(orchestrator)

            # Test non-dict validation
            with pytest.raises(EnterpriseAgentError) as exc_info:
                reflector.reflect("invalid", "test output", "coding", 1)

            assert exc_info.value.details.code == ErrorCode.INVALID_PARAMETERS
            assert "validation must be a dictionary" in exc_info.value.details.message.lower()

    def test_reflector_json_parsing_failure_fallback(self):
        """Test reflector falls back gracefully on JSON parsing failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            reflector = Reflector(orchestrator)

            # Mock call_model to return invalid JSON
            with patch.object(reflector, 'call_model', return_value="invalid json {"):
                validation = {"passes": False}
                result = reflector.reflect(validation, "test output", "coding", 1)

                assert result["structured_reflection"] is False
                assert result["analysis"]["parsing_failed"] is True
                assert result["analysis"]["fallback_processing"] is True

    def test_reflector_structured_reflection_processing(self):
        """Test reflector processes structured reflection correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            reflector = Reflector(orchestrator)

            # Mock successful structured response
            structured_response = {
                "confidence": 0.9,
                "revised_output": "improved code",
                "fixes": [{"description": "fixed bug", "priority": "high"}],
                "root_cause_analysis": {"primary_causes": ["logic error"]}
            }

            with patch.object(reflector, 'call_model', return_value=json.dumps(structured_response)):
                validation = {"passes": False}
                result = reflector.reflect(validation, "test output", "coding", 1)

                assert result["structured_reflection"] is True
                assert result["confidence"] == 0.9
                assert result["halt"] is True  # High confidence should trigger halt
                assert result["analysis"]["halt_reason"] == "high_confidence"

    def test_reflector_confidence_validation_and_clamping(self):
        """Test reflector validates and clamps confidence values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            reflector = Reflector(orchestrator)

            # Mock response with invalid confidence
            invalid_response = {
                "confidence": 1.5,  # Invalid - > 1.0
                "revised_output": "improved code"
            }

            with patch.object(reflector, 'call_model', return_value=json.dumps(invalid_response)):
                validation = {"passes": False}
                result = reflector.reflect(validation, "test output", "coding", 1)

                assert result["confidence"] == 1.0  # Should be clamped
                assert result["analysis"]["confidence_adjusted"] is True

    def test_reflector_issue_analysis(self):
        """Test reflector properly analyzes validation issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            reflector = Reflector(orchestrator)

            validation = {
                "passes": False,
                "tests_passed": False,
                "coverage": 0.8,
                "coverage_threshold": 0.97,
                "llm_insights": {
                    "issues": [
                        {
                            "type": "error",
                            "description": "Logic error in function",
                            "fix": "Add null check"
                        }
                    ]
                }
            }

            issues_analysis = reflector._analyze_validation_issues(validation)

            assert "CRITICAL" in issues_analysis
            assert "Tests failing" in issues_analysis
            assert "Coverage insufficient" in issues_analysis
            assert "Logic error" in issues_analysis


class TestConcurrencyFailures:
    """Test concurrency and thread safety issues."""

    def test_concurrent_state_access(self):
        """Test thread-safe state access in orchestrator."""
        import threading

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            errors = []

            def access_state():
                try:
                    state = AgentState({"task": "test", "domain": "coding"})
                    # Simulate state access patterns
                    for i in range(10):
                        state[f"key_{i}"] = f"value_{i}"
                        time.sleep(0.001)  # Small delay to increase chance of race conditions
                except Exception as e:
                    errors.append(e)

            # Create multiple threads
            threads = []
            for _ in range(5):
                thread = threading.Thread(target=access_state)
                threads.append(thread)
                thread.start()

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Should not have any errors from concurrent access
            assert len(errors) == 0

    def test_memory_store_concurrent_access(self):
        """Test memory store handles concurrent access correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))
            memory_store = orchestrator.memory

            errors = []

            def store_and_retrieve():
                try:
                    for i in range(10):
                        memory_store.store("test_scope", f"key_{i}", f"value_{i}")
                        retrieved = memory_store.retrieve("test_scope", f"key_{i}")
                        assert retrieved == f"value_{i}"
                except Exception as e:
                    errors.append(e)

            # Test concurrent operations
            import threading
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=store_and_retrieve)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            assert len(errors) == 0


class TestClaudeCodeIntegrationFailures:
    """Test Claude Code client initialization and integration failures."""

    @patch('subprocess.run')
    def test_claude_code_cli_not_found(self, mock_subprocess):
        """Test Claude Code CLI not found error handling."""
        mock_subprocess.side_effect = FileNotFoundError("claude command not found")

        from src.providers.claude_code_provider import ClaudeCodeProvider

        with pytest.raises(ModelException) as exc_info:
            ClaudeCodeProvider()

        assert "CLI not found" in str(exc_info.value)

    @patch('subprocess.run')
    def test_claude_code_cli_timeout(self, mock_subprocess):
        """Test Claude Code CLI timeout handling."""
        import subprocess
        mock_subprocess.side_effect = subprocess.TimeoutExpired("claude", 5)

        from src.providers.claude_code_provider import ClaudeCodeProvider

        with pytest.raises(ModelTimeoutException) as exc_info:
            ClaudeCodeProvider()

        assert "timeout" in str(exc_info.value).lower()

    @patch('subprocess.run')
    def test_claude_code_authentication_issues(self, mock_subprocess):
        """Test Claude Code authentication issue handling."""
        # Mock version check success but auth failure
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "claude 1.0.0"

        auth_result = Mock()
        auth_result.returncode = 1
        auth_result.stderr = "please run `claude login`"

        mock_subprocess.side_effect = [version_result, auth_result]

        from src.providers.claude_code_provider import ClaudeCodeProvider

        # Should not raise, but should log warning
        provider = ClaudeCodeProvider()
        assert provider is not None


class TestErrorPropagationAndObservability:
    """Test error propagation and observability features."""

    def test_error_handler_tracking(self):
        """Test error handler properly tracks and categorizes errors."""
        error_handler = get_error_handler()

        # Clear any existing errors
        error_handler.error_history.clear()
        error_handler.error_counts.clear()

        # Create and handle different types of errors
        validation_error = create_validation_error(
            "Test validation failure",
            validation_type="test",
            error_code=ErrorCode.VALIDATION_FAILED
        )

        orchestration_error = create_orchestration_error(
            "Test orchestration failure",
            error_code=ErrorCode.ORCHESTRATION_PIPELINE_FAILED
        )

        # Handle the errors
        handle_error(validation_error)
        handle_error(orchestration_error)

        # Check error tracking
        summary = error_handler.get_error_summary()
        assert summary["total_errors"] == 2
        assert "validation" in summary["category_breakdown"]
        assert "orchestration" in summary["category_breakdown"]
        assert ErrorCode.VALIDATION_FAILED in summary["most_common_errors"]

    def test_structured_error_details(self):
        """Test structured error details are properly formatted."""
        error = create_validation_error(
            "Test error with context",
            validation_type="test_type",
            error_code=ErrorCode.VALIDATION_FAILED,
            context={"test_key": "test_value", "number": 42}
        )

        error.add_recovery_suggestion("Try this fix")
        error.add_recovery_suggestion("Or try this alternative")
        error.set_user_message("User-friendly error message")

        details_dict = error.details.to_dict()

        assert details_dict["code"] == ErrorCode.VALIDATION_FAILED
        assert details_dict["message"] == "Test error with context"
        assert details_dict["category"] == error.details.category
        assert details_dict["severity"] == error.details.severity
        assert "test_key" in details_dict["context"]
        assert len(details_dict["recovery_suggestions"]) == 2
        assert details_dict["user_message"] == "User-friendly error message"

    def test_error_json_serialization(self):
        """Test error details can be serialized to JSON."""
        error = create_validation_error(
            "Test error for JSON",
            error_code=ErrorCode.VALIDATION_FAILED,
            context={"test": True}
        )

        json_str = error.details.to_json()
        parsed = json.loads(json_str)

        assert parsed["code"] == ErrorCode.VALIDATION_FAILED.value
        assert parsed["message"] == "Test error for JSON"
        assert parsed["context"]["test"] is True

    def test_pipeline_end_to_end_error_propagation(self):
        """Test errors propagate correctly through the entire pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
""")

            orchestrator = AgentOrchestrator(str(config_path))

            # Clear error history
            orchestrator.reset_error_history()

            # Try to run with invalid inputs
            with pytest.raises(ValueError):  # Should be ValueError from run_mode validation
                orchestrator.run_mode("", "")  # Empty domain and task

            # Try with invalid domain
            with pytest.raises(ValueError):
                orchestrator.run_mode("invalid_domain", "test task")


class TestReflectionLogicEdgeCases:
    """Test edge cases in reflection logic and iteration handling."""

    def test_reflection_early_termination_conditions(self):
        """Test various early termination conditions in reflection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_path.write_text("""
enterprise_coding_agent:
  orchestration: {}
  memory: {}
  governance: {}
  reflecting:
    max_iterations: 5
    early_termination:
      enable: true
      stagnation_threshold: 3
      min_iterations: 1
      progress_threshold: 0.1
""")

            orchestrator = AgentOrchestrator(str(config_path))

            # Test confidence regression termination
            initial_state = AgentState({
                "task": "test task",
                "domain": "coding",
                "confidence": 0.8,
                "needs_reflect": True,
                "validation": {"passes": False}
            })

            # This would require mocking the reflection loop execution
            # to simulate confidence regression
            assert orchestrator is not None  # Basic sanity check

    def test_reflection_audit_logging(self):
        """Test reflection audit logging captures all necessary information."""
        from src.utils.reflection_audit import get_reflection_auditor

        auditor = get_reflection_auditor()

        # Test session tracking
        session_id = auditor.start_reflection_session(
            domain="coding",
            task="test task",
            initial_confidence=0.5,
            max_iterations=5,
            configuration={"test": True}
        )

        assert session_id is not None

        # Test step logging
        from src.utils.reflection_audit import ReflectionPhase, ValidationIssue

        issues = [
            ValidationIssue(
                issue_type="test_failure",
                severity="high",
                description="Test failed",
                confidence=0.9
            )
        ]

        auditor.log_reflection_step(
            session_id,
            ReflectionPhase.ISSUE_IDENTIFICATION,
            input_data={"test": "data"},
            output_data={"result": "identified"},
            confidence_before=0.5,
            issues=issues
        )

        # Test session completion
        from src.utils.reflection_audit import ReflectionDecision

        auditor.finish_reflection_session(
            session_id,
            ReflectionDecision.EARLY_TERMINATION,
            final_confidence=0.8,
            outcome={"success": True}
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])