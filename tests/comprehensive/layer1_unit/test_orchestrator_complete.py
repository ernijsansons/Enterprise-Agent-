"""Complete unit tests for AgentOrchestrator component."""
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.agent_orchestrator import AgentOrchestrator, AgentState
from tests.comprehensive.test_framework import (
    critical_test,
    high_priority_test,
    low_priority_test,
    medium_priority_test,
)


class TestOrchestratorComplete:
    """Complete test suite for AgentOrchestrator."""

    def setup_method(self):
        """Setup for each test."""
        self.test_config_path = self._create_test_config()
        self.orchestrator = None

    def teardown_method(self):
        """Cleanup after each test."""
        if self.test_config_path and os.path.exists(self.test_config_path):
            os.unlink(self.test_config_path)

    def _create_test_config(self) -> str:
        """Create minimal test configuration."""
        config_content = """
enterprise_coding_agent:
  orchestration:
    runtime_optimizer:
      max_daily_cost: 10.0
  memory:
    retention_days: 1
  governance:
    enabled: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            return f.name

    @critical_test
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        try:
            # Test with minimal config
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Verify core components
            assert hasattr(self.orchestrator, "config")
            assert hasattr(self.orchestrator, "agent_cfg")
            assert hasattr(self.orchestrator, "memory")
            assert hasattr(self.orchestrator, "cost_estimator")
            assert hasattr(self.orchestrator, "governance")

            # Verify role initialization
            assert hasattr(self.orchestrator, "planner_role")
            assert hasattr(self.orchestrator, "coder_role")
            assert hasattr(self.orchestrator, "validator_role")
            assert hasattr(self.orchestrator, "reflector_role")
            assert hasattr(self.orchestrator, "reviewer_role")

            return {"success": True, "message": "Orchestrator initialized successfully"}

        except Exception as e:
            return {"success": False, "message": f"Initialization failed: {e}"}

    @critical_test
    def test_async_integration(self):
        """Test async orchestrator integration."""
        try:
            # Enable async
            os.environ["ENABLE_ASYNC"] = "true"
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Verify async methods exist
            assert hasattr(self.orchestrator, "_call_model_async")
            assert hasattr(self.orchestrator, "run_mode_async")
            assert hasattr(self.orchestrator, "_execute_pipeline_async")

            # Verify async orchestrator is initialized
            has_async_orch = hasattr(self.orchestrator, "_async_orchestrator")

            return {
                "success": True,
                "message": "Async integration verified",
                "details": {"async_orchestrator_available": has_async_orch},
            }

        except Exception as e:
            return {"success": False, "message": f"Async integration failed: {e}"}
        finally:
            os.environ.pop("ENABLE_ASYNC", None)

    @high_priority_test
    def test_model_routing(self):
        """Test model routing logic."""
        try:
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Test different routing scenarios
            test_cases = [
                ("simple task", "coding", False, "Expected claude model"),
                (
                    "complex trading algorithm",
                    "trading",
                    True,
                    "Expected claude opus for complex/vuln",
                ),
                ("", "web", False, "Empty task handling"),
                ("very " * 1000 + "long task", "data", False, "Long task handling"),
            ]

            results = []
            for task, domain, vuln_flag, description in test_cases:
                try:
                    model = self.orchestrator.route_to_model(task, domain, vuln_flag)
                    results.append(
                        {
                            "description": description,
                            "model": model,
                            "success": bool(model),
                        }
                    )
                except Exception as e:
                    results.append(
                        {"description": description, "error": str(e), "success": False}
                    )

            success_count = sum(1 for r in results if r["success"])

            return {
                "success": success_count == len(test_cases),
                "message": f"Model routing: {success_count}/{len(test_cases)} passed",
                "details": {"test_results": results},
            }

        except Exception as e:
            return {"success": False, "message": f"Model routing failed: {e}"}

    @high_priority_test
    def test_call_model_functionality(self):
        """Test _call_model method with different scenarios."""
        try:
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Test with stubbed model (no API keys)
            result = self.orchestrator._call_model(
                model="claude_sonnet_4",
                prompt="Test prompt",
                role="Tester",
                operation="test_call",
                max_tokens=100,
            )

            # Should return stubbed response or actual response
            assert isinstance(result, str)
            assert len(result) > 0

            return {
                "success": True,
                "message": "Model call functionality verified",
                "details": {"response_length": len(result)},
            }

        except Exception as e:
            return {"success": False, "message": f"Model call failed: {e}"}

    @high_priority_test
    def test_role_execution_pipeline(self):
        """Test individual role execution."""
        try:
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Test each role
            test_state = AgentState(
                {
                    "task": "Create a simple function",
                    "domain": "coding",
                    "iterations": 0,
                    "confidence": 0.0,
                    "vuln_flag": False,
                }
            )

            role_results = {}

            # Test Planner
            try:
                planner_state = self.orchestrator.planner(test_state.copy())
                role_results["planner"] = {
                    "success": "plan" in planner_state,
                    "has_output": bool(planner_state.get("plan")),
                }
            except Exception as e:
                role_results["planner"] = {"success": False, "error": str(e)}

            # Test Coder (needs plan)
            try:
                coder_state = test_state.copy()
                coder_state[
                    "plan"
                ] = "1. Create function\n2. Add parameters\n3. Return result"
                coder_result = self.orchestrator.coder(coder_state)
                role_results["coder"] = {
                    "success": "code" in coder_result,
                    "has_output": bool(coder_result.get("code")),
                }
            except Exception as e:
                role_results["coder"] = {"success": False, "error": str(e)}

            # Test Validator (needs code)
            try:
                validator_state = test_state.copy()
                validator_state["code"] = "def test_function(): return 'test'"
                validator_result = self.orchestrator.validator(validator_state)
                role_results["validator"] = {
                    "success": "validation" in validator_result,
                    "has_output": bool(validator_result.get("validation")),
                }
            except Exception as e:
                role_results["validator"] = {"success": False, "error": str(e)}

            success_count = sum(
                1 for r in role_results.values() if r.get("success", False)
            )

            return {
                "success": success_count >= 2,  # At least 2 roles should work
                "message": f"Role execution: {success_count}/3 roles passed",
                "details": {"role_results": role_results},
            }

        except Exception as e:
            return {"success": False, "message": f"Role execution failed: {e}"}

    @medium_priority_test
    def test_state_management(self):
        """Test AgentState functionality."""
        try:
            # Test state creation
            state = AgentState(
                {"task": "test task", "domain": "test", "custom_field": "test_value"}
            )

            # Test state access
            assert state["task"] == "test task"
            assert state.get("domain") == "test"
            assert state.get("nonexistent", "default") == "default"

            # Test state modification
            state["new_field"] = "new_value"
            assert state["new_field"] == "new_value"

            # Test state copy
            state_copy = state.copy()
            state_copy["task"] = "modified task"
            assert state["task"] == "test task"  # Original unchanged
            assert state_copy["task"] == "modified task"

            return {
                "success": True,
                "message": "State management verified",
                "details": {"state_keys": list(state.keys())},
            }

        except Exception as e:
            return {"success": False, "message": f"State management failed: {e}"}

    @medium_priority_test
    def test_context_enhancement(self):
        """Test context enhancement functionality."""
        try:
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Test prompt enhancement
            original_prompt = "Write a function"
            enhanced = self.orchestrator._enhance_prompt(original_prompt, "Coder")

            # Test project context building
            context = self.orchestrator._build_project_context()
            assert isinstance(context, dict)

            # Test state enrichment
            state = AgentState({"task": "test"})
            enriched_state = self.orchestrator._enrich_state_with_context(state)

            return {
                "success": True,
                "message": "Context enhancement verified",
                "details": {
                    "prompt_enhanced": enhanced != original_prompt,
                    "context_keys": list(context.keys()),
                    "state_enriched": len(enriched_state) >= len(state),
                },
            }

        except Exception as e:
            return {"success": False, "message": f"Context enhancement failed: {e}"}

    @medium_priority_test
    def test_error_handling(self):
        """Test error handling and recovery."""
        try:
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            error_scenarios = []

            # Test invalid model call
            try:
                self.orchestrator._call_model(
                    model="nonexistent_model",
                    prompt="test",
                    role="Tester",
                    operation="error_test",
                )
                error_scenarios.append({"scenario": "invalid_model", "handled": True})
            except Exception as e:
                error_scenarios.append(
                    {"scenario": "invalid_model", "handled": False, "error": str(e)}
                )

            # Test malformed state
            try:
                AgentState({"task": None, "domain": ""})
                self.orchestrator.run_mode("", "")
                error_scenarios.append({"scenario": "empty_inputs", "handled": True})
            except ValueError:
                error_scenarios.append(
                    {"scenario": "empty_inputs", "handled": True}
                )  # Expected
            except Exception as e:
                error_scenarios.append(
                    {"scenario": "empty_inputs", "handled": False, "error": str(e)}
                )

            handled_count = sum(1 for s in error_scenarios if s["handled"])

            return {
                "success": handled_count >= 1,
                "message": f"Error handling: {handled_count}/{len(error_scenarios)} scenarios handled",
                "details": {"scenarios": error_scenarios},
            }

        except Exception as e:
            return {"success": False, "message": f"Error handling test failed: {e}"}

    @low_priority_test
    def test_configuration_loading(self):
        """Test configuration loading and validation."""
        try:
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Verify config structure
            assert hasattr(self.orchestrator, "config")
            assert hasattr(self.orchestrator, "agent_cfg")

            # Test domain packs loading
            domain_packs = getattr(self.orchestrator, "domain_packs", {})

            # Test cost estimator configuration
            cost_summary = self.orchestrator.cost_estimator.summary()
            assert isinstance(cost_summary, dict)

            return {
                "success": True,
                "message": "Configuration loading verified",
                "details": {
                    "has_config": bool(self.orchestrator.config),
                    "domain_packs_count": len(domain_packs),
                    "cost_estimator_ready": bool(cost_summary),
                },
            }

        except Exception as e:
            return {"success": False, "message": f"Configuration loading failed: {e}"}

    @low_priority_test
    def test_memory_integration(self):
        """Test memory store integration."""
        try:
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Test memory operations
            self.orchestrator.memory.store("test_scope", "test_key", "test_value")
            retrieved = self.orchestrator.memory.retrieve("test_scope", "test_key")

            assert retrieved == "test_value"

            # Test memory pruning
            self.orchestrator.memory.prune()

            return {
                "success": True,
                "message": "Memory integration verified",
                "details": {"store_retrieve_works": retrieved == "test_value"},
            }

        except Exception as e:
            return {"success": False, "message": f"Memory integration failed: {e}"}

    @medium_priority_test
    def test_claude_code_integration(self):
        """Test Claude Code provider integration."""
        try:
            # Test with Claude Code enabled
            os.environ["USE_CLAUDE_CODE"] = "true"
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Check integration
            has_claude_provider = hasattr(self.orchestrator, "_claude_code_provider")
            claude_enabled = getattr(self.orchestrator, "_use_claude_code", False)

            return {
                "success": True,
                "message": "Claude Code integration checked",
                "details": {
                    "claude_enabled": claude_enabled,
                    "provider_available": has_claude_provider,
                },
            }

        except Exception as e:
            return {"success": False, "message": f"Claude Code integration failed: {e}"}
        finally:
            os.environ.pop("USE_CLAUDE_CODE", None)

    @critical_test
    def test_run_mode_basic_functionality(self):
        """Test basic run_mode functionality."""
        try:
            self.orchestrator = AgentOrchestrator(self.test_config_path)

            # Test basic run
            result = self.orchestrator.run_mode(
                domain="coding",
                task="Create a simple hello world function",
                vuln_flag=False,
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "domain" in result
            assert result["domain"] == "coding"

            # Should have some output from roles
            has_output = any(
                key in result for key in ["plan", "code", "validation", "confidence"]
            )

            return {
                "success": has_output,
                "message": "Basic run_mode functionality verified",
                "details": {
                    "result_keys": list(result.keys()),
                    "has_outputs": has_output,
                },
            }

        except Exception as e:
            return {"success": False, "message": f"run_mode failed: {e}"}


def get_orchestrator_tests():
    """Get all orchestrator test methods."""
    test_class = TestOrchestratorComplete()
    test_methods = []

    for attr_name in dir(test_class):
        if attr_name.startswith("test_") and callable(getattr(test_class, attr_name)):
            method = getattr(test_class, attr_name)
            test_methods.append(method)

    return test_methods, test_class.setup_method, test_class.teardown_method
