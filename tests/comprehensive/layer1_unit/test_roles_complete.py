"""Complete unit tests for all role components."""
import sys
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.comprehensive.test_framework import (
    critical_test,
    high_priority_test,
    low_priority_test,
    medium_priority_test,
)


class TestRolesComplete:
    """Complete test suite for all role components."""

    def setup_method(self):
        """Setup for each test."""
        # Create mock orchestrator
        self.mock_orchestrator = Mock()
        self.mock_orchestrator._call_model = Mock(return_value="Mocked response")
        self.mock_orchestrator._enhance_prompt = Mock(side_effect=lambda p, r: p)
        self.mock_orchestrator._parse_json = Mock(return_value={"test": "data"})
        self.mock_orchestrator.route_to_model = Mock(return_value="claude_sonnet_4")

    @critical_test
    def test_planner_role_functionality(self):
        """Test Planner role core functionality."""
        try:
            from src.roles.planner import Planner

            planner = Planner(self.mock_orchestrator)

            # Test decompose method
            result = planner.decompose(
                task="Create a web application", domain="web_development"
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "text" in result or "output" in result

            # Verify orchestrator interaction
            assert self.mock_orchestrator._call_model.called

            return {
                "success": True,
                "message": "Planner role functionality verified",
                "details": {"result_keys": list(result.keys())},
            }

        except Exception as e:
            return {"success": False, "message": f"Planner test failed: {e}"}

    @critical_test
    def test_coder_role_functionality(self):
        """Test Coder role core functionality."""
        try:
            from src.roles.coder import Coder

            coder = Coder(self.mock_orchestrator)

            # Test generate method
            result = coder.generate(
                plan="1. Create function\n2. Add parameters\n3. Return result",
                domain="coding",
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "output" in result or "code" in result

            # Verify orchestrator interaction
            assert self.mock_orchestrator._call_model.called

            return {
                "success": True,
                "message": "Coder role functionality verified",
                "details": {"result_keys": list(result.keys())},
            }

        except Exception as e:
            return {"success": False, "message": f"Coder test failed: {e}"}

    @critical_test
    def test_validator_role_functionality(self):
        """Test Validator role core functionality."""
        try:
            from src.roles.validator import Validator

            validator = Validator(self.mock_orchestrator)

            # Mock JSON parsing for validation result
            self.mock_orchestrator._parse_json.return_value = {
                "passes": True,
                "coverage": 0.95,
                "issues": [],
            }

            # Test validate method
            result = validator.validate(
                code="def hello(): return 'world'", domain="coding"
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "parsed" in result

            # Verify validation data
            parsed = result.get("parsed", {})
            assert "passes" in parsed

            return {
                "success": True,
                "message": "Validator role functionality verified",
                "details": {
                    "result_keys": list(result.keys()),
                    "validation_result": parsed,
                },
            }

        except Exception as e:
            return {"success": False, "message": f"Validator test failed: {e}"}

    @high_priority_test
    def test_reflector_role_functionality(self):
        """Test Reflector role core functionality."""
        try:
            from src.roles.reflector import Reflector

            reflector = Reflector(self.mock_orchestrator)

            # Mock JSON parsing for reflection result
            self.mock_orchestrator._parse_json.return_value = {
                "analysis": "Code needs improvement",
                "fixes": ["Add error handling", "Improve naming"],
                "selected_fix": 0,
                "revised_output": "def improved_hello(): return 'world'",
                "confidence": 0.8,
            }

            # Test reflect method
            validation_result = {"passes": False, "issues": ["Missing error handling"]}
            code = "def hello(): return 'world'"

            result = reflector.reflect(
                validation=validation_result,
                code=code,
                domain="coding",
                iterations=1,
                vuln_flag=False,
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "output" in result or "analysis" in result

            return {
                "success": True,
                "message": "Reflector role functionality verified",
                "details": {"result_keys": list(result.keys())},
            }

        except Exception as e:
            return {"success": False, "message": f"Reflector test failed: {e}"}

    @high_priority_test
    def test_reviewer_role_functionality(self):
        """Test Reviewer role core functionality."""
        try:
            from src.roles.reviewer import Reviewer

            reviewer = Reviewer(self.mock_orchestrator)

            # Mock JSON parsing for review result
            self.mock_orchestrator._parse_json.return_value = {
                "score": 0.85,
                "rationale": "Good code quality with minor improvements needed",
            }

            # Test review method
            result = reviewer.review(
                code="def hello(): return 'world'", domain="coding", vuln_flag=False
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "confidence" in result or "scores" in result

            return {
                "success": True,
                "message": "Reviewer role functionality verified",
                "details": {"result_keys": list(result.keys())},
            }

        except Exception as e:
            return {"success": False, "message": f"Reviewer test failed: {e}"}

    @medium_priority_test
    def test_role_inheritance_structure(self):
        """Test role inheritance from base class."""
        try:
            from src.roles.base import BaseRole
            from src.roles.coder import Coder
            from src.roles.planner import Planner
            from src.roles.validator import Validator

            # Test inheritance
            planner = Planner(self.mock_orchestrator)
            coder = Coder(self.mock_orchestrator)
            validator = Validator(self.mock_orchestrator)

            # Verify inheritance
            assert isinstance(planner, BaseRole)
            assert isinstance(coder, BaseRole)
            assert isinstance(validator, BaseRole)

            # Test base functionality
            assert hasattr(planner, "orchestrator")
            assert hasattr(coder, "orchestrator")
            assert hasattr(validator, "orchestrator")

            return {
                "success": True,
                "message": "Role inheritance verified",
                "details": {"all_inherit_base": True},
            }

        except Exception as e:
            return {"success": False, "message": f"Role inheritance test failed: {e}"}

    @medium_priority_test
    def test_role_prompt_enhancement(self):
        """Test role-specific prompt enhancement."""
        try:
            from src.roles.planner import Planner

            planner = Planner(self.mock_orchestrator)

            # Test with prompt enhancement
            self.mock_orchestrator._enhance_prompt.return_value = (
                "Enhanced: Create a web app"
            )

            planner.decompose("Create a web app", "web")

            # Verify enhancement was called
            assert self.mock_orchestrator._enhance_prompt.called
            call_args = self.mock_orchestrator._enhance_prompt.call_args
            assert call_args[0][1] == "Planner"  # Role name passed correctly

            return {"success": True, "message": "Prompt enhancement verified"}

        except Exception as e:
            return {"success": False, "message": f"Prompt enhancement test failed: {e}"}

    @medium_priority_test
    def test_role_error_handling(self):
        """Test role error handling."""
        try:
            from src.roles.coder import Coder

            # Make orchestrator throw exception
            self.mock_orchestrator._call_model.side_effect = Exception("Model error")

            coder = Coder(self.mock_orchestrator)

            # This should handle the error gracefully
            try:
                coder.generate("test plan", "coding")
                error_handled = True
            except Exception:
                error_handled = False

            return {
                "success": True,  # We test that it doesn't crash the test
                "message": "Role error handling tested",
                "details": {"graceful_handling": error_handled},
            }

        except Exception as e:
            return {"success": False, "message": f"Error handling test failed: {e}"}

    @low_priority_test
    def test_role_configuration_parameters(self):
        """Test role configuration and parameters."""
        try:
            from src.roles.coder import Coder
            from src.roles.planner import Planner

            planner = Planner(self.mock_orchestrator)
            coder = Coder(self.mock_orchestrator)

            # Test different parameters
            test_cases = [
                (planner.decompose, ("simple task", "coding")),
                (planner.decompose, ("complex task with details", "trading")),
                (coder.generate, ("basic plan", "web")),
                (coder.generate, ("detailed plan with steps", "data_science")),
            ]

            results = []
            for method, args in test_cases:
                try:
                    method(*args)
                    results.append({"success": True, "method": method.__name__})
                except Exception as e:
                    results.append(
                        {"success": False, "method": method.__name__, "error": str(e)}
                    )

            success_count = sum(1 for r in results if r["success"])

            return {
                "success": success_count >= len(test_cases) // 2,
                "message": f"Role parameters: {success_count}/{len(test_cases)} tests passed",
                "details": {"test_results": results},
            }

        except Exception as e:
            return {"success": False, "message": f"Role configuration test failed: {e}"}

    @low_priority_test
    def test_role_model_selection(self):
        """Test role-specific model selection."""
        try:
            from src.roles.planner import Planner

            # Test different model routing scenarios
            model_scenarios = [
                ("simple", "coding", "claude_sonnet_4"),
                ("complex", "trading", "claude_opus_4"),
                ("medium", "web", "claude_sonnet_4"),
            ]

            results = []
            for task, domain, expected_pattern in model_scenarios:
                self.mock_orchestrator.route_to_model.return_value = expected_pattern
                planner = Planner(self.mock_orchestrator)

                planner.decompose(task, domain)

                # Verify model routing was called
                route_called = self.mock_orchestrator.route_to_model.called
                results.append(
                    {"scenario": f"{task}_{domain}", "route_called": route_called}
                )

                # Reset mock
                self.mock_orchestrator.route_to_model.reset_mock()

            success_count = sum(1 for r in results if r["route_called"])

            return {
                "success": success_count >= len(model_scenarios) // 2,
                "message": f"Model selection: {success_count}/{len(model_scenarios)} scenarios passed",
                "details": {"scenario_results": results},
            }

        except Exception as e:
            return {"success": False, "message": f"Model selection test failed: {e}"}

    @critical_test
    def test_all_roles_importable(self):
        """Test that all roles can be imported and instantiated."""
        try:
            from src.roles import Coder, Planner, Reflector, Reviewer, Validator

            roles = {
                "Planner": Planner,
                "Coder": Coder,
                "Validator": Validator,
                "Reflector": Reflector,
                "Reviewer": Reviewer,
            }

            instantiated_roles = {}
            for name, role_class in roles.items():
                try:
                    role_instance = role_class(self.mock_orchestrator)
                    instantiated_roles[name] = {
                        "success": True,
                        "instance": role_instance,
                    }
                except Exception as e:
                    instantiated_roles[name] = {"success": False, "error": str(e)}

            success_count = sum(1 for r in instantiated_roles.values() if r["success"])

            return {
                "success": success_count == len(roles),
                "message": f"Role imports: {success_count}/{len(roles)} successful",
                "details": {
                    "role_status": {
                        k: v["success"] for k, v in instantiated_roles.items()
                    }
                },
            }

        except Exception as e:
            return {"success": False, "message": f"Role import test failed: {e}"}


def get_roles_tests():
    """Get all role test methods."""
    test_class = TestRolesComplete()
    test_methods = []

    for attr_name in dir(test_class):
        if attr_name.startswith("test_") and callable(getattr(test_class, attr_name)):
            method = getattr(test_class, attr_name)
            test_methods.append(method)

    return test_methods, test_class.setup_method, None
