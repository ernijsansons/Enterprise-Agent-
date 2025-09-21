#!/usr/bin/env python3
"""Test script to validate configurable reflection loop parameters."""
import os
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not available, using simplified config format")
    yaml = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def create_test_config(
    max_iterations=3, confidence_threshold=0.7, early_termination_enable=True
):
    """Create a test configuration with custom reflection parameters."""
    config = {
        "enterprise_coding_agent": {
            "reflecting": {
                "max_iterations": max_iterations,
                "confidence_threshold": confidence_threshold,
                "early_termination": {
                    "enable": early_termination_enable,
                    "stagnation_threshold": 2,
                    "min_iterations": 1,
                    "progress_threshold": 0.05,
                },
            },
            "reviewing": {"confidence_threshold": confidence_threshold},
            "memory": {},
            "orchestration": {"runtime_optimizer": {}},
            "governance": {},
        }
    }
    return config


def test_config_loading():
    """Test that configuration is properly loaded."""
    print("Testing configuration loading...")

    if not yaml:
        print("⚠️  Skipping config loading test (PyYAML not available)")
        return True

    # Create temporary config file
    test_config = create_test_config(max_iterations=7, confidence_threshold=0.9)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name

    try:
        from src.agent_orchestrator import AgentOrchestrator

        # Initialize orchestrator with test config
        orchestrator = AgentOrchestrator(config_path)

        # Check that configuration is loaded correctly
        reflecting_cfg = orchestrator.agent_cfg.get("reflecting", {})
        assert (
            reflecting_cfg.get("max_iterations") == 7
        ), f"Expected max_iterations=7, got {reflecting_cfg.get('max_iterations')}"
        assert (
            reflecting_cfg.get("confidence_threshold") == 0.9
        ), f"Expected confidence_threshold=0.9, got {reflecting_cfg.get('confidence_threshold')}"

        early_term_cfg = reflecting_cfg.get("early_termination", {})
        assert (
            early_term_cfg.get("enable") is True
        ), f"Expected early_termination.enable=True, got {early_term_cfg.get('enable')}"
        assert (
            early_term_cfg.get("stagnation_threshold") == 2
        ), f"Expected stagnation_threshold=2, got {early_term_cfg.get('stagnation_threshold')}"

        print("✅ Configuration loading test passed")
        return True

    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

    finally:
        # Cleanup
        os.unlink(config_path)


def test_environment_variable_override():
    """Test that environment variables can override config values."""
    print("\nTesting environment variable override...")

    # Set environment variables
    os.environ["REFLECTION_MAX_ITERATIONS"] = "10"
    os.environ["REFLECTION_CONFIDENCE_THRESHOLD"] = "0.95"

    try:
        # Create test config with different values
        test_config = create_test_config(max_iterations=3, confidence_threshold=0.7)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        from src.agent_orchestrator import AgentOrchestrator

        # Create a mock state to test reflection routing
        class MockState(dict):
            def get(self, key, default=None):
                return super().get(key, default)

        orchestrator = AgentOrchestrator(config_path)

        # Test environment variable override in _reflect_route
        mock_state = MockState({"iterations": 5, "confidence": 0.8})

        # This should return "coder" because iterations (5) < env_max_iterations (10)
        # and confidence (0.8) < env_confidence_threshold (0.95)
        route = orchestrator._reflect_route(mock_state)
        assert route == "coder", f"Expected route 'coder', got '{route}'"

        # Test with higher iterations
        mock_state = MockState({"iterations": 12, "confidence": 0.8})
        route = orchestrator._reflect_route(mock_state)
        assert route == "reviewer", f"Expected route 'reviewer', got '{route}'"

        print("✅ Environment variable override test passed")
        return True

    except Exception as e:
        print(f"❌ Environment variable override test failed: {e}")
        return False

    finally:
        # Cleanup environment variables
        os.environ.pop("REFLECTION_MAX_ITERATIONS", None)
        os.environ.pop("REFLECTION_CONFIDENCE_THRESHOLD", None)
        if "config_path" in locals():
            os.unlink(config_path)


def test_early_termination_configuration():
    """Test that early termination configuration is properly applied."""
    print("\nTesting early termination configuration...")

    try:
        # Create test config with early termination disabled
        test_config = create_test_config(early_termination_enable=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        from src.agent_orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator(config_path)

        # Check that early termination configuration is loaded
        reflecting_cfg = orchestrator.agent_cfg.get("reflecting", {})
        early_term_cfg = reflecting_cfg.get("early_termination", {})

        assert (
            early_term_cfg.get("enable") is False
        ), f"Expected early_termination.enable=False, got {early_term_cfg.get('enable')}"

        print("✅ Early termination configuration test passed")
        return True

    except Exception as e:
        print(f"❌ Early termination configuration test failed: {e}")
        return False

    finally:
        if "config_path" in locals():
            os.unlink(config_path)


def main():
    """Run all configuration tests."""
    print("Configurable Reflection Loop Parameters Test Suite")
    print("=" * 60)

    tests_passed = 0
    total_tests = 3

    if test_config_loading():
        tests_passed += 1

    if test_environment_variable_override():
        tests_passed += 1

    if test_early_termination_configuration():
        tests_passed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print(
            "🎉 All tests passed! Configurable reflection parameters are working correctly."
        )
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
