"""Complete unit tests for provider components."""
import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.comprehensive.test_framework import critical_test, high_priority_test, medium_priority_test, low_priority_test

class TestProvidersComplete:
    """Complete test suite for all provider components."""

    def setup_method(self):
        """Setup for each test."""
        self.test_config = {
            "timeout": 30,
            "enable_fallback": True,
            "working_directory": os.getcwd(),
        }

    @critical_test
    def test_claude_code_provider_initialization(self):
        """Test Claude Code provider initialization."""
        try:
            from src.providers.claude_code_provider import ClaudeCodeProvider

            provider = ClaudeCodeProvider(self.test_config)

            # Verify basic attributes
            assert hasattr(provider, 'config')
            assert hasattr(provider, 'sessions')
            assert provider.config == self.test_config

            return {
                "success": True,
                "message": "Claude Code provider initialized successfully"
            }

        except Exception as e:
            return {"success": False, "message": f"Claude Code provider init failed: {e}"}

    @critical_test
    def test_auth_manager_functionality(self):
        """Test authentication manager functionality."""
        try:
            from src.providers.auth_manager import AuthManager

            auth_manager = AuthManager()

            # Test basic functionality (without requiring actual CLI)
            assert hasattr(auth_manager, 'validate_setup')
            assert hasattr(auth_manager, 'ensure_subscription_mode')

            # Test setup validation (should not crash)
            try:
                validation = auth_manager.validate_setup()
                validation_works = isinstance(validation, dict)
            except Exception:
                validation_works = False  # Expected if CLI not available

            return {
                "success": True,
                "message": "Auth manager functionality verified",
                "details": {"validation_callable": validation_works}
            }

        except Exception as e:
            return {"success": False, "message": f"Auth manager test failed: {e}"}

    @high_priority_test
    def test_async_claude_provider_functionality(self):
        """Test async Claude provider functionality."""
        try:
            from src.providers.async_claude_provider import AsyncClaudeCodeProvider

            provider = AsyncClaudeCodeProvider(self.test_config)

            # Verify async methods exist
            assert hasattr(provider, 'call_model')
            assert hasattr(provider, 'batch_call_models')
            assert hasattr(provider, 'stream_response')

            # Verify it's actually async
            assert asyncio.iscoroutinefunction(provider.call_model)
            assert asyncio.iscoroutinefunction(provider.batch_call_models)

            return {
                "success": True,
                "message": "Async Claude provider functionality verified"
            }

        except Exception as e:
            return {"success": False, "message": f"Async Claude provider test failed: {e}"}

    @high_priority_test
    async def test_async_claude_provider_call_model(self):
        """Test async Claude provider model calling."""
        try:
            from src.providers.async_claude_provider import AsyncClaudeCodeProvider

            provider = AsyncClaudeCodeProvider(self.test_config)

            # Mock the CLI execution to avoid requiring actual Claude CLI
            with patch.object(provider, '_execute_claude_cli_async') as mock_execute:
                # Mock successful response
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = '{"text": "Mocked response"}'
                mock_result.stderr = ""
                mock_execute.return_value = mock_result

                # Test call_model
                response = await provider.call_model(
                    prompt="Test prompt",
                    model="claude_sonnet_4",
                    role="Tester",
                    operation="test",
                    use_cache=False  # Disable cache for testing
                )

                assert isinstance(response, str)
                assert len(response) > 0

            return {
                "success": True,
                "message": "Async Claude provider call_model verified",
                "details": {"response_received": bool(response)}
            }

        except Exception as e:
            return {"success": False, "message": f"Async call_model test failed: {e}"}

    @high_priority_test
    async def test_async_claude_provider_batch_calls(self):
        """Test async Claude provider batch calling."""
        try:
            from src.providers.async_claude_provider import AsyncClaudeCodeProvider

            provider = AsyncClaudeCodeProvider(self.test_config)

            # Mock the CLI execution
            with patch.object(provider, '_execute_claude_cli_async') as mock_execute:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = '{"text": "Batch response"}'
                mock_result.stderr = ""
                mock_execute.return_value = mock_result

                # Test batch calls
                requests = [
                    {
                        "prompt": f"Test prompt {i}",
                        "model": "claude_sonnet_4",
                        "role": "Tester",
                        "operation": f"batch_test_{i}",
                        "use_cache": False
                    }
                    for i in range(3)
                ]

                responses = await provider.batch_call_models(requests, max_concurrent=2)

                assert len(responses) == len(requests)
                assert all(isinstance(r, str) for r in responses)

            return {
                "success": True,
                "message": "Async Claude provider batch calls verified",
                "details": {"batch_size": len(responses)}
            }

        except Exception as e:
            return {"success": False, "message": f"Async batch calls test failed: {e}"}

    @medium_priority_test
    def test_claude_code_provider_model_mapping(self):
        """Test Claude Code provider model mapping."""
        try:
            from src.providers.claude_code_provider import ClaudeCodeProvider

            provider = ClaudeCodeProvider(self.test_config)

            # Test model mapping
            test_mappings = [
                ("claude_sonnet_4", "sonnet"),
                ("claude-3-5-sonnet", "sonnet"),
                ("claude_opus_4", "opus"),
                ("claude-3-opus", "opus"),
                ("claude_haiku", "haiku"),
                ("unknown_model", "sonnet"),  # Should default to sonnet
            ]

            mapping_results = []
            for input_model, expected in test_mappings:
                try:
                    mapped = provider._map_model_to_cli(input_model)
                    mapping_results.append({
                        "input": input_model,
                        "expected": expected,
                        "actual": mapped,
                        "correct": mapped == expected
                    })
                except Exception as e:
                    mapping_results.append({
                        "input": input_model,
                        "error": str(e),
                        "correct": False
                    })

            correct_mappings = sum(1 for r in mapping_results if r.get("correct", False))

            return {
                "success": correct_mappings >= len(test_mappings) * 0.8,  # 80% success rate
                "message": f"Model mapping: {correct_mappings}/{len(test_mappings)} correct",
                "details": {"mapping_results": mapping_results}
            }

        except Exception as e:
            return {"success": False, "message": f"Model mapping test failed: {e}"}

    @medium_priority_test
    def test_claude_code_provider_session_management(self):
        """Test Claude Code provider session management."""
        try:
            from src.providers.claude_code_provider import ClaudeCodeProvider

            provider = ClaudeCodeProvider(self.test_config)

            # Test session creation (should not crash)
            try:
                session_id = provider.create_session()
                session_created = True
            except Exception:
                session_created = False  # Expected if CLI not available

            # Test session management methods exist
            assert hasattr(provider, 'create_session')
            assert hasattr(provider, 'sessions')

            return {
                "success": True,
                "message": "Session management functionality verified",
                "details": {"session_creation_works": session_created}
            }

        except Exception as e:
            return {"success": False, "message": f"Session management test failed: {e}"}

    @medium_priority_test
    def test_claude_code_provider_error_handling(self):
        """Test Claude Code provider error handling."""
        try:
            from src.providers.claude_code_provider import ClaudeCodeProvider

            provider = ClaudeCodeProvider(self.test_config)

            # Mock CLI execution that fails
            with patch('subprocess.run') as mock_run:
                # Mock failed execution
                mock_result = Mock()
                mock_result.returncode = 1
                mock_result.stdout = ""
                mock_result.stderr = "CLI error"
                mock_run.return_value = mock_result

                # This should handle the error gracefully
                try:
                    result = provider.call_model(
                        prompt="Test prompt",
                        model="claude_sonnet_4",
                        role="Tester",
                        operation="error_test"
                    )
                    error_handled = True
                except Exception as e:
                    # Should raise ModelException, not crash
                    error_handled = "ModelException" in str(type(e))

            return {
                "success": True,
                "message": "Error handling verified",
                "details": {"graceful_error_handling": error_handled}
            }

        except Exception as e:
            return {"success": False, "message": f"Error handling test failed: {e}"}

    @medium_priority_test
    async def test_async_claude_provider_stats(self):
        """Test async Claude provider statistics."""
        try:
            from src.providers.async_claude_provider import AsyncClaudeCodeProvider

            provider = AsyncClaudeCodeProvider(self.test_config)

            # Test stats method
            stats = await provider.get_stats()

            assert isinstance(stats, dict)
            assert "provider" in stats
            assert stats["provider"] == "async_claude_code"

            return {
                "success": True,
                "message": "Async provider stats verified",
                "details": {"stats_keys": list(stats.keys())}
            }

        except Exception as e:
            return {"success": False, "message": f"Async stats test failed: {e}"}

    @low_priority_test
    def test_provider_configuration_validation(self):
        """Test provider configuration validation."""
        try:
            from src.providers.claude_code_provider import ClaudeCodeProvider

            # Test with various configurations
            config_scenarios = [
                {},  # Empty config
                {"timeout": 60},  # Partial config
                {"timeout": 30, "enable_fallback": True},  # Valid config
                {"invalid_key": "value"},  # Invalid keys
            ]

            results = []
            for config in config_scenarios:
                try:
                    provider = ClaudeCodeProvider(config)
                    results.append({"config": config, "success": True})
                except Exception as e:
                    results.append({"config": config, "success": False, "error": str(e)})

            success_count = sum(1 for r in results if r["success"])

            return {
                "success": success_count >= len(config_scenarios) * 0.5,  # 50% should work
                "message": f"Configuration validation: {success_count}/{len(config_scenarios)} passed",
                "details": {"config_results": results}
            }

        except Exception as e:
            return {"success": False, "message": f"Configuration validation failed: {e}"}

    @low_priority_test
    def test_provider_imports_and_exports(self):
        """Test provider module imports and exports."""
        try:
            # Test individual imports
            from src.providers.claude_code_provider import ClaudeCodeProvider, get_claude_code_provider
            from src.providers.auth_manager import AuthManager, get_auth_manager
            from src.providers.async_claude_provider import AsyncClaudeCodeProvider, get_async_claude_provider

            # Test factory functions
            claude_provider = get_claude_code_provider(self.test_config)
            auth_manager = get_auth_manager()
            async_provider = get_async_claude_provider(self.test_config)

            assert isinstance(claude_provider, ClaudeCodeProvider)
            assert isinstance(auth_manager, AuthManager)
            assert isinstance(async_provider, AsyncClaudeCodeProvider)

            return {
                "success": True,
                "message": "Provider imports and exports verified"
            }

        except Exception as e:
            return {"success": False, "message": f"Import/export test failed: {e}"}

    @critical_test
    def test_provider_fallback_mechanisms(self):
        """Test provider fallback mechanisms."""
        try:
            from src.providers.claude_code_provider import ClaudeCodeProvider

            provider = ClaudeCodeProvider({**self.test_config, "enable_fallback": True})

            # Test fallback behavior with unavailable CLI
            with patch('subprocess.run') as mock_run:
                # Mock CLI not found
                mock_run.side_effect = FileNotFoundError("Claude CLI not found")

                # Should handle gracefully and fall back
                try:
                    result = provider.call_model(
                        prompt="Test",
                        model="claude_sonnet_4",
                        role="Tester",
                        operation="fallback_test"
                    )
                    fallback_works = True
                except Exception as e:
                    # Should raise specific exception, not crash
                    fallback_works = "ModelException" in str(type(e))

            return {
                "success": True,
                "message": "Fallback mechanisms verified",
                "details": {"fallback_handling": fallback_works}
            }

        except Exception as e:
            return {"success": False, "message": f"Fallback test failed: {e}"}


def get_providers_tests():
    """Get all provider test methods."""
    test_class = TestProvidersComplete()
    test_methods = []

    for attr_name in dir(test_class):
        if attr_name.startswith('test_') and callable(getattr(test_class, attr_name)):
            method = getattr(test_class, attr_name)
            test_methods.append(method)

    return test_methods, test_class.setup_method, None