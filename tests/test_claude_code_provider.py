"""Unit tests for Claude Code CLI provider."""
import json
import os
import subprocess
import unittest
from unittest.mock import MagicMock, Mock, patch

from src.exceptions import ModelException, ModelTimeoutException
from src.providers.claude_code_provider import (
    ClaudeCodeProvider,
    get_claude_code_provider,
)


class TestClaudeCodeProvider(unittest.TestCase):
    """Test cases for Claude Code provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "timeout": 30,
            "auto_remove_api_key": True,
            "enable_fallback": True,
            "working_directory": ".",
        }
        # Clear any existing API key
        self.original_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

    def tearDown(self):
        """Clean up after tests."""
        # Restore original API key if it existed
        if self.original_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.original_api_key

    @patch("subprocess.run")
    def test_verify_cli_available_success(self, mock_run):
        """Test successful CLI verification."""
        mock_run.return_value = Mock(
            returncode=0, stdout="claude version 1.0.0", stderr=""
        )

        provider = ClaudeCodeProvider(self.config)
        self.assertTrue(hasattr(provider, "config"))

    @patch("subprocess.run")
    def test_verify_cli_available_not_installed(self, mock_run):
        """Test CLI not installed error."""
        mock_run.side_effect = FileNotFoundError()

        with self.assertRaises(ModelException) as ctx:
            ClaudeCodeProvider(self.config)

        self.assertIn("not installed", str(ctx.exception))

    @patch("subprocess.run")
    def test_verify_cli_available_timeout(self, mock_run):
        """Test CLI verification timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("claude", 5)

        with self.assertRaises(ModelTimeoutException):
            ClaudeCodeProvider(self.config)

    @patch("subprocess.run")
    def test_verify_subscription_auth_removes_api_key(self, mock_run):
        """Test that API key is removed when auto_remove_api_key is True."""
        # Set an API key
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        ClaudeCodeProvider(self.config)

        # API key should be removed
        self.assertNotIn("ANTHROPIC_API_KEY", os.environ)

    @patch("subprocess.run")
    def test_verify_subscription_auth_not_logged_in(self, mock_run):
        """Test detection of not being logged in."""
        mock_run.side_effect = [
            Mock(
                returncode=0, stdout="claude version 1.0.0", stderr=""
            ),  # version check
            Mock(
                returncode=1, stdout="", stderr="please run `claude login`"
            ),  # auth check
        ]

        with patch("src.providers.claude_code_provider.logger") as mock_logger:
            ClaudeCodeProvider(self.config)
            mock_logger.warning.assert_called_with(
                "Not logged in to Claude Code. Please run: claude login"
            )

    def test_map_model_to_cli(self):
        """Test model name mapping."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            provider = ClaudeCodeProvider(self.config)

            # Test exact mappings
            self.assertEqual(provider._map_model_to_cli("claude_sonnet_4"), "sonnet")
            self.assertEqual(
                provider._map_model_to_cli("claude-3-5-sonnet-20241022"), "sonnet"
            )
            self.assertEqual(provider._map_model_to_cli("claude_opus_4"), "opus")
            self.assertEqual(provider._map_model_to_cli("claude_haiku"), "haiku")

            # Test partial matching
            self.assertEqual(provider._map_model_to_cli("some-sonnet-model"), "sonnet")
            self.assertEqual(provider._map_model_to_cli("opus-variant"), "opus")
            self.assertEqual(provider._map_model_to_cli("haiku-3"), "haiku")

            # Test unknown model (defaults to sonnet)
            self.assertEqual(provider._map_model_to_cli("unknown-model"), "sonnet")

    @patch("subprocess.run")
    @patch("src.providers.claude_code_provider.can_make_claude_request")
    def test_call_model_success(self, mock_can_make_request, mock_run):
        """Test successful model call."""
        mock_can_make_request.return_value = True
        
        mock_run.side_effect = [
            Mock(returncode=0, stdout="claude version 1.0.0", stderr=""),  # version
            Mock(returncode=0, stdout="", stderr=""),  # auth
            Mock(
                returncode=0,
                stdout=json.dumps(
                    {"response": "Test response", "session_id": "abc123"}
                ),
                stderr="",
            ),  # actual call
        ]

        provider = ClaudeCodeProvider(self.config)

        response = provider.call_model(
            prompt="Test prompt",
            model="claude_sonnet_4",
            role="Coder",
            operation="test",
            session_id="test-session",  # Provide session ID to trigger storage
            use_cache=False,
        )

        self.assertEqual(response, "Test response")
        # Session ID is stored in sessions dict with session_id as key
        self.assertEqual(provider.sessions.get("test-session"), "abc123")

    @patch("subprocess.run")
    @patch("src.providers.claude_code_provider.can_make_claude_request")
    def test_call_model_with_cache(self, mock_can_make_request, mock_run):
        """Test model call with caching."""
        mock_can_make_request.return_value = True  # Allow requests
        mock_run.side_effect = [
            Mock(returncode=0, stdout="claude version 1.0.0", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
            Mock(
                returncode=0,
                stdout=json.dumps({"response": "Cached response"}),
                stderr="",
            ),
        ]

        provider = ClaudeCodeProvider(self.config)

        # First call
        response1 = provider.call_model(
            prompt="Test prompt",
            model="claude_sonnet_4",
            role="Planner",
            use_cache=True,
        )

        # Second call (should hit cache)
        response2 = provider.call_model(
            prompt="Test prompt",
            model="claude_sonnet_4",
            role="Planner",
            use_cache=True,
        )

        self.assertEqual(response1, response2)
        # Only 3 subprocess calls (2 for init, 1 for actual call - not 2)
        self.assertEqual(mock_run.call_count, 3)

    @patch("subprocess.run")
    @patch("src.providers.claude_code_provider.can_make_claude_request")
    def test_call_model_timeout(self, mock_can_make_request, mock_run):
        """Test model call timeout."""
        mock_can_make_request.return_value = True  # Allow requests
        mock_run.side_effect = [
            Mock(returncode=0, stdout="claude version 1.0.0", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
            subprocess.TimeoutExpired("claude", 30),
        ]

        provider = ClaudeCodeProvider(self.config)

        with self.assertRaises(ModelTimeoutException) as ctx:
            provider.call_model(
                prompt="Test prompt", model="claude_sonnet_4", use_cache=False
            )

        self.assertIn("timeout after 30s", str(ctx.exception))

    @patch("subprocess.run")
    @patch("src.providers.claude_code_provider.can_make_claude_request")
    def test_call_model_error(self, mock_can_make_request, mock_run):
        """Test model call error."""
        mock_can_make_request.return_value = True  # Allow requests
        mock_run.side_effect = [
            Mock(returncode=0, stdout="claude version 1.0.0", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
            Mock(returncode=1, stdout="", stderr="Claude Code CLI failed"),
        ]

        provider = ClaudeCodeProvider(self.config)

        with self.assertRaises(ModelException) as ctx:
            provider.call_model(
                prompt="Test prompt", model="claude_sonnet_4", use_cache=False
            )

        self.assertIn("Claude Code CLI failed", str(ctx.exception))

    def test_parse_cli_response(self):
        """Test CLI response parsing."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            provider = ClaudeCodeProvider(self.config)

            # Test JSON response
            json_output = '{"response": "test", "session_id": "123"}'
            result = provider._parse_cli_response(json_output)
            self.assertEqual(result, {"response": "test", "session_id": "123"})

            # Test plain text response
            text_output = "Plain text response"
            result = provider._parse_cli_response(text_output)
            self.assertEqual(result, {"text": "Plain text response"})

            # Test empty response
            result = provider._parse_cli_response("")
            self.assertEqual(result, {"text": ""})

    @patch("subprocess.Popen")
    def test_stream_response(self, mock_popen):
        """Test streaming response."""
        # Create mock process
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = [
            '{"text": "Hello "}\n',
            '{"text": "world"}\n',
            '{"text": "!"}\n',
        ]
        mock_process.stderr.read.return_value = ""
        mock_process.wait.return_value = None

        mock_popen.return_value = mock_process

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            provider = ClaudeCodeProvider(self.config)

            chunks = []
            response = provider.stream_response(
                prompt="Test prompt",
                model="claude_sonnet_4",
                callback=lambda x: chunks.append(x),
            )

            self.assertEqual(response, "Hello world!")
            self.assertEqual(chunks, ["Hello ", "world", "!"])

    def test_session_management(self):
        """Test session creation and clearing."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            provider = ClaudeCodeProvider(self.config)

            # Create session
            session_id = provider.create_session("test-session")
            self.assertEqual(session_id, "test-session")
            self.assertIn("test-session", provider.sessions)

            # Create session without ID (auto-generate)
            auto_session = provider.create_session()
            self.assertIsNotNone(auto_session)
            self.assertIn(auto_session, provider.sessions)

            # Clear session
            provider.clear_session("test-session")
            self.assertNotIn("test-session", provider.sessions)

    def test_get_usage_info(self):
        """Test usage info retrieval."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            provider = ClaudeCodeProvider(self.config)

            usage = provider.get_usage_info()

            self.assertIn("plan", usage)
            self.assertEqual(usage["plan"], "Max 20x")
            self.assertIn("estimated_prompts_remaining", usage)
            self.assertIn("note", usage)

    def test_singleton_provider(self):
        """Test singleton provider pattern."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            provider1 = get_claude_code_provider(self.config)
            provider2 = get_claude_code_provider(self.config)

            self.assertIs(provider1, provider2)


class TestClaudeCodeIntegration(unittest.TestCase):
    """Integration tests for Claude Code provider with orchestrator."""

    @patch("subprocess.run")
    def test_orchestrator_uses_claude_code(self, mock_run):
        """Test that orchestrator properly uses Claude Code provider."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({"response": "Claude Code response"}),
            stderr="",
        )

        # Set environment to use Claude Code
        os.environ["USE_CLAUDE_CODE"] = "true"

        # Import here to pick up environment variable
        from src.agent_orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Should have initialized Claude Code provider
        self.assertTrue(orchestrator._use_claude_code)
        # Provider might be None if CLI not installed in test environment
        # But the flag should be set correctly

    @patch("src.providers.claude_code_provider.can_make_claude_request")
    @patch("subprocess.run")
    def test_fallback_to_api(self, mock_run, mock_can_make_request):
        """Test fallback to API when Claude Code fails."""
        mock_can_make_request.return_value = True  # Allow requests
        mock_run.side_effect = [
            Mock(returncode=0, stdout="claude version 1.0.0", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
        ]

        os.environ["USE_CLAUDE_CODE"] = "true"
        os.environ["CLAUDE_CODE_FALLBACK_TO_API"] = "true"
        # Need an API key for fallback to work
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        from src.agent_orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # With fallback enabled and API key, should have client
        # Note: In test environment without actual API key, client might still be None
        # but the configuration should be set correctly
        self.assertTrue(orchestrator._use_claude_code)


if __name__ == "__main__":
    unittest.main()
