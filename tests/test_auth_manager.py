"""Unit tests for Claude Code authentication manager."""
import json
import os
import unittest
from unittest.mock import Mock, patch

from src.providers.auth_manager import ClaudeAuthManager, get_auth_manager


class TestClaudeAuthManager(unittest.TestCase):
    """Test cases for Claude authentication manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.auth_manager = ClaudeAuthManager()
        self.original_api_key = os.environ.get("ANTHROPIC_API_KEY")

    def tearDown(self):
        """Clean up after tests."""
        # Restore original API key if it existed
        if self.original_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.original_api_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_ensure_subscription_mode_removes_api_key(self):
        """Test that API key is removed from environment."""
        # Set API key
        os.environ["ANTHROPIC_API_KEY"] = "test-api-key"

        result = self.auth_manager.ensure_subscription_mode()

        self.assertTrue(result)
        self.assertNotIn("ANTHROPIC_API_KEY", os.environ)

    def test_ensure_subscription_mode_no_api_key(self):
        """Test behavior when no API key is present."""
        # Ensure no API key
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        result = self.auth_manager.ensure_subscription_mode()

        self.assertTrue(result)

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.write_text")
    def test_remove_api_key_from_env_file(self, mock_write, mock_read, mock_exists):
        """Test removing API key from .env file."""
        mock_exists.return_value = True
        mock_read.return_value = """
OPENAI_API_KEY=sk-openai-123
ANTHROPIC_API_KEY=sk-ant-456
OTHER_VAR=value
"""

        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        self.auth_manager.ensure_subscription_mode()

        # Check that the file was modified
        mock_write.assert_called_once()
        written_content = mock_write.call_args[0][0]
        self.assertIn("# ANTHROPIC_API_KEY", written_content)
        self.assertIn("Disabled for Claude Code subscription mode", written_content)
        self.assertIn("OPENAI_API_KEY=sk-openai-123", written_content)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_is_logged_in_success(self, mock_exists, mock_run):
        """Test successful login check."""
        mock_exists.return_value = True  # Token file exists
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        result = self.auth_manager.is_logged_in()

        self.assertTrue(result)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_is_logged_in_no_token(self, mock_exists, mock_run):
        """Test login check with no token file."""
        mock_exists.return_value = False  # No token file

        result = self.auth_manager.is_logged_in()

        self.assertFalse(result)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_is_logged_in_needs_login(self, mock_exists, mock_run):
        """Test login check when login is needed."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(
            returncode=1, stdout="", stderr="please run `claude login`"
        )

        result = self.auth_manager.is_logged_in()

        self.assertFalse(result)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_is_logged_in_unauthorized(self, mock_exists, mock_run):
        """Test login check with unauthorized error."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(
            returncode=1, stdout="", stderr="unauthorized access"
        )

        result = self.auth_manager.is_logged_in()

        self.assertFalse(result)

    @patch("subprocess.run")
    def test_login_already_logged_in(self, mock_run):
        """Test login when already logged in."""
        with patch.object(self.auth_manager, "is_logged_in", return_value=True):
            result = self.auth_manager.login()

            self.assertTrue(result)
            mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_login_interactive_success(self, mock_run):
        """Test successful interactive login."""
        mock_run.return_value = Mock(returncode=0)

        with patch.object(self.auth_manager, "is_logged_in", return_value=False):
            result = self.auth_manager.login(interactive=True)

            self.assertTrue(result)
            mock_run.assert_called_once_with(
                ["claude", "login"], capture_output=False, text=True
            )

    @patch("subprocess.run")
    def test_login_interactive_failure(self, mock_run):
        """Test failed interactive login."""
        mock_run.return_value = Mock(returncode=1)

        with patch.object(self.auth_manager, "is_logged_in", return_value=False):
            result = self.auth_manager.login(interactive=True)

            self.assertFalse(result)

    def test_login_non_interactive(self):
        """Test non-interactive login (not supported)."""
        with patch.object(self.auth_manager, "is_logged_in", return_value=False):
            result = self.auth_manager.login(interactive=False)

            self.assertFalse(result)

    @patch("subprocess.run")
    def test_setup_token_success(self, mock_run):
        """Test successful token setup."""
        mock_run.return_value = Mock(
            returncode=0, stdout="Token created successfully", stderr=""
        )

        result = self.auth_manager.setup_token()

        self.assertTrue(result)

    @patch("subprocess.run")
    def test_setup_token_failure(self, mock_run):
        """Test failed token setup."""
        mock_run.return_value = Mock(
            returncode=1, stdout="", stderr="Failed to create token"
        )

        result = self.auth_manager.setup_token()

        self.assertFalse(result)

    def test_verify_subscription_plan_with_api_key(self):
        """Test plan verification with API key present."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        with patch.object(self.auth_manager, "is_logged_in", return_value=True):
            plan_info = self.auth_manager.verify_subscription_plan()

            self.assertTrue(plan_info["authenticated"])
            self.assertTrue(plan_info["using_api_key"])
            self.assertIn("Remove ANTHROPIC_API_KEY", plan_info["recommendations"][0])

    def test_verify_subscription_plan_not_authenticated(self):
        """Test plan verification when not authenticated."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        with patch.object(self.auth_manager, "is_logged_in", return_value=False):
            plan_info = self.auth_manager.verify_subscription_plan()

            self.assertFalse(plan_info["authenticated"])
            self.assertFalse(plan_info["using_api_key"])
            self.assertIn("Run 'claude login'", plan_info["recommendations"][0])

    def test_verify_subscription_plan_authenticated_no_api(self):
        """Test plan verification when properly authenticated."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        with patch.object(self.auth_manager, "is_logged_in", return_value=True):
            plan_info = self.auth_manager.verify_subscription_plan()

            self.assertTrue(plan_info["authenticated"])
            self.assertFalse(plan_info["using_api_key"])
            self.assertEqual(plan_info["plan_type"], "Max subscription (assumed)")
            self.assertEqual(len(plan_info["recommendations"]), 0)

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_get_config(self, mock_read, mock_exists):
        """Test configuration retrieval."""
        mock_exists.return_value = True
        mock_read.return_value = json.dumps({"key": "value"})

        config = self.auth_manager.get_config()

        self.assertEqual(config, {"key": "value"})

    @patch("pathlib.Path.exists")
    def test_get_config_no_file(self, mock_exists):
        """Test configuration retrieval when file doesn't exist."""
        mock_exists.return_value = False

        config = self.auth_manager.get_config()

        self.assertEqual(config, {})

    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.mkdir")
    def test_update_config(self, mock_mkdir, mock_write):
        """Test configuration update."""
        with patch.object(
            self.auth_manager, "get_config", return_value={"old": "value"}
        ):
            result = self.auth_manager.update_config({"new": "data"})

            self.assertTrue(result)
            mock_write.assert_called_once()
            written_data = json.loads(mock_write.call_args[0][0])
            self.assertEqual(written_data, {"old": "value", "new": "data"})

    @patch("subprocess.run")
    def test_setup_for_automation_full_flow(self, mock_run):
        """Test full automation setup flow."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        with patch.object(
            self.auth_manager, "ensure_subscription_mode", return_value=True
        ):
            with patch.object(self.auth_manager, "is_logged_in", return_value=True):
                with patch.object(self.auth_manager, "setup_token", return_value=True):
                    with patch.object(
                        self.auth_manager, "update_config", return_value=True
                    ) as mock_update:
                        result = self.auth_manager.setup_for_automation()

                        self.assertTrue(result)
                        mock_update.assert_called_once()
                        config = mock_update.call_args[0][0]
                        self.assertFalse(config["auto_accept"])
                        self.assertFalse(config["verbose"])
                        self.assertEqual(config["theme"], "none")

    def test_setup_for_automation_not_logged_in(self):
        """Test automation setup when not logged in."""
        with patch.object(
            self.auth_manager, "ensure_subscription_mode", return_value=True
        ):
            with patch.object(self.auth_manager, "is_logged_in", return_value=False):
                result = self.auth_manager.setup_for_automation()

                self.assertFalse(result)

    def test_get_auth_manager_singleton(self):
        """Test singleton pattern for auth manager."""
        manager1 = get_auth_manager()
        manager2 = get_auth_manager()

        self.assertIs(manager1, manager2)


if __name__ == "__main__":
    unittest.main()
