"""Tests for enhanced Enterprise Agent features."""
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.providers.auth_manager import ClaudeAuthManager
from src.utils.config_validator import ConfigValidator
from src.utils.notifications import (
    NotificationLevel,
    NotificationManager,
    NotificationType,
    notify_authentication_issue,
    notify_cli_failure,
)
from src.utils.usage_monitor import UsageMonitor, UsageWindow


class TestEnhancedAuthManager:
    """Test enhanced authentication manager."""

    def setup_method(self):
        """Setup test environment."""
        self.auth_manager = ClaudeAuthManager()

    def teardown_method(self):
        """Clean up test environment."""
        # Reset global state to prevent test pollution
        from src.providers.auth_manager import reset_auth_manager
        from src.utils.notifications import reset_notification_manager
        from src.utils.usage_monitor import reset_usage_monitor

        reset_auth_manager()
        reset_notification_manager()
        reset_usage_monitor()

        # Clean up environment variables
        import os
        for key in ['ANTHROPIC_API_KEY', 'USE_CLAUDE_CODE']:
            if key in os.environ:
                del os.environ[key]

    @patch("subprocess.run")
    def test_verify_claude_status_json(self, mock_run):
        """Test JSON status verification."""
        # Mock successful JSON response
        mock_run.return_value = Mock(
            returncode=0, stdout='{"authenticated": true, "plan": "max"}', stderr=""
        )

        result = self.auth_manager.verify_claude_status_json()
        assert result["authenticated"] is True
        assert result["plan"] == "max"

    @patch("subprocess.run")
    def test_verify_claude_status_fallback(self, mock_run):
        """Test fallback when JSON not available."""
        # Mock command that doesn't support JSON
        mock_run.return_value = Mock(
            returncode=1, stdout="", stderr="unknown flag: --json"
        )

        with patch.object(self.auth_manager, "is_logged_in", return_value=True):
            result = self.auth_manager.verify_claude_status_json()
            assert result["authenticated"] is True
            assert result["method"] == "fallback"

    @patch("subprocess.run")
    def test_validate_setup_comprehensive(self, mock_run):
        """Test comprehensive setup validation."""
        # Mock CLI available
        mock_run.return_value = Mock(
            returncode=0, stdout="1.0.120 (Claude Code)", stderr=""
        )

        with patch.object(
            self.auth_manager, "verify_claude_status_json"
        ) as mock_status:
            mock_status.return_value = {"authenticated": True}

            result = self.auth_manager.validate_setup()

            assert result["cli_available"] is True
            assert result["authenticated"] is True
            assert result["status"] == "ready"
            assert result["ready"] is True

    def test_validate_setup_cli_missing(self):
        """Test validation when CLI is missing."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = self.auth_manager.validate_setup()

            assert result["cli_available"] is False
            assert "Claude Code CLI not installed" in result["issues"]
            assert any("npm install" in rec for rec in result["recommendations"])

    def test_api_key_conflict_detection(self):
        """Test API key conflict detection."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            result = self.auth_manager.validate_setup()

            assert result["api_key_conflict"] is True
            assert not result["subscription_mode"]
            assert any("API billing" in issue for issue in result["issues"])


class TestNotificationSystem:
    """Test notification system."""

    def setup_method(self):
        """Setup test environment."""
        self.notification_manager = NotificationManager()
        self.notification_manager.handlers = (
            []
        )  # Remove default console handler for testing

    def teardown_method(self):
        """Clean up test environment."""
        # Reset global notification state
        from src.utils.notifications import reset_notification_manager
        reset_notification_manager()

    def test_notification_creation(self):
        """Test notification creation."""
        self.notification_manager.notify(
            type=NotificationType.CLI_FAILURE,
            level=NotificationLevel.ERROR,
            title="Test Failure",
            message="Test message",
            action_required=True,
            recommendations=["Fix it"],
        )

        assert len(self.notification_manager.notifications) == 1
        notification = self.notification_manager.notifications[0]
        assert notification.type == NotificationType.CLI_FAILURE
        assert notification.level == NotificationLevel.ERROR
        assert notification.action_required is True
        assert "Fix it" in notification.recommendations

    def test_notification_filtering(self):
        """Test notification filtering."""
        # Add different types of notifications
        self.notification_manager.notify(
            NotificationType.CLI_FAILURE,
            NotificationLevel.ERROR,
            "CLI Error",
            "CLI failed",
        )
        self.notification_manager.notify(
            NotificationType.AUTHENTICATION,
            NotificationLevel.WARNING,
            "Auth Warning",
            "Auth issue",
        )

        # Filter by type
        cli_notifications = self.notification_manager.get_notifications(
            type=NotificationType.CLI_FAILURE
        )
        assert len(cli_notifications) == 1
        assert cli_notifications[0].title == "CLI Error"

        # Filter by level
        error_notifications = self.notification_manager.get_notifications(
            level=NotificationLevel.ERROR
        )
        assert len(error_notifications) == 1

    def test_notification_clearing(self):
        """Test notification clearing."""
        # Add notifications
        self.notification_manager.notify(
            NotificationType.CLI_FAILURE,
            NotificationLevel.ERROR,
            "Error 1",
            "Message 1",
        )
        self.notification_manager.notify(
            NotificationType.AUTHENTICATION,
            NotificationLevel.WARNING,
            "Warning 1",
            "Message 2",
        )

        assert len(self.notification_manager.notifications) == 2

        # Clear by type
        cleared = self.notification_manager.clear_notifications(
            type=NotificationType.CLI_FAILURE
        )
        assert cleared == 1
        assert len(self.notification_manager.notifications) == 1

    def test_cli_failure_notification(self):
        """Test CLI failure notification."""
        from src.utils.notifications import clear_notifications, get_notifications

        clear_notifications()  # Clear any existing notifications
        notify_cli_failure("test_operation", "Connection failed", fallback_used=True)

        notifications = get_notifications()
        cli_notifications = [n for n in notifications if n.type == NotificationType.CLI_FAILURE]
        assert len(cli_notifications) == 1
        assert "test_operation" in cli_notifications[0].title
        assert "API fallback" in str(cli_notifications[0].recommendations)

    def test_authentication_issue_notification(self):
        """Test authentication issue notification."""
        from src.utils.notifications import clear_notifications, get_notifications

        clear_notifications()  # Clear any existing notifications
        notify_authentication_issue("not_logged_in")

        notifications = get_notifications()
        auth_notifications = [n for n in notifications if n.type == NotificationType.AUTHENTICATION]
        assert len(auth_notifications) == 1
        assert any("claude login" in rec for rec in auth_notifications[0].recommendations)


class TestUsageMonitor:
    """Test usage monitoring system."""

    def setup_method(self):
        """Setup test environment."""
        # Use temporary file for testing
        self.temp_dir = tempfile.mkdtemp()
        config = {
            "max_prompts_per_window": 10,  # Small limit for testing
            "window_hours": 1,
            "warning_threshold": 80,
        }
        self.usage_monitor = UsageMonitor(config)
        self.usage_monitor.usage_file = Path(self.temp_dir) / "test_usage.json"
        self.usage_monitor._reset_pause_state()  # Ensure clean state for testing
        self.usage_monitor.reset_usage()  # Clear any existing usage windows

    def teardown_method(self):
        """Clean up test environment."""
        # Reset global usage monitor state
        from src.utils.usage_monitor import reset_usage_monitor
        from src.utils.notifications import reset_notification_manager

        reset_usage_monitor()
        reset_notification_manager()

        # Clean up temp directory
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_usage_window_creation(self):
        """Test usage window creation."""
        window = UsageWindow(start_time=time.time(), end_time=time.time() + 3600)

        assert window.prompt_count == 0
        assert len(window.requests) == 0

        window.add_request("Coder", "generate", 100)
        assert window.prompt_count == 1
        assert len(window.requests) == 1

    def test_usage_recording(self):
        """Test usage recording."""
        # Record several requests
        for i in range(5):
            success = self.usage_monitor.record_request("Coder", f"operation_{i}", 100)
            assert success is True

        stats = self.usage_monitor.get_usage_stats()
        assert stats["current_usage"] == 5
        assert stats["limit"] == 10
        assert stats["usage_percentage"] == 50.0

    def test_usage_limit_enforcement(self):
        """Test usage limit enforcement."""
        # Fill up to limit
        for i in range(10):
            success = self.usage_monitor.record_request("Coder", f"operation_{i}")
            assert success is True

        # Next request should be blocked
        success = self.usage_monitor.record_request("Coder", "blocked_operation")
        assert success is False
        assert self.usage_monitor.paused is True

    def test_usage_warning_threshold(self):
        """Test usage warning threshold."""
        # Record requests up to warning threshold (80% of 10 = 8)
        for i in range(8):
            self.usage_monitor.record_request("Coder", f"operation_{i}")

        # Should have triggered warning notification
        from src.utils.notifications import get_notification_manager

        notifications = get_notification_manager().get_notifications(
            type=NotificationType.USAGE_WARNING
        )
        assert len(notifications) > 0

    def test_usage_persistence(self):
        """Test usage persistence."""
        # Record some usage
        self.usage_monitor.record_request("Coder", "test_operation")

        # Create new monitor instance (simulating restart)
        new_monitor = UsageMonitor(self.usage_monitor.config)
        new_monitor.usage_file = self.usage_monitor.usage_file

        # Should load previous usage
        stats = new_monitor.get_usage_stats()
        assert stats["current_usage"] >= 1

    def test_pause_and_unpause(self):
        """Test manual pause/unpause."""
        assert self.usage_monitor.can_make_request() is True

        self.usage_monitor.force_pause(0.1)  # Pause for 0.1 hours
        assert self.usage_monitor.can_make_request() is False

        self.usage_monitor.unpause()
        assert self.usage_monitor.can_make_request() is True


class TestConfigValidator:
    """Test configuration validator."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"

    def teardown_method(self):
        """Clean up test environment."""
        # Reset global state
        from src.utils.notifications import reset_notification_manager
        from src.utils.usage_monitor import reset_usage_monitor
        from src.providers.auth_manager import reset_auth_manager

        reset_notification_manager()
        reset_usage_monitor()
        reset_auth_manager()

        # Clean up temp directory
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Clean up environment variables
        import os
        for key in ['ANTHROPIC_API_KEY', 'USE_CLAUDE_CODE']:
            if key in os.environ:
                del os.environ[key]

    def create_test_config(self, content: str):
        """Create test configuration file."""
        self.config_file.write_text(content)

    def test_valid_config_validation(self):
        """Test validation of valid configuration."""
        config_content = """
enterprise_coding_agent:
  orchestration:
    max_workers: 4
  memory:
    enabled: true
  governance:
    enabled: true
"""
        self.create_test_config(config_content)

        validator = ConfigValidator(str(self.config_file))
        results = validator.validate_all()

        assert results["valid"] is True
        assert len(results["issues"]) == 0

    def test_missing_config_file(self):
        """Test validation with missing config file."""
        validator = ConfigValidator("nonexistent.yaml")
        results = validator.validate_all()

        assert results["valid"] is False
        assert any("not found" in issue["message"] for issue in results["issues"])

    def test_invalid_yaml_syntax(self):
        """Test validation with invalid YAML."""
        invalid_yaml = """
invalid: yaml: content
  - missing
    indentation
"""
        self.create_test_config(invalid_yaml)

        validator = ConfigValidator(str(self.config_file))
        results = validator.validate_all()

        assert results["valid"] is False
        assert any("YAML syntax" in issue["message"] for issue in results["issues"])

    def test_environment_validation(self):
        """Test environment validation."""
        validator = ConfigValidator()

        with patch.dict(
            os.environ, {"USE_CLAUDE_CODE": "true", "ANTHROPIC_API_KEY": "sk-test"}
        ):
            results = validator.validate_all()

            # Should detect API key conflict
            assert any("API billing" in issue["message"] for issue in results["issues"])

    @patch("subprocess.run")
    def test_dependency_validation(self, mock_run):
        """Test dependency validation."""
        # Mock Claude CLI not found
        mock_run.side_effect = FileNotFoundError

        validator = ConfigValidator()
        results = validator.validate_all()

        dep_results = results["dependencies"]
        assert dep_results["claude_cli"]["installed"] is False
        assert any(
            "Claude Code CLI" in warning["message"] for warning in results["warnings"]
        )

    def test_security_validation(self):
        """Test security validation."""
        config_content = """
enterprise_coding_agent:
  api_key: "sk-dangerous-key-in-config"
  claude_code:
    auto_mode: true
"""
        self.create_test_config(config_content)

        validator = ConfigValidator(str(self.config_file))
        results = validator.validate_all()

        security_results = results["security"]
        assert security_results["api_key_exposure"] is True
        assert "auto_mode enabled" in security_results["insecure_settings"]

    def test_recommendation_generation(self):
        """Test recommendation generation."""
        validator = ConfigValidator("nonexistent.yaml")
        results = validator.validate_all()

        recommendations = results["recommendations"]
        assert len(recommendations) > 0
        assert any("configuration file" in rec for rec in recommendations)


class TestIntegration:
    """Test integration between components."""

    def setup_method(self):
        """Setup test environment."""
        # Reset global state before each test
        from src.utils.notifications import reset_notification_manager, clear_notifications
        from src.utils.usage_monitor import reset_usage_monitor
        from src.providers.auth_manager import reset_auth_manager

        reset_notification_manager()
        reset_usage_monitor()
        reset_auth_manager()
        clear_notifications()

    def teardown_method(self):
        """Clean up test environment."""
        # Reset global state after each test
        from src.utils.notifications import reset_notification_manager
        from src.utils.usage_monitor import reset_usage_monitor
        from src.providers.auth_manager import reset_auth_manager

        reset_notification_manager()
        reset_usage_monitor()
        reset_auth_manager()

    @patch("subprocess.run")
    def test_auth_notification_integration(self, mock_run):
        """Test integration between auth manager and notifications."""
        # Mock CLI failure - this happens in ClaudeCodeProvider, not AuthManager
        mock_run.side_effect = FileNotFoundError

        # Try to create a ClaudeCodeProvider which should trigger the notification
        from src.providers.claude_code_provider import ClaudeCodeProvider
        from src.exceptions import ModelException

        try:
            ClaudeCodeProvider({})
        except ModelException:
            pass  # Expected to fail

        # Should generate notification
        from src.utils.notifications import get_notification_manager

        notifications = get_notification_manager().get_notifications(
            type=NotificationType.AUTHENTICATION
        )
        assert len(notifications) > 0

    def test_usage_monitor_notification_integration(self):
        """Test integration between usage monitor and notifications."""
        config = {"max_prompts_per_window": 1, "warning_threshold": 100}
        monitor = UsageMonitor(config)

        # Ensure monitor starts in unpaused state
        monitor._reset_pause_state()
        monitor.reset_usage()

        # Record request that should trigger limit and notification
        success = monitor.record_request("Test", "operation")
        assert success is True  # First request should succeed

        # Record second request that should hit the limit
        success = monitor.record_request("Test", "operation2")
        assert success is False  # Second request should fail (hit limit)

        # Should have paused and sent notification
        assert monitor.paused is True

        from src.utils.notifications import get_notification_manager

        notifications = get_notification_manager().get_notifications(
            type=NotificationType.USAGE_WARNING
        )
        assert len(notifications) > 0


if __name__ == "__main__":
    pytest.main([__file__])
