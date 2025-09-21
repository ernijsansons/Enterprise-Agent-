"""Authentication manager for Claude Code subscription."""
from __future__ import annotations

import json
import logging
import os
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ClaudeAuthManager:
    """Manages Claude Code subscription authentication."""

    def __init__(self):
        """Initialize authentication manager."""
        self.config_path = Path.home() / ".claude" / "config.json"
        self.token_path = Path.home() / ".claude" / "token"

    def ensure_subscription_mode(self) -> bool:
        """Ensure Claude Code is in subscription mode by removing API key if present.

        Returns:
            True if in subscription mode (after removing API key if needed)
        """
        # Check for API key in environment
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if api_key:
            logger.warning(
                "ANTHROPIC_API_KEY detected! Removing from environment to use Max subscription "
                "instead of API billing. Please also remove ANTHROPIC_API_KEY from your .env file."
            )
            # Remove from current environment
            del os.environ["ANTHROPIC_API_KEY"]
            logger.info(
                "ANTHROPIC_API_KEY removed from environment. Now using subscription mode."
            )

            # Also check and update .env files
            self._remove_api_key_from_env_files()

        return True

    def _remove_api_key_from_env_files(self) -> None:
        """Remove ANTHROPIC_API_KEY from .env files in current directory."""
        env_files = [Path(".env"), Path(".env.local"), Path(".env.production")]

        for env_file in env_files:
            if env_file.exists():
                try:
                    content = env_file.read_text(encoding="utf-8")
                    modified_content = self._comment_out_api_key(content)
                    if modified_content != content:
                        env_file.write_text(modified_content, encoding="utf-8")
                        logger.info(f"Commented out ANTHROPIC_API_KEY in {env_file}")
                except Exception as e:
                    logger.warning(f"Could not modify {env_file}: {e}")

    def _comment_out_api_key(self, content: str) -> str:
        """Comment out ANTHROPIC_API_KEY lines in env file content."""
        lines = content.split("\n")
        modified_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("ANTHROPIC_API_KEY=") and not stripped.startswith(
                "#"
            ):
                # Comment out the line
                modified_lines.append(
                    f"# {line}  # Disabled for Claude Code subscription mode"
                )
            else:
                modified_lines.append(line)

        return "\n".join(modified_lines)

    def check_env_file_for_api_key(self, env_file: Path) -> bool:
        """Check if ANTHROPIC_API_KEY is present in .env file.

        Args:
            env_file: Path to .env file

        Returns:
            True if API key is present, False otherwise
        """
        try:
            if not env_file.exists():
                return False

            lines = env_file.read_text().splitlines()
            for line in lines:
                if line.strip().startswith(
                    "ANTHROPIC_API_KEY"
                ) and not line.strip().startswith("#"):
                    logger.warning(
                        f"ANTHROPIC_API_KEY found in {env_file}. "
                        "Please comment it out to use subscription mode."
                    )
                    return True
            return False

        except Exception as e:
            logger.error(f"Failed to check .env file: {e}")
            return False

    def is_logged_in(self) -> bool:
        """Check if user is logged in to Claude Code.

        Returns:
            True if logged in, False otherwise
        """
        try:
            # Check if token file exists
            if not self.token_path.exists():
                return False

            # Try a minimal command to verify auth
            result = subprocess.run(  # nosec B603, B607
                ["claude", "--print", "--model", "haiku", "echo test"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=False,  # Explicitly disable shell to prevent injection
            )

            # Check for login prompt in output
            if "please run `claude login`" in result.stderr.lower():
                return False

            if "unauthorized" in result.stderr.lower():
                return False

            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to check login status: {e}")
            return False

    def login(self, interactive: bool = True) -> bool:
        """Log in to Claude Code with subscription.

        Args:
            interactive: Whether to use interactive login

        Returns:
            True if login successful, False otherwise
        """
        if self.is_logged_in():
            logger.info("Already logged in to Claude Code")
            return True

        try:
            if interactive:
                logger.info(
                    "Please log in to Claude Code with your Max subscription.\n"
                    "This will open a browser window for authentication."
                )

                # Run interactive login
                result = subprocess.run(  # nosec B603, B607
                    ["claude", "login"],
                    capture_output=False,  # Allow user interaction
                    text=True,
                    shell=False,  # Explicitly disable shell
                )

                if result.returncode == 0:
                    logger.info("Successfully logged in to Claude Code")
                    return True
                else:
                    logger.error("Login failed")
                    return False

            else:
                logger.error(
                    "Non-interactive login not supported. "
                    "Please run 'claude login' manually."
                )
                return False

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def setup_token(self) -> bool:
        """Set up a long-lived authentication token.

        Returns:
            True if token setup successful, False otherwise
        """
        try:
            logger.info("Setting up long-lived authentication token...")

            result = subprocess.run(  # nosec B603, B607
                ["claude", "setup-token"], capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                logger.info("Authentication token set up successfully")
                return True
            else:
                logger.error(f"Token setup failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to set up token: {e}")
            return False

    def verify_claude_status_json(self) -> Dict[str, Any]:
        """Verify Claude Code status using JSON output for automated parsing.

        Returns:
            Detailed status information
        """
        try:
            result = subprocess.run(  # nosec B603, B607
                ["claude", "auth", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout.strip():
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # Fallback to text parsing
                    return self._parse_auth_status_text(result.stdout)
            else:
                # Command failed or no JSON support, use fallback
                return self._fallback_auth_check()

        except Exception as e:
            logger.warning(f"JSON auth status check failed: {e}")
            return self._fallback_auth_check()

    def _parse_auth_status_text(self, output: str) -> Dict[str, Any]:
        """Parse text-based auth status output."""
        authenticated = (
            "logged in" in output.lower() or "authenticated" in output.lower()
        )
        return {
            "authenticated": authenticated,
            "plan_type": "subscription" if authenticated else "unknown",
            "status": "active" if authenticated else "not_logged_in",
            "raw_output": output,
        }

    def _fallback_auth_check(self) -> Dict[str, Any]:
        """Fallback authentication check using existing methods."""
        authenticated = self.is_logged_in()
        return {
            "authenticated": authenticated,
            "plan_type": "subscription" if authenticated else "unknown",
            "status": "active" if authenticated else "not_logged_in",
            "method": "fallback",
        }

    def validate_setup(self) -> Dict[str, Any]:
        """Comprehensive setup validation with actionable recommendations.

        Returns:
            Detailed validation results with recommendations
        """
        validation: Dict[str, Any] = {
            "cli_available": False,
            "cli_version": None,
            "authenticated": False,
            "subscription_mode": False,
            "api_key_conflict": False,
            "status": "unknown",
            "issues": [],
            "recommendations": [],
            "ready": False,
        }

        # Check CLI availability
        try:
            result = subprocess.run(  # nosec B603, B607
                ["claude", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                validation["cli_available"] = True
                validation["cli_version"] = result.stdout.strip()
            else:
                validation["issues"].append("Claude Code CLI not responding properly")
                validation["recommendations"].append(
                    "Reinstall Claude Code CLI: npm install -g @anthropic-ai/claude-code"
                )
        except FileNotFoundError:
            validation["issues"].append("Claude Code CLI not installed")
            validation["recommendations"].append(
                "Install Claude Code CLI: npm install -g @anthropic-ai/claude-code"
            )
        except Exception as e:
            validation["issues"].append(f"CLI check failed: {str(e)}")

        # Check authentication
        if validation["cli_available"]:
            auth_status = self.verify_claude_status_json()
            validation["authenticated"] = auth_status.get("authenticated", False)

            if not validation["authenticated"]:
                validation["issues"].append("Not logged in to Claude Code")
                validation["recommendations"].append(
                    "Log in with your Max subscription: claude login"
                )

        # Check for API key conflicts
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key and api_key != "STUBBED_FALLBACK":
            validation["api_key_conflict"] = True
            validation["subscription_mode"] = False
            validation["issues"].append(
                "ANTHROPIC_API_KEY will cause API billing instead of subscription"
            )
            validation["recommendations"].append(
                "Remove ANTHROPIC_API_KEY from environment: unset ANTHROPIC_API_KEY"
            )
            validation["recommendations"].append(
                "Comment out ANTHROPIC_API_KEY in .env file"
            )
        else:
            validation["subscription_mode"] = True

        # Overall status
        if (
            validation["cli_available"]
            and validation["authenticated"]
            and validation["subscription_mode"]
        ):
            validation["status"] = "ready"
            validation["ready"] = True
        elif validation["cli_available"] and not validation["authenticated"]:
            validation["status"] = "needs_login"
        elif not validation["cli_available"]:
            validation["status"] = "needs_installation"
        else:
            validation["status"] = "needs_configuration"

        return validation

    def verify_subscription_plan(self) -> Dict[str, Any]:
        """Verify the subscription plan details.

        Returns:
            Dictionary with plan information
        """
        if not self.is_logged_in():
            return {
                "authenticated": False,
                "using_api_key": False,
                "plan_type": "Unknown",
                "recommendations": [
                    "Run 'claude login' to authenticate with your Max subscription"
                ],
            }

        if os.getenv("ANTHROPIC_API_KEY"):
            return {
                "authenticated": True,
                "using_api_key": True,
                "plan_type": "Max subscription (API mode - will incur charges)",
                "recommendations": [
                    "Remove ANTHROPIC_API_KEY from environment variables"
                ],
            }
        else:
            return {
                "authenticated": True,
                "using_api_key": False,
                "plan_type": "Max subscription (assumed)",
                "recommendations": [],
            }

    def get_config(self) -> Dict[str, Any]:
        """Get Claude Code configuration.

        Returns:
            Configuration dictionary
        """
        try:
            if self.config_path.exists():
                return json.loads(self.config_path.read_text())
            return {}
        except Exception as e:
            logger.error(f"Failed to read config: {e}")
            return {}

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update Claude Code configuration.

        Args:
            updates: Configuration updates

        Returns:
            True if update successful, False otherwise
        """
        try:
            config = self.get_config()
            config.update(updates)

            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Write updated config
            self.config_path.write_text(json.dumps(config, indent=2))

            logger.info("Claude Code configuration updated")
            return True

        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False

    def setup_for_automation(self) -> bool:
        """Set up Claude Code for automated usage.

        Returns:
            True if setup successful, False otherwise
        """
        logger.info("Setting up Claude Code for automation...")

        # 1. Ensure subscription mode
        if not self.ensure_subscription_mode():
            return False

        # 2. Check login status
        if not self.is_logged_in():
            logger.error(
                "Not logged in to Claude Code. "
                "Please run 'claude login' manually first."
            )
            return False

        # 3. Set up long-lived token
        if not self.setup_token():
            logger.warning("Failed to set up long-lived token (optional)")

        # 4. Update configuration for automation
        config_updates = {
            "auto_accept": False,  # Still require confirmation for safety
            "verbose": False,  # Reduce noise in automation
            "theme": "none",  # No color codes in output
        }

        self.update_config(config_updates)

        logger.info("Claude Code setup for automation complete")
        return True


# Singleton instance
_auth_manager: Optional[ClaudeAuthManager] = None


def get_auth_manager() -> ClaudeAuthManager:
    """Get or create auth manager instance.

    Returns:
        ClaudeAuthManager instance
    """
    global _auth_manager

    if _auth_manager is None:
        _auth_manager = ClaudeAuthManager()

    return _auth_manager


def reset_auth_manager() -> None:
    """Reset the global auth manager instance for testing."""
    global _auth_manager
    _auth_manager = None


__all__ = ["ClaudeAuthManager", "get_auth_manager", "reset_auth_manager"]
