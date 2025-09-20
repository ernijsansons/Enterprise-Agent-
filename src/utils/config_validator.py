"""Configuration validation utilities for Enterprise Agent."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.utils.notifications import notify_configuration_issue


class ConfigValidator:
    """Validates Enterprise Agent configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration validator.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "configs/agent_config_v3.4.yaml"
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []

    def validate_all(self) -> Dict[str, Any]:
        """Perform comprehensive configuration validation.

        Returns:
            Validation results with issues and recommendations
        """
        results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "environment": {},
            "dependencies": {},
            "security": {}
        }

        # Validate configuration file
        config_valid, config_data = self._validate_config_file()
        if not config_valid:
            results["valid"] = False

        # Validate environment
        env_results = self._validate_environment()
        results["environment"] = env_results

        # Validate dependencies
        dep_results = self._validate_dependencies()
        results["dependencies"] = dep_results

        # Validate security settings
        security_results = self._validate_security(config_data)
        results["security"] = security_results

        # Validate Claude Code integration
        claude_results = self._validate_claude_code()
        results["claude_code"] = claude_results

        # Collect all issues and warnings
        results["issues"] = self.issues
        results["warnings"] = self.warnings

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations()

        # Overall validation status
        results["valid"] = len(self.issues) == 0

        return results

    def _validate_config_file(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate the main configuration file.

        Returns:
            Tuple of (is_valid, config_data)
        """
        try:
            config_path = Path(self.config_path)
            if not config_path.exists():
                self._add_issue(
                    "config_file",
                    f"Configuration file not found: {config_path}",
                    "critical"
                )
                return False, {}

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                self._add_issue(
                    "config_file",
                    "Configuration file is empty or invalid",
                    "critical"
                )
                return False, {}

            # Validate required sections
            required_sections = [
                "enterprise_coding_agent",
                "enterprise_coding_agent.orchestration",
                "enterprise_coding_agent.memory",
                "enterprise_coding_agent.governance"
            ]

            for section in required_sections:
                if not self._get_nested_value(config_data, section):
                    self._add_warning(
                        "config_structure",
                        f"Missing configuration section: {section}"
                    )

            return True, config_data

        except yaml.YAMLError as e:
            self._add_issue(
                "config_file",
                f"Invalid YAML syntax: {e}",
                "critical"
            )
            return False, {}
        except Exception as e:
            self._add_issue(
                "config_file",
                f"Error reading configuration: {e}",
                "critical"
            )
            return False, {}

    def _validate_environment(self) -> Dict[str, Any]:
        """Validate environment variables and settings.

        Returns:
            Environment validation results
        """
        env_results = {
            "claude_code_enabled": False,
            "api_keys_present": {},
            "conflicts": [],
            "missing_required": []
        }

        # Check Claude Code setting
        use_claude_code = os.getenv("USE_CLAUDE_CODE", "false").lower() == "true"
        env_results["claude_code_enabled"] = use_claude_code

        # Check API keys
        api_keys = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
        }

        for key, value in api_keys.items():
            env_results["api_keys_present"][key] = bool(value and value != "STUBBED_FALLBACK")

        # Check for conflicts
        if use_claude_code and env_results["api_keys_present"]["ANTHROPIC_API_KEY"]:
            conflict = "ANTHROPIC_API_KEY present with Claude Code enabled - will cause API billing"
            env_results["conflicts"].append(conflict)
            self._add_issue(
                "environment",
                conflict,
                "high"
            )

        # Check required environment for different modes
        if use_claude_code:
            # Claude Code mode requirements are checked separately
            pass
        else:
            # API mode requirements
            if not any(env_results["api_keys_present"].values()):
                self._add_issue(
                    "environment",
                    "No API keys available - system will run in stub mode",
                    "medium"
                )

        return env_results

    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate system dependencies.

        Returns:
            Dependency validation results
        """
        dep_results = {
            "python_version": None,
            "claude_cli": {
                "installed": False,
                "version": None,
                "working": False
            },
            "required_packages": {},
            "missing_dependencies": []
        }

        # Check Python version
        import sys
        dep_results["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        if sys.version_info < (3, 8):
            self._add_issue(
                "dependencies",
                f"Python {dep_results['python_version']} is too old. Requires Python 3.8+",
                "critical"
            )

        # Check Claude CLI
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                dep_results["claude_cli"]["installed"] = True
                dep_results["claude_cli"]["version"] = result.stdout.strip()
                dep_results["claude_cli"]["working"] = True
            else:
                dep_results["claude_cli"]["installed"] = True
                dep_results["claude_cli"]["working"] = False
                self._add_warning(
                    "dependencies",
                    "Claude CLI installed but not responding correctly"
                )
        except FileNotFoundError:
            self._add_warning(
                "dependencies",
                "Claude Code CLI not installed - required for subscription mode"
            )
        except subprocess.TimeoutExpired:
            dep_results["claude_cli"]["installed"] = True
            dep_results["claude_cli"]["working"] = False
            self._add_warning(
                "dependencies",
                "Claude CLI timeout - may be network issue"
            )

        # Check required Python packages
        required_packages = [
            "yaml", "anthropic", "openai", "requests"
        ]

        for package in required_packages:
            try:
                __import__(package)
                dep_results["required_packages"][package] = True
            except ImportError:
                dep_results["required_packages"][package] = False
                dep_results["missing_dependencies"].append(package)

        if dep_results["missing_dependencies"]:
            self._add_issue(
                "dependencies",
                f"Missing required packages: {', '.join(dep_results['missing_dependencies'])}",
                "high"
            )

        return dep_results

    def _validate_security(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate security settings.

        Args:
            config_data: Configuration data

        Returns:
            Security validation results
        """
        security_results = {
            "pii_scrubbing_enabled": True,  # Default enabled
            "secret_handling": True,        # Default enabled
            "api_key_exposure": False,
            "insecure_settings": []
        }

        # Check for API keys in config files (security risk)
        config_str = str(config_data)
        if "sk-" in config_str or "api_key" in config_str.lower():
            security_results["api_key_exposure"] = True
            self._add_issue(
                "security",
                "Potential API key found in configuration file",
                "critical"
            )

        # Check for insecure auto-mode settings
        auto_mode = self._get_nested_value(config_data, "claude_code.auto_mode")
        if auto_mode:
            security_results["insecure_settings"].append("auto_mode enabled")
            self._add_warning(
                "security",
                "Auto-mode enabled - dangerous in production"
            )

        return security_results

    def _validate_claude_code(self) -> Dict[str, Any]:
        """Validate Claude Code specific configuration.

        Returns:
            Claude Code validation results
        """
        from src.providers.auth_manager import get_auth_manager

        claude_results = {
            "enabled": False,
            "cli_available": False,
            "authenticated": False,
            "subscription_mode": False,
            "ready": False
        }

        # Check if enabled
        use_claude_code = os.getenv("USE_CLAUDE_CODE", "false").lower() == "true"
        claude_results["enabled"] = use_claude_code

        if use_claude_code:
            try:
                auth_manager = get_auth_manager()
                validation = auth_manager.validate_setup()

                claude_results.update({
                    "cli_available": validation["cli_available"],
                    "authenticated": validation["authenticated"],
                    "subscription_mode": validation["subscription_mode"],
                    "ready": validation["ready"]
                })

                # Add issues from auth validation
                for issue in validation["issues"]:
                    self._add_issue("claude_code", issue, "medium")

            except Exception as e:
                self._add_issue(
                    "claude_code",
                    f"Error validating Claude Code setup: {e}",
                    "high"
                )

        return claude_results

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation.

        Args:
            data: Dictionary to search
            path: Dot-separated path (e.g., "a.b.c")

        Returns:
            Value at path or None if not found
        """
        keys = path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _add_issue(self, category: str, message: str, severity: str) -> None:
        """Add validation issue.

        Args:
            category: Issue category
            message: Issue description
            severity: Severity level (critical, high, medium, low)
        """
        issue = {
            "category": category,
            "message": message,
            "severity": severity
        }
        self.issues.append(issue)

    def _add_warning(self, category: str, message: str) -> None:
        """Add validation warning.

        Args:
            category: Warning category
            message: Warning description
        """
        warning = {
            "category": category,
            "message": message
        }
        self.warnings.append(warning)

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results.

        Returns:
            List of recommendations
        """
        recommendations = []

        # Group issues by category
        issues_by_category = {}
        for issue in self.issues:
            category = issue["category"]
            if category not in issues_by_category:
                issues_by_category[category] = []
            issues_by_category[category].append(issue)

        # Generate category-specific recommendations
        if "config_file" in issues_by_category:
            recommendations.append("Fix configuration file issues before proceeding")

        if "environment" in issues_by_category:
            recommendations.append("Review environment variable configuration")
            recommendations.append("Remove ANTHROPIC_API_KEY when using Claude Code")

        if "dependencies" in issues_by_category:
            recommendations.append("Install missing dependencies: pip install -r requirements.txt")
            recommendations.append("Install Claude Code CLI: npm install -g @anthropic-ai/claude-code")

        if "security" in issues_by_category:
            recommendations.append("Address security issues immediately")
            recommendations.append("Remove API keys from configuration files")

        if "claude_code" in issues_by_category:
            recommendations.append("Complete Claude Code setup: claude login")
            recommendations.append("Verify Claude Code authentication status")

        # General recommendations based on warnings
        warning_categories = set(w["category"] for w in self.warnings)

        if "security" in warning_categories:
            recommendations.append("Review security settings for production use")

        if not recommendations:
            recommendations.append("Configuration validation passed - system ready")

        return recommendations

    def validate_and_notify(self) -> bool:
        """Validate configuration and send notifications for issues.

        Returns:
            True if validation passed, False otherwise
        """
        results = self.validate_all()

        # Send notifications for issues
        for issue in results["issues"]:
            if issue["severity"] in ["critical", "high"]:
                notify_configuration_issue(
                    f"{issue['category']}: {issue['message']}",
                    results["recommendations"]
                )

        return results["valid"]


def validate_enterprise_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Validate Enterprise Agent configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Validation results
    """
    validator = ConfigValidator(config_path)
    return validator.validate_all()


def quick_health_check() -> bool:
    """Perform quick health check of system configuration.

    Returns:
        True if system is healthy, False otherwise
    """
    validator = ConfigValidator()

    # Quick checks only
    try:
        # Check config file exists
        if not Path(validator.config_path).exists():
            return False

        # Check Claude Code if enabled
        if os.getenv("USE_CLAUDE_CODE", "false").lower() == "true":
            try:
                subprocess.run(
                    ["claude", "--version"],
                    capture_output=True,
                    timeout=3
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return False

        return True

    except Exception:
        return False


__all__ = [
    "ConfigValidator",
    "validate_enterprise_config",
    "quick_health_check"
]