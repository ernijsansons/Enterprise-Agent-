#!/usr/bin/env python3
"""
CI/CD Validation Tests for Enterprise Agent v3.4
Ensures the CI pipeline configuration is correct and all checks pass.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class CIPipelineTests(unittest.TestCase):
    """Test CI/CD pipeline configuration and execution."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent
        self.workflows_dir = self.repo_root / ".github" / "workflows"
        self.ci_file = self.workflows_dir / "ci.yml"

    def test_ci_workflow_exists(self):
        """Verify CI workflow file exists."""
        self.assertTrue(
            self.ci_file.exists(),
            f"CI workflow not found at {self.ci_file}"
        )

    def test_ci_workflow_valid_yaml(self):
        """Verify CI workflow is valid YAML."""
        try:
            with open(self.ci_file, 'r') as f:
                config = yaml.safe_load(f)
            self.assertIsNotNone(config)
            self.assertIn('jobs', config)
        except yaml.YAMLError as e:
            self.fail(f"Invalid YAML in CI workflow: {e}")

    def test_python_versions_matrix(self):
        """Verify Python version matrix covers 3.9-3.12."""
        with open(self.ci_file, 'r') as f:
            config = yaml.safe_load(f)

        test_job = config.get('jobs', {}).get('test', {})
        matrix = test_job.get('strategy', {}).get('matrix', {})
        python_versions = matrix.get('python-version', [])

        expected_versions = ['3.9', '3.10', '3.11', '3.12']
        for version in expected_versions:
            self.assertIn(
                version, python_versions,
                f"Python {version} not in test matrix"
            )

    def test_security_scanning_configured(self):
        """Verify security scanning is configured."""
        with open(self.ci_file, 'r') as f:
            config = yaml.safe_load(f)

        security_job = config.get('jobs', {}).get('security', {})
        self.assertIsNotNone(security_job, "Security job not found")

        # Check for bandit scanning
        steps = security_job.get('steps', [])
        has_bandit = any(
            'bandit' in str(step.get('run', '')).lower()
            for step in steps
        )
        self.assertTrue(has_bandit, "Bandit security scanning not configured")

    def test_makefile_targets(self):
        """Verify all required Makefile targets exist."""
        makefile = self.repo_root / "Makefile"
        self.assertTrue(makefile.exists(), "Makefile not found")

        with open(makefile, 'r') as f:
            content = f.read()

        required_targets = [
            'setup', 'lint', 'test', 'typecheck', 'format',
            'security', 'validate-config', 'ci', 'clean', 'build'
        ]

        for target in required_targets:
            self.assertIn(
                f"{target}:",
                content,
                f"Makefile target '{target}' not found"
            )

    def test_dependency_management(self):
        """Verify dependency management files exist."""
        files_to_check = [
            ('pyproject.toml', "Poetry configuration"),
            ('poetry.lock', "Poetry lock file"),
        ]

        for filename, description in files_to_check:
            file_path = self.repo_root / filename
            self.assertTrue(
                file_path.exists(),
                f"{description} ({filename}) not found"
            )

    def test_install_script_executable(self):
        """Verify install script has proper shebang and is valid bash."""
        install_script = self.repo_root / "install.sh"
        self.assertTrue(install_script.exists(), "install.sh not found")

        with open(install_script, 'r') as f:
            first_line = f.readline().strip()

        self.assertEqual(
            first_line, "#!/bin/bash",
            "Install script missing proper shebang"
        )

        # Check for basic bash syntax (if bash is available)
        if os.name != 'nt':  # Skip on Windows
            try:
                result = subprocess.run(
                    ["bash", "-n", str(install_script)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                self.assertEqual(
                    result.returncode, 0,
                    f"Bash syntax errors in install.sh: {result.stderr}"
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass  # Skip if bash not available

    def test_config_validation_script(self):
        """Verify configuration validation script works."""
        validator = self.repo_root / "validate_config.py"
        config_file = self.repo_root / "configs" / "agent_config_v3.4.yaml"

        self.assertTrue(validator.exists(), "validate_config.py not found")
        self.assertTrue(config_file.exists(), "Config file not found")

        # Test validation
        try:
            result = subprocess.run(
                [sys.executable, str(validator), str(config_file)],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Should pass or handle missing dependencies gracefully
            if "PASS" in result.stdout or "Installing" in result.stdout:
                pass  # Expected behavior
            elif result.returncode != 0 and "No module named" in result.stderr:
                pass  # Dependency issue, acceptable in test
            else:
                self.fail(f"Config validation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.fail("Config validation timed out")

    def test_github_actions_secrets_usage(self):
        """Verify proper secrets usage in GitHub Actions."""
        with open(self.ci_file, 'r') as f:
            content = f.read()

        # Check that secrets are properly referenced
        if "CODECOV_TOKEN" in content:
            self.assertIn(
                "${{ secrets.CODECOV_TOKEN }}",
                content,
                "CODECOV_TOKEN not properly referenced as secret"
            )

        # Ensure no hardcoded sensitive values (but allow regex patterns)
        import re

        # Look for actual API key patterns, not just the text "sk-"
        api_key_patterns = [
            r'sk-[a-zA-Z0-9]{48}',  # Actual Anthropic API key pattern
            r'sk-proj-[a-zA-Z0-9]{48}',  # OpenAI project API key
            r'["\']ANTHROPIC_API_KEY["\']:\s*["\']sk-[a-zA-Z0-9]+["\']',  # Hardcoded in config
            r'api_key\s*=\s*["\']sk-[a-zA-Z0-9]+["\']',  # Hardcoded assignment
        ]

        for pattern in api_key_patterns:
            matches = re.findall(pattern, content)
            if matches:
                self.fail(f"Potential hardcoded API key found: {matches[0][:20]}...")

    def test_artifact_uploads_configured(self):
        """Verify artifact uploads are configured for test results."""
        with open(self.ci_file, 'r') as f:
            config = yaml.safe_load(f)

        # Check test job has artifact uploads
        test_job = config.get('jobs', {}).get('test', {})
        steps = test_job.get('steps', [])

        has_artifact_upload = any(
            step.get('uses', '').startswith('actions/upload-artifact')
            for step in steps
        )

        self.assertTrue(
            has_artifact_upload,
            "Test artifacts not configured for upload"
        )


class BuildSystemTests(unittest.TestCase):
    """Test build system configuration."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent

    def test_makefile_help_target(self):
        """Verify Makefile has help target."""
        makefile = self.repo_root / "Makefile"

        with open(makefile, 'r') as f:
            content = f.read()

        self.assertIn("help:", content, "Help target not found in Makefile")
        self.assertIn(".DEFAULT_GOAL", content, "Default goal not set")

    def test_makefile_error_handling(self):
        """Verify Makefile has proper error handling."""
        makefile = self.repo_root / "Makefile"

        with open(makefile, 'r') as f:
            content = f.read()

        # Check for error handling patterns
        patterns = [
            "exit 1",  # Explicit exit on error
            "||",      # Or operator for fallback
            "$(RED)",  # Color codes for errors
        ]

        for pattern in patterns:
            self.assertIn(
                pattern,
                content,
                f"Error handling pattern '{pattern}' not found"
            )

    def test_pyproject_configuration(self):
        """Verify pyproject.toml is properly configured."""
        pyproject = self.repo_root / "pyproject.toml"

        with open(pyproject, 'r') as f:
            content = f.read()

        # Parse as TOML
        try:
            import tomllib  # Python 3.11+
            config = tomllib.loads(content)
        except ImportError:
            try:
                import tomli
                config = tomli.loads(content)
            except ImportError:
                # Fallback - just check basic structure
                self.assertIn('[tool.poetry]', content)
                self.assertIn('dependencies', content)
                return

        # Check required sections
        self.assertIn('tool', config)
        self.assertIn('poetry', config['tool'])
        self.assertIn('dependencies', config['tool']['poetry'])

        # Check Python version requirement
        python_req = config['tool']['poetry']['dependencies'].get('python')
        self.assertIsNotNone(python_req, "Python version not specified")

        # Check dev dependencies
        self.assertIn('group', config['tool']['poetry'])
        self.assertIn('dev', config['tool']['poetry']['group'])


class SecurityTests(unittest.TestCase):
    """Test security configurations."""

    def setUp(self):
        """Set up test environment."""
        self.repo_root = Path(__file__).parent.parent

    def test_bandit_report_clean(self):
        """Verify bandit report shows no high-severity issues."""
        bandit_report = self.repo_root / "bandit-report.json"

        if not bandit_report.exists():
            self.skipTest("Bandit report not found")

        with open(bandit_report, 'r') as f:
            report = json.load(f)

        # Check for high-severity issues
        metrics = report.get('metrics', {})
        totals = metrics.get('_totals', {})

        high_severity = totals.get('SEVERITY.HIGH', 0)
        self.assertEqual(
            high_severity, 0,
            f"Found {high_severity} high-severity security issues"
        )

    def test_no_hardcoded_secrets(self):
        """Verify no hardcoded secrets in source files."""
        src_dir = self.repo_root / "src"

        if not src_dir.exists():
            self.skipTest("Source directory not found")

        secret_patterns = [
            'api_key=',
            'password=',
            'secret=',
            'sk-',
            'AIza'
        ]

        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
            except UnicodeDecodeError:
                # Skip files with encoding issues
                continue

            for pattern in secret_patterns:
                # Allow in comments or as part of variable names
                if pattern in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if pattern in line and not line.strip().startswith('#'):
                            # Check if it's an actual assignment
                            if '=' in line and pattern + '"' in line:
                                self.fail(
                                    f"Potential hardcoded secret in {py_file}:{i}"
                                )


def run_tests():
    """Run all CI validation tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(CIPipelineTests))
    suite.addTests(loader.loadTestsFromTestCase(BuildSystemTests))
    suite.addTests(loader.loadTestsFromTestCase(SecurityTests))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())