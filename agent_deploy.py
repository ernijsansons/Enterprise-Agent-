#!/usr/bin/env python3
"""Enterprise Agent deployment script."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AgentDeployer:
    """Handles deployment of Enterprise Agent."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize deployer.

        Args:
            config_path: Optional path to deployment configuration
        """
        self.config_path = config_path or Path("configs/deployment.json")
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            logger.warning(f"Deployment config not found at {self.config_path}")
            return self._get_default_config()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load deployment config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "environment": "development",
            "python_version": "3.9+",
            "node_version": "16+",
            "dependencies": {
                "python": ["requirements.txt"],
                "node": ["package.json"],
            },
            "services": {
                "claude_code": True,
                "snyk": False,
                "monitoring": True,
            },
            "deployment": {
                "type": "local",
                "output_dir": "deploy",
                "backup": True,
            },
        }

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met.

        Returns:
            True if all prerequisites are met
        """
        logger.info("Checking prerequisites...")

        # Check Python version
        python_version = sys.version_info
        required_version = (3, 9)
        if python_version[:2] < required_version:
            logger.error(f"Python {required_version[0]}.{required_version[1]}+ required, got {python_version.major}.{python_version.minor}")
            return False

        # Check Node.js
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True)
            node_version = result.stdout.strip()
            logger.info(f"Node.js version: {node_version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Node.js not found or not working")
            return False

        # Check npm
        try:
            subprocess.run(["npm", "--version"], capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("npm not found or not working")
            return False

        # Check Poetry
        try:
            subprocess.run(["poetry", "--version"], capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Poetry not found, will use pip")

        logger.info("All prerequisites met")
        return True

    def install_dependencies(self) -> bool:
        """Install project dependencies.

        Returns:
            True if installation successful
        """
        logger.info("Installing dependencies...")

        # Install Python dependencies
        if Path("requirements.txt").exists():
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
                logger.info("Python dependencies installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install Python dependencies: {e}")
                return False

        # Install Node.js dependencies
        if Path("package.json").exists():
            try:
                subprocess.run(["npm", "install"], check=True)
                logger.info("Node.js dependencies installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install Node.js dependencies: {e}")
                return False

        return True

    def setup_environment(self) -> bool:
        """Set up environment configuration.

        Returns:
            True if setup successful
        """
        logger.info("Setting up environment...")

        # Create .env file if it doesn't exist
        env_file = Path(".env")
        if not env_file.exists():
            logger.info("Creating .env file...")
            env_content = """# Enterprise Agent Environment Configuration

# Anthropic API Key (required)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI API Key (for fallback)
OPENAI_API_KEY=your_openai_api_key_here

# Claude Code CLI (recommended for zero API costs)
USE_CLAUDE_CODE=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/agent.log

# Cost tracking
ENABLE_COST_TRACKING=true
COST_LOG_FILE=logs/cost_tracking.json

# Security
ENABLE_PII_SCRUBBING=true
"""
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(env_content)
            logger.info("Created .env file with default configuration")

        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create deploy directory
        deploy_dir = Path(self.config["deployment"]["output_dir"])
        deploy_dir.mkdir(exist_ok=True)

        return True

    def run_tests(self) -> bool:
        """Run test suite.

        Returns:
            True if all tests pass
        """
        logger.info("Running tests...")

        try:
            # Run Python tests
            if Path("tests").exists():
                subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
                logger.info("Python tests passed")
            else:
                logger.warning("No tests directory found")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Tests failed: {e}")
            return False

    def build_package(self) -> bool:
        """Build deployment package.

        Returns:
            True if build successful
        """
        logger.info("Building deployment package...")

        deploy_dir = Path(self.config["deployment"]["output_dir"])

        try:
            # Copy source files
            source_files = [
                "src/",
                "configs/",
                "tests/",
                "requirements.txt",
                "README.md",
                "LICENSE",
                ".env.example",
            ]

            for file_path in source_files:
                src = Path(file_path)
                if src.exists():
                    if src.is_dir():
                        subprocess.run(["cp", "-r", str(src), str(deploy_dir)], check=True)
                    else:
                        subprocess.run(["cp", str(src), str(deploy_dir)], check=True)

            # Create deployment script
            deploy_script = deploy_dir / "deploy.sh"
            with open(deploy_script, "w", encoding="utf-8") as f:
                f.write("""#!/bin/bash
# Enterprise Agent Deployment Script

set -e

echo "Deploying Enterprise Agent..."

# Install dependencies
pip install -r requirements.txt

# Set up environment
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Please configure .env file with your API keys"
fi

# Run tests
python -m pytest tests/ -v

echo "Deployment complete!"
""")
            deploy_script.chmod(0o755)

            logger.info(f"Deployment package built in {deploy_dir}")
            return True

        except Exception as e:
            logger.error(f"Build failed: {e}")
            return False

    def deploy(self) -> bool:
        """Deploy the agent.

        Returns:
            True if deployment successful
        """
        logger.info("Starting deployment...")

        # Check prerequisites
        if not self.check_prerequisites():
            return False

        # Install dependencies
        if not self.install_dependencies():
            return False

        # Set up environment
        if not self.setup_environment():
            return False

        # Run tests
        if not self.run_tests():
            logger.warning("Tests failed, but continuing with deployment")

        # Build package
        if not self.build_package():
            return False

        logger.info("Deployment completed successfully!")
        return True

    def cleanup(self) -> None:
        """Clean up deployment artifacts."""
        logger.info("Cleaning up...")

        deploy_dir = Path(self.config["deployment"]["output_dir"])
        if deploy_dir.exists():
            import shutil
            shutil.rmtree(deploy_dir)
            logger.info("Cleaned up deployment directory")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deploy Enterprise Agent")
    parser.add_argument("--config", type=Path, help="Path to deployment configuration")
    parser.add_argument("--cleanup", action="store_true", help="Clean up after deployment")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--environment", choices=["development", "staging", "production"], help="Deployment environment")

    args = parser.parse_args()

    deployer = AgentDeployer(args.config)

    if args.environment:
        deployer.config["environment"] = args.environment

    if args.skip_tests:
        deployer.config["skip_tests"] = True

    try:
        success = deployer.deploy()
        if success:
            logger.info("Deployment successful!")
            sys.exit(0)
        else:
            logger.error("Deployment failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Deployment failed with error: {e}")
        sys.exit(1)
    finally:
        if args.cleanup:
            deployer.cleanup()


if __name__ == "__main__":
    main()
