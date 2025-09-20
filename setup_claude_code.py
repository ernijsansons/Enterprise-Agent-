#!/usr/bin/env python
"""Setup script for Claude Code CLI integration with Enterprise Agent.

This script helps users configure their Enterprise Agent to use Claude Code CLI
(included in Anthropic Max subscription) instead of the API (which costs extra).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


class Colors:
    """Terminal colors for output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str) -> None:
    """Print a header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def check_claude_cli_installed() -> bool:
    """Check if Claude Code CLI is installed."""
    try:
        result = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print_success(f"Claude Code CLI is installed: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print_error("Claude Code CLI is not installed")
    return False


def install_claude_cli() -> bool:
    """Install Claude Code CLI using npm."""
    print_info("Installing Claude Code CLI...")

    # Check if npm is available
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print_error(
            "npm is not installed. Please install Node.js first: https://nodejs.org/"
        )
        return False

    # Install Claude Code CLI
    try:
        print("Running: npm install -g @anthropic-ai/claude-code")
        result = subprocess.run(
            ["npm", "install", "-g", "@anthropic-ai/claude-code"],
            capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            print_success("Claude Code CLI installed successfully")
            return True
        else:
            print_error("Failed to install Claude Code CLI")
            return False
    except Exception as e:
        print_error(f"Installation failed: {e}")
        return False


def check_claude_login() -> bool:
    """Check if user is logged in to Claude Code."""
    try:
        result = subprocess.run(
            ["claude", "--print", "--model", "haiku", "echo test"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if "please run `claude login`" in result.stderr.lower():
            print_warning("Not logged in to Claude Code")
            return False

        if "unauthorized" in result.stderr.lower():
            print_warning("Authentication expired or invalid")
            return False

        if result.returncode == 0:
            print_success("Logged in to Claude Code")
            return True
    except Exception:
        pass

    print_warning("Could not verify login status")
    return False


def login_to_claude() -> bool:
    """Log in to Claude Code with subscription."""
    print_info("Logging in to Claude Code...")
    print_info("This will open a browser window for authentication")
    print_info("Please log in with your Anthropic account that has Max subscription")

    input("\nPress Enter to continue...")

    try:
        result = subprocess.run(["claude", "login"], capture_output=False)
        if result.returncode == 0:
            print_success("Successfully logged in to Claude Code")
            return True
        else:
            print_error("Login failed")
            return False
    except Exception as e:
        print_error(f"Login failed: {e}")
        return False


def setup_token() -> bool:
    """Set up a long-lived authentication token."""
    print_info("Setting up long-lived authentication token...")

    try:
        result = subprocess.run(
            ["claude", "setup-token"], capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print_success("Authentication token set up successfully")
            return True
        else:
            print_warning("Token setup failed (optional feature)")
            return False
    except Exception as e:
        print_warning(f"Token setup failed (optional): {e}")
        return False


def check_api_key() -> bool:
    """Check if ANTHROPIC_API_KEY is set and warn user."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        print_warning("ANTHROPIC_API_KEY is set in environment!")
        print_warning(
            "This will cause Claude Code to use API billing instead of your subscription"
        )
        print_info("To use your Max subscription, remove the API key from:")
        print_info("  - Environment variables")
        print_info("  - .env file")
        print_info("  - Any shell configuration files")
        return False

    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        content = env_file.read_text()
        if "ANTHROPIC_API_KEY" in content and not content.split("ANTHROPIC_API_KEY")[
            0
        ].endswith("#"):
            print_warning("ANTHROPIC_API_KEY found in .env file!")
            print_info("Comment it out or remove it to use your subscription")
            return False

    print_success("No ANTHROPIC_API_KEY found (good for subscription usage)")
    return True


def create_env_file() -> bool:
    """Create .env file from template."""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        print_info(".env file already exists")
        response = input("Do you want to update it for Claude Code? (y/n): ").lower()
        if response != "y":
            return True

    if not env_example.exists():
        print_error(".env.example file not found")
        return False

    # Copy the example file
    shutil.copy2(env_example, env_file)
    print_success("Created .env file from .env.example")

    # Update the file with Claude Code settings
    content = env_file.read_text()

    # Set USE_CLAUDE_CODE=true
    content = content.replace("USE_CLAUDE_CODE=false", "USE_CLAUDE_CODE=true")
    content = content.replace("# USE_CLAUDE_CODE=true", "USE_CLAUDE_CODE=true")

    # Comment out ANTHROPIC_API_KEY if present
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith(
            "ANTHROPIC_API_KEY="
        ) and not line.strip().startswith("#"):
            new_lines.append(f"# {line}  # Commented out for Claude Code subscription")
        else:
            new_lines.append(line)

    env_file.write_text("\n".join(new_lines))
    print_success("Updated .env file for Claude Code usage")

    print_info("\nPlease update the following in .env:")
    print_info("  - OPENAI_API_KEY (for embeddings)")
    print_info("  - Any other API keys you need")

    return True


def verify_setup() -> bool:
    """Verify the complete setup."""
    print_header("Verifying Setup")

    all_good = True

    # Check CLI installed
    if not check_claude_cli_installed():
        all_good = False

    # Check login
    if not check_claude_login():
        all_good = False

    # Check API key
    if not check_api_key():
        all_good = False

    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        content = env_file.read_text()
        if "USE_CLAUDE_CODE=true" in content:
            print_success(".env file configured for Claude Code")
        else:
            print_warning(".env file not configured for Claude Code")
            all_good = False
    else:
        print_warning(".env file not found")
        all_good = False

    return all_good


def main():
    """Main setup flow."""
    print_header("Claude Code Setup for Enterprise Agent")

    print("This script will help you set up Claude Code CLI to use with")
    print("your Anthropic Max subscription ($200/month) instead of paying")
    print("for API usage on top of your subscription.\n")

    # Step 1: Check/Install CLI
    if not check_claude_cli_installed():
        response = input("\nDo you want to install Claude Code CLI? (y/n): ").lower()
        if response == "y":
            if not install_claude_cli():
                print_error("Setup failed: Could not install Claude Code CLI")
                return 1
        else:
            print_error("Claude Code CLI is required. Please install manually:")
            print("npm install -g @anthropic-ai/claude-code")
            return 1

    # Step 2: Check/Perform login
    if not check_claude_login():
        response = input("\nDo you want to log in to Claude Code? (y/n): ").lower()
        if response == "y":
            if not login_to_claude():
                print_error("Setup failed: Could not log in")
                return 1
        else:
            print_warning("You'll need to log in manually later:")
            print("claude login")

    # Step 3: Set up token (optional)
    response = input("\nDo you want to set up a long-lived token? (y/n): ").lower()
    if response == "y":
        setup_token()

    # Step 4: Check API key situation
    check_api_key()

    # Step 5: Create/Update .env file
    response = input("\nDo you want to create/update .env file? (y/n): ").lower()
    if response == "y":
        create_env_file()

    # Final verification
    print_header("Setup Complete")

    if verify_setup():
        print_success("\nEverything is set up correctly!")
        print_info("\nYou can now run the Enterprise Agent with:")
        print("python src/agent_orchestrator.py")
        print_info(
            "\nYour agent will use Claude Code (covered by your Max subscription)"
        )
        print_info("instead of the API (which would cost extra).")
    else:
        print_warning("\nSetup is incomplete. Please address the issues above.")
        print_info("\nYou can run this script again to complete setup:")
        print("python setup_claude_code.py")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        sys.exit(1)
