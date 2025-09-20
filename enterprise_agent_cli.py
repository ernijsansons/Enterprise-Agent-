#!/usr/bin/env python
"""Enterprise Agent CLI - Use agent across multiple projects."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent_orchestrator import AgentOrchestrator


class EnterpriseAgentCLI:
    """CLI interface for Enterprise Agent."""

    def __init__(self):
        """Initialize CLI."""
        self.home_dir = Path.home() / ".enterprise-agent"
        self.global_config = self.home_dir / "config.yml"
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure required directories exist."""
        self.home_dir.mkdir(exist_ok=True)
        (self.home_dir / "cache").mkdir(exist_ok=True)
        (self.home_dir / "logs").mkdir(exist_ok=True)
        (self.home_dir / "templates").mkdir(exist_ok=True)

    def init_project(self, project_dir: Optional[Path] = None) -> None:
        """Initialize agent in a project.

        Args:
            project_dir: Project directory to initialize
        """
        project_dir = project_dir or Path.cwd()
        agent_dir = project_dir / ".enterprise-agent"

        if agent_dir.exists():
            print(f"✓ Agent already initialized in {project_dir}")
            return

        # Create project structure
        agent_dir.mkdir()
        (agent_dir / "config.yml").write_text("""# Enterprise Agent Project Configuration
# This overrides global settings for this project

# Use Claude Code CLI instead of API (saves money with Max subscription)
use_claude_code: true

# Project-specific model preferences
models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini

# Domain for this project (coding, ui, social, content, trading, real_estate)
default_domain: coding

# Project context
context:
  description: "My project"
  tech_stack: []
  conventions: []

# Cache settings
cache:
  enabled: true
  ttl: 3600

# Custom templates directory (relative to .enterprise-agent/)
templates_dir: templates

# History tracking
history:
  enabled: true
  max_entries: 100
""")

        (agent_dir / "templates").mkdir()
        (agent_dir / "context").mkdir()
        (agent_dir / "history").mkdir()
        (agent_dir / ".gitignore").write_text("""cache/
history/
*.log
""")

        # Create example template
        (agent_dir / "templates" / "code_review.md").write_text("""# Code Review Template
Please review the following code for:
- Security vulnerabilities
- Performance issues
- Best practices
- Test coverage
""")

        print(f"✅ Initialized Enterprise Agent in {project_dir}")
        print(f"   Created .enterprise-agent/ directory")
        print(f"   Edit .enterprise-agent/config.yml to customize")

    def run_agent(
        self,
        domain: str = "coding",
        input_text: str = "",
        project_dir: Optional[Path] = None,
        config_file: Optional[Path] = None,
        interactive: bool = False,
    ) -> Any:
        """Run the agent with specified parameters.

        Args:
            domain: Domain to use
            input_text: Input prompt
            project_dir: Project directory
            config_file: Custom config file
            interactive: Interactive mode

        Returns:
            Agent result
        """
        project_dir = project_dir or Path.cwd()

        # Set working directory for agent
        original_dir = os.getcwd()
        os.chdir(project_dir)

        try:
            # Load project-specific config if exists
            project_config = project_dir / ".enterprise-agent" / "config.yml"
            if project_config.exists():
                self._load_project_config(project_config)

            # Override with custom config if provided
            if config_file and config_file.exists():
                self._load_project_config(config_file)

            # Initialize orchestrator
            orchestrator = AgentOrchestrator()

            if interactive:
                return self._run_interactive(orchestrator, domain)
            else:
                # Run single command
                print(f"🚀 Running Enterprise Agent")
                print(f"   Domain: {domain}")
                print(f"   Project: {project_dir}")
                print(f"   Input: {input_text[:100]}...")

                result = orchestrator.run(
                    domain=domain,
                    initial_state={"input": input_text}
                )

                # Save to history
                self._save_history(project_dir, domain, input_text, result)

                return result

        finally:
            os.chdir(original_dir)

    def _run_interactive(self, orchestrator: AgentOrchestrator, domain: str) -> None:
        """Run in interactive mode.

        Args:
            orchestrator: Agent orchestrator
            domain: Default domain
        """
        print("\n🤖 Enterprise Agent - Interactive Mode")
        print("Type 'help' for commands, 'exit' to quit\n")

        while True:
            try:
                user_input = input(f"[{domain}]> ").strip()

                if user_input.lower() == "exit":
                    break
                elif user_input.lower() == "help":
                    self._show_help()
                elif user_input.startswith("/domain "):
                    domain = user_input.split(" ", 1)[1]
                    print(f"Switched to domain: {domain}")
                elif user_input.startswith("/"):
                    print(f"Unknown command: {user_input}")
                elif user_input:
                    result = orchestrator.run(
                        domain=domain,
                        initial_state={"input": user_input}
                    )
                    print(f"\n{result}\n")

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")

    def _show_help(self) -> None:
        """Show interactive mode help."""
        print("""
Commands:
  /domain <name>  - Switch domain (coding, ui, social, etc.)
  /history       - Show command history
  /clear         - Clear screen
  help           - Show this help
  exit           - Exit interactive mode
""")

    def _load_project_config(self, config_file: Path) -> None:
        """Load project configuration.

        Args:
            config_file: Path to config file
        """
        if not config_file.exists():
            return

        import yaml
        config = yaml.safe_load(config_file.read_text())

        # Apply configuration to environment
        if config.get("use_claude_code"):
            os.environ["USE_CLAUDE_CODE"] = "true"

        if "models" in config:
            if "primary" in config["models"]:
                os.environ["PRIMARY_MODEL"] = config["models"]["primary"]
            if "fallback" in config["models"]:
                os.environ["FALLBACK_MODEL"] = config["models"]["fallback"]

    def _save_history(
        self,
        project_dir: Path,
        domain: str,
        input_text: str,
        result: Any
    ) -> None:
        """Save command to history.

        Args:
            project_dir: Project directory
            domain: Domain used
            input_text: Input prompt
            result: Agent result
        """
        history_dir = project_dir / ".enterprise-agent" / "history"
        if not history_dir.exists():
            return

        from datetime import datetime
        timestamp = datetime.now().isoformat()

        history_entry = {
            "timestamp": timestamp,
            "domain": domain,
            "input": input_text,
            "result": str(result)[:500]  # Truncate large results
        }

        history_file = history_dir / f"{timestamp.split('T')[0]}.jsonl"
        with history_file.open("a") as f:
            f.write(json.dumps(history_entry) + "\n")

    def analyze_project(self, project_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Analyze a project and provide insights.

        Args:
            project_dir: Project directory to analyze

        Returns:
            Analysis results
        """
        project_dir = project_dir or Path.cwd()

        analysis = {
            "project_dir": str(project_dir),
            "files": {},
            "tech_stack": [],
            "suggestions": []
        }

        # Detect technology stack
        if (project_dir / "package.json").exists():
            analysis["tech_stack"].append("Node.js/JavaScript")
        if (project_dir / "requirements.txt").exists():
            analysis["tech_stack"].append("Python")
        if (project_dir / "Cargo.toml").exists():
            analysis["tech_stack"].append("Rust")
        if (project_dir / "go.mod").exists():
            analysis["tech_stack"].append("Go")

        # Count files by type
        for pattern in ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.rs", "*.go"]:
            files = list(project_dir.rglob(pattern))
            if files:
                analysis["files"][pattern] = len(files)

        # Generate suggestions
        if "Python" in analysis["tech_stack"]:
            analysis["suggestions"].append(
                "Use domain=coding with Python-specific prompts"
            )
        if "*.tsx" in analysis["files"] or "*.jsx" in analysis["files"]:
            analysis["suggestions"].append(
                "Use domain=ui for React component generation"
            )

        return analysis

    def setup_claude_code(self) -> None:
        """Setup Claude Code CLI integration."""
        print("🔧 Setting up Claude Code CLI integration...\n")

        # Check if Claude Code is installed
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✅ Claude Code CLI installed: {result.stdout.strip()}")
            else:
                raise FileNotFoundError()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ Claude Code CLI not found")
            print("Install with: npm install -g @anthropic-ai/claude-code")
            return

        # Check login status
        try:
            result = subprocess.run(
                ["claude", "--print", "--model", "haiku", "test"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if "please run `claude login`" in result.stderr.lower():
                print("⚠️  Not logged in to Claude Code")
                print("Run: claude login")
            else:
                print("✅ Logged in to Claude Code")
        except Exception:
            print("⚠️  Could not verify login status")

        # Update global config
        if not self.global_config.exists():
            self.global_config.write_text("""# Global Enterprise Agent Configuration
use_claude_code: true
models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini
""")
            print(f"✅ Created global config: {self.global_config}")

        print("\n✅ Claude Code setup complete!")
        print("Your agent will now use Claude Code (zero API cost)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enterprise Agent CLI - Multi-domain AI agent"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize agent in project")
    init_parser.add_argument(
        "--dir",
        type=Path,
        help="Project directory (default: current)"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run agent")
    run_parser.add_argument(
        "--domain",
        default="coding",
        choices=["coding", "ui", "social", "content", "trading", "real_estate"],
        help="Domain to use"
    )
    run_parser.add_argument(
        "--input",
        required=True,
        help="Input prompt"
    )
    run_parser.add_argument(
        "--project-dir",
        type=Path,
        help="Project directory"
    )
    run_parser.add_argument(
        "--config",
        type=Path,
        help="Custom config file"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Run in interactive mode"
    )
    interactive_parser.add_argument(
        "--domain",
        default="coding",
        help="Initial domain"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze project")
    analyze_parser.add_argument(
        "--dir",
        type=Path,
        help="Project directory"
    )

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup Claude Code")

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    cli = EnterpriseAgentCLI()

    if args.command == "init":
        cli.init_project(args.dir)
    elif args.command == "run":
        result = cli.run_agent(
            domain=args.domain,
            input_text=args.input,
            project_dir=args.project_dir,
            config_file=args.config
        )
        print(f"\n✅ Complete: {result}")
    elif args.command == "interactive":
        cli.run_agent(
            domain=args.domain,
            interactive=True
        )
    elif args.command == "analyze":
        analysis = cli.analyze_project(args.dir)
        print(json.dumps(analysis, indent=2))
    elif args.command == "setup":
        cli.setup_claude_code()
    elif args.command == "version":
        print("Enterprise Agent CLI v3.4.0")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()