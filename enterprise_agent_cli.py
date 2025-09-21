#!/usr/bin/env python
"""Enterprise Agent CLI - Use agent across multiple projects with enhanced configuration."""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent_orchestrator import AgentOrchestrator
from src.utils.errors import EnterpriseAgentError, ErrorCode, handle_error


class EnterpriseAgentCLI:
    """CLI interface for Enterprise Agent with enhanced configuration and domain support."""

    # Domain configurations with descriptions and capabilities
    DOMAIN_CONFIGS = {
        "coding": {
            "description": "Software development, debugging, code review",
            "capabilities": ["code_generation", "debugging", "refactoring", "testing"],
            "default_models": {"primary": "claude-3-5-sonnet-20241022", "fallback": "gpt-4o-mini"},
            "reflection_iterations": 5,
            "confidence_threshold": 0.8
        },
        "ui": {
            "description": "User interface design and development",
            "capabilities": ["component_design", "responsive_design", "accessibility"],
            "default_models": {"primary": "claude-3-5-sonnet-20241022", "fallback": "gpt-4-vision-preview"},
            "reflection_iterations": 3,
            "confidence_threshold": 0.75
        },
        "social": {
            "description": "Social media content and strategy",
            "capabilities": ["content_creation", "strategy", "analytics"],
            "default_models": {"primary": "claude-3-opus-20240229", "fallback": "gpt-4"},
            "reflection_iterations": 2,
            "confidence_threshold": 0.7
        },
        "content": {
            "description": "Content writing and documentation",
            "capabilities": ["writing", "editing", "documentation", "research"],
            "default_models": {"primary": "claude-3-5-sonnet-20241022", "fallback": "gpt-4"},
            "reflection_iterations": 3,
            "confidence_threshold": 0.8
        },
        "trading": {
            "description": "Financial analysis and trading strategies",
            "capabilities": ["market_analysis", "risk_assessment", "strategy_development"],
            "default_models": {"primary": "claude-3-opus-20240229", "fallback": "gpt-4"},
            "reflection_iterations": 5,
            "confidence_threshold": 0.9,
            "security_enhanced": True
        },
        "real_estate": {
            "description": "Real estate analysis and investment",
            "capabilities": ["property_analysis", "market_research", "investment_planning"],
            "default_models": {"primary": "claude-3-5-sonnet-20241022", "fallback": "gpt-4"},
            "reflection_iterations": 4,
            "confidence_threshold": 0.8
        },
        "research": {
            "description": "Research and data analysis",
            "capabilities": ["data_analysis", "literature_review", "hypothesis_testing"],
            "default_models": {"primary": "claude-3-opus-20240229", "fallback": "gpt-4"},
            "reflection_iterations": 4,
            "confidence_threshold": 0.85
        },
        "security": {
            "description": "Security analysis and penetration testing",
            "capabilities": ["vulnerability_assessment", "security_audit", "threat_modeling"],
            "default_models": {"primary": "claude-3-opus-20240229", "fallback": "gpt-4"},
            "reflection_iterations": 6,
            "confidence_threshold": 0.95,
            "security_enhanced": True
        }
    }

    def __init__(self, verbose: bool = False, log_level: str = "INFO"):
        """Initialize CLI with enhanced configuration.

        Args:
            verbose: Enable verbose output
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.verbose = verbose
        self.home_dir = Path.home() / ".enterprise-agent"
        self.global_config = self.home_dir / "config.yml"
        self.ensure_directories()

        # Configure logging
        self._setup_logging(log_level)

        # Track session metrics
        self.session_start = time.time()
        self.commands_executed = 0
        self.errors_encountered = 0

    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration."""
        log_dir = self.home_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"enterprise_agent_{time.strftime('%Y%m%d')}.log"

        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout) if self.verbose else logging.NullHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def ensure_directories(self):
        """Ensure required directories exist."""
        self.home_dir.mkdir(exist_ok=True)
        (self.home_dir / "cache").mkdir(exist_ok=True)
        (self.home_dir / "logs").mkdir(exist_ok=True)
        (self.home_dir / "templates").mkdir(exist_ok=True)
        (self.home_dir / "domains").mkdir(exist_ok=True)

    def list_domains(self) -> None:
        """List available domains with descriptions."""
        print("\n🎯 Available Domains:")
        print("=" * 50)

        for domain_name, config in self.DOMAIN_CONFIGS.items():
            print(f"\n{domain_name:12} - {config['description']}")
            print(f"{'':12}   Capabilities: {', '.join(config['capabilities'])}")
            print(f"{'':12}   Primary Model: {config['default_models']['primary']}")
            print(f"{'':12}   Reflection: {config['reflection_iterations']} iterations")

            if config.get('security_enhanced'):
                print(f"{'':12}   🔒 Security Enhanced Domain")

    def get_domain_config(self, domain: str) -> Dict[str, Any]:
        """Get configuration for a specific domain.

        Args:
            domain: Domain name

        Returns:
            Domain configuration

        Raises:
            ValueError: If domain is not supported
        """
        if domain not in self.DOMAIN_CONFIGS:
            available = ', '.join(self.DOMAIN_CONFIGS.keys())
            raise ValueError(f"Unsupported domain '{domain}'. Available: {available}")

        return self.DOMAIN_CONFIGS[domain].copy()

    def validate_domain(self, domain: str) -> bool:
        """Validate that a domain is supported.

        Args:
            domain: Domain name to validate

        Returns:
            True if domain is valid
        """
        return domain in self.DOMAIN_CONFIGS

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
        (agent_dir / "config.yml").write_text(
            """# Enterprise Agent Project Configuration
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
"""
        )

        (agent_dir / "templates").mkdir()
        (agent_dir / "context").mkdir()
        (agent_dir / "history").mkdir()
        (agent_dir / ".gitignore").write_text(
            """cache/
history/
*.log
"""
        )

        # Create example template
        (agent_dir / "templates" / "code_review.md").write_text(
            """# Code Review Template
Please review the following code for:
- Security vulnerabilities
- Performance issues
- Best practices
- Test coverage
"""
        )

        print(f"✅ Initialized Enterprise Agent in {project_dir}")
        print("   Created .enterprise-agent/ directory")
        print("   Edit .enterprise-agent/config.yml to customize")

    def run_agent(
        self,
        domain: str = "coding",
        input_text: str = "",
        project_dir: Optional[Path] = None,
        config_file: Optional[Path] = None,
        interactive: bool = False,
        dry_run: bool = False,
        profile: bool = False,
    ) -> Any:
        """Run the agent with specified parameters and enhanced error handling.

        Args:
            domain: Domain to use
            input_text: Input prompt
            project_dir: Project directory
            config_file: Custom config file
            interactive: Interactive mode
            dry_run: Preview actions without execution
            profile: Enable performance profiling

        Returns:
            Agent result

        Raises:
            EnterpriseAgentError: On execution errors
        """
        execution_start = time.time()
        project_dir = project_dir or Path.cwd()

        try:
            # Validate domain
            if not self.validate_domain(domain):
                available_domains = ', '.join(self.DOMAIN_CONFIGS.keys())
                raise EnterpriseAgentError(
                    ErrorCode.INVALID_DOMAIN,
                    f"Invalid domain '{domain}'. Available domains: {available_domains}",
                    context={"requested_domain": domain, "available_domains": list(self.DOMAIN_CONFIGS.keys())},
                    recovery_suggestions=[
                        "Use 'enterprise-agent domains' to list available domains",
                        f"Try one of: {available_domains}",
                        "Check domain spelling"
                    ]
                )

            # Get domain configuration
            domain_config = self.get_domain_config(domain)

            # Set working directory for agent
            original_dir = os.getcwd()
            os.chdir(project_dir)

            # Configure environment based on domain
            self._configure_domain_environment(domain, domain_config)

            if dry_run:
                return self._dry_run_preview(domain, input_text, domain_config)

            # Load configurations
            self._load_configurations(project_dir, config_file)

            # Initialize orchestrator with error handling
            try:
                orchestrator = AgentOrchestrator()
                self.logger.info(f"Initialized orchestrator for domain: {domain}")
            except Exception as e:
                raise EnterpriseAgentError(
                    ErrorCode.ORCHESTRATION_INIT_FAILED,
                    f"Failed to initialize agent orchestrator: {str(e)}",
                    context={"domain": domain, "project_dir": str(project_dir)},
                    recovery_suggestions=[
                        "Check configuration files",
                        "Verify dependencies are installed",
                        "Check API key configuration"
                    ],
                    cause=e
                )

            if interactive:
                return self._run_interactive(orchestrator, domain, domain_config)
            else:
                return self._run_single_command(
                    orchestrator, domain, input_text, project_dir,
                    domain_config, profile
                )

        except EnterpriseAgentError:
            self.errors_encountered += 1
            raise
        except Exception as e:
            self.errors_encountered += 1
            error_details = handle_error(e, {"domain": domain, "project_dir": str(project_dir)})
            raise EnterpriseAgentError(
                ErrorCode.SYSTEM_ERROR,
                f"Unexpected error during agent execution: {str(e)}",
                context={"domain": domain, "project_dir": str(project_dir)},
                cause=e
            )
        finally:
            os.chdir(original_dir)
            execution_time = time.time() - execution_start
            self.commands_executed += 1
            self.logger.info(f"Command execution completed in {execution_time:.2f}s")

    def _configure_domain_environment(self, domain: str, domain_config: Dict[str, Any]) -> None:
        """Configure environment variables based on domain configuration."""
        # Set reflection parameters based on domain
        os.environ["REFLECTION_MAX_ITERATIONS"] = str(domain_config["reflection_iterations"])
        os.environ["REFLECTION_CONFIDENCE_THRESHOLD"] = str(domain_config["confidence_threshold"])

        # Enable enhanced security for sensitive domains
        if domain_config.get("security_enhanced"):
            os.environ["SECURITY_ENHANCED"] = "true"
            self.logger.info(f"Enabled enhanced security for domain: {domain}")

    def _load_configurations(self, project_dir: Path, config_file: Optional[Path]) -> None:
        """Load project and custom configurations."""
        # Load project-specific config if exists
        project_config = project_dir / ".enterprise-agent" / "config.yml"
        if project_config.exists():
            self._load_project_config(project_config)
            self.logger.debug(f"Loaded project config: {project_config}")

        # Override with custom config if provided
        if config_file and config_file.exists():
            self._load_project_config(config_file)
            self.logger.debug(f"Loaded custom config: {config_file}")

    def _dry_run_preview(self, domain: str, input_text: str, domain_config: Dict[str, Any]) -> Dict[str, Any]:
        """Preview what would be executed without running the agent."""
        return {
            "dry_run": True,
            "domain": domain,
            "domain_config": domain_config,
            "input": input_text,
            "would_execute": {
                "reflection_iterations": domain_config["reflection_iterations"],
                "confidence_threshold": domain_config["confidence_threshold"],
                "primary_model": domain_config["default_models"]["primary"],
                "fallback_model": domain_config["default_models"]["fallback"],
                "security_enhanced": domain_config.get("security_enhanced", False)
            }
        }

    def _run_single_command(
        self,
        orchestrator: AgentOrchestrator,
        domain: str,
        input_text: str,
        project_dir: Path,
        domain_config: Dict[str, Any],
        profile: bool
    ) -> Any:
        """Run a single command with the agent."""
        print("🚀 Running Enterprise Agent")
        print(f"   Domain: {domain} ({domain_config['description']})")
        print(f"   Project: {project_dir}")
        print(f"   Input: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")

        if profile:
            import cProfile
            import pstats
            import io

            pr = cProfile.Profile()
            pr.enable()

        try:
            result = orchestrator.run_mode(domain, input_text)

            if profile:
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats()
                result["profile_data"] = s.getvalue()

            # Save to history
            self._save_history(project_dir, domain, input_text, result)

            # Add execution metadata
            result["execution_metadata"] = {
                "domain": domain,
                "domain_config": domain_config,
                "timestamp": time.time(),
                "project_dir": str(project_dir)
            }

            return result

        except Exception as e:
            if profile:
                pr.disable()

            raise EnterpriseAgentError(
                ErrorCode.ORCHESTRATION_PIPELINE_FAILED,
                f"Agent execution failed: {str(e)}",
                context={
                    "domain": domain,
                    "input_text": input_text[:200],
                    "project_dir": str(project_dir)
                },
                recovery_suggestions=[
                    "Check input format",
                    "Verify domain configuration",
                    "Review error logs"
                ],
                cause=e
            )

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
                        domain=domain, initial_state={"input": user_input}
                    )
                    print(f"\n{result}\n")

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")

    def _show_help(self) -> None:
        """Show interactive mode help."""
        print(
            """
Commands:
  /domain <name>  - Switch domain (coding, ui, social, etc.)
  /history       - Show command history
  /clear         - Clear screen
  help           - Show this help
  exit           - Exit interactive mode
"""
        )

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
        self, project_dir: Path, domain: str, input_text: str, result: Any
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
            "result": str(result)[:500],  # Truncate large results
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
            "suggestions": [],
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
                ["claude", "--version"], capture_output=True, text=True, timeout=5
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
                timeout=10,
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
            self.global_config.write_text(
                """# Global Enterprise Agent Configuration
use_claude_code: true
models:
  primary: claude-3-5-sonnet-20241022
  fallback: gpt-4o-mini
"""
            )
            print(f"✅ Created global config: {self.global_config}")

        print("\n✅ Claude Code setup complete!")
        print("Your agent will now use Claude Code (zero API cost)")

    def show_status(self) -> None:
        """Show agent status and configuration."""
        print("\n📊 Enterprise Agent Status")
        print("=" * 40)

        # Session information
        session_duration = time.time() - self.session_start
        print(f"Session Duration: {session_duration:.1f}s")
        print(f"Commands Executed: {self.commands_executed}")
        print(f"Errors Encountered: {self.errors_encountered}")

        # Configuration
        print("\nConfiguration:")
        print(f"  Home Directory: {self.home_dir}")
        print(f"  Global Config: {self.global_config}")
        print(f"  Verbose Mode: {self.verbose}")

        # Check dependencies
        print("\nDependencies:")
        try:
            import yaml
            print("  ✅ PyYAML available")
        except ImportError:
            print("  ❌ PyYAML not available")

        try:
            result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"  ✅ Claude Code CLI: {result.stdout.strip()}")
            else:
                print("  ❌ Claude Code CLI not working")
        except Exception:
            print("  ❌ Claude Code CLI not found")

        # Domain count
        print(f"\nAvailable Domains: {len(self.DOMAIN_CONFIGS)}")

    def show_history(self, limit: int = 10, domain_filter: Optional[str] = None) -> None:
        """Show command history.

        Args:
            limit: Number of entries to show
            domain_filter: Filter by domain
        """
        print(f"\n📜 Command History (last {limit} entries)")
        if domain_filter:
            print(f"    Filtered by domain: {domain_filter}")
        print("=" * 50)

        # Collect history from all projects
        history_entries = []

        # Look for .enterprise-agent directories
        for project_dir in Path.cwd().parent.rglob('.enterprise-agent'):
            history_dir = project_dir / 'history'
            if history_dir.exists():
                for history_file in history_dir.glob('*.jsonl'):
                    try:
                        with history_file.open() as f:
                            for line in f:
                                entry = json.loads(line.strip())
                                if not domain_filter or entry.get('domain') == domain_filter:
                                    entry['project'] = str(project_dir.parent.name)
                                    history_entries.append(entry)
                    except Exception:
                        continue

        # Sort by timestamp and limit
        history_entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        history_entries = history_entries[:limit]

        if not history_entries:
            print("No history entries found.")
            return

        for entry in history_entries:
            timestamp = entry.get('timestamp', 'Unknown')
            domain = entry.get('domain', 'Unknown')
            project = entry.get('project', 'Unknown')
            input_text = entry.get('input', '')

            print(f"\n[{timestamp}] {domain} @ {project}")
            print(f"  Input: {input_text[:80]}{'...' if len(input_text) > 80 else ''}")

    def show_version(self) -> None:
        """Show version information."""
        print("\n🤖 Enterprise Agent CLI")
        print("=" * 30)
        print("Version: 3.4.0")
        print("Author: Enterprise Agent Team")
        print("License: MIT")
        print("\nFeatures:")
        print("  ✅ Multi-domain support")
        print("  ✅ Claude Code integration")
        print("  ✅ Configurable reflection loops")
        print("  ✅ Structured error handling")
        print("  ✅ Enhanced caching")
        print("  ✅ Performance profiling")
        print("  ✅ Interactive mode")

    def _print_analysis_text(self, analysis: Dict[str, Any]) -> None:
        """Print analysis in human-readable text format."""
        print(f"\n📁 Project Analysis: {analysis['project_dir']}")
        print("=" * 50)

        if analysis['tech_stack']:
            print("\n🔧 Technology Stack:")
            for tech in analysis['tech_stack']:
                print(f"  • {tech}")

        if analysis['files']:
            print("\n📄 File Count:")
            for pattern, count in analysis['files'].items():
                print(f"  {pattern}: {count} files")

        if analysis['suggestions']:
            print("\n💡 Suggestions:")
            for suggestion in analysis['suggestions']:
                print(f"  • {suggestion}")


def main():
    """Main CLI entry point with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enterprise Agent CLI - Multi-domain AI agent with configurable controls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  enterprise-agent domains                              # List available domains
  enterprise-agent run --domain coding --input "Create a Python class for user management"
  enterprise-agent run --domain ui --input "Design a responsive dashboard" --dry-run
  enterprise-agent interactive --domain research       # Start interactive session
  enterprise-agent analyze --dir ./my-project          # Analyze project structure
  enterprise-agent setup                               # Setup Claude Code integration

For more information, visit: https://github.com/enterprise-agent
        """
    )

    # Global options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Set logging level")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Domains command (NEW)
    domains_parser = subparsers.add_parser("domains", help="List available domains")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize agent in project")
    init_parser.add_argument("--dir", type=Path, help="Project directory (default: current)")

    # Run command (ENHANCED)
    run_parser = subparsers.add_parser("run", help="Run agent")
    run_parser.add_argument(
        "--domain", "-d",
        default="coding",
        help="Domain to use (use 'domains' command to list available)"
    )
    run_parser.add_argument("--input", "-i", required=True, help="Input prompt")
    run_parser.add_argument("--project-dir", type=Path, help="Project directory")
    run_parser.add_argument("--config", "-c", type=Path, help="Custom config file")
    run_parser.add_argument("--dry-run", action="store_true",
                           help="Preview actions without execution")
    run_parser.add_argument("--profile", action="store_true",
                           help="Enable performance profiling")

    # Interactive command (ENHANCED)
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    interactive_parser.add_argument("--domain", "-d", default="coding",
                                   help="Initial domain")

    # Analyze command (ENHANCED)
    analyze_parser = subparsers.add_parser("analyze", help="Analyze project")
    analyze_parser.add_argument("--dir", type=Path, help="Project directory")
    analyze_parser.add_argument("--output", "-o", choices=["json", "text"], default="json",
                               help="Output format")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup Claude Code integration")

    # Status command (NEW)
    status_parser = subparsers.add_parser("status", help="Show agent status and configuration")

    # History command (NEW)
    history_parser = subparsers.add_parser("history", help="Show command history")
    history_parser.add_argument("--limit", "-n", type=int, default=10,
                               help="Number of recent entries to show")
    history_parser.add_argument("--domain", help="Filter by domain")

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    args = parser.parse_args()

    # Initialize CLI with global options
    try:
        cli = EnterpriseAgentCLI(verbose=args.verbose, log_level=args.log_level)
    except Exception as e:
        print(f"❌ Failed to initialize CLI: {e}")
        return 1

    try:
        if args.command == "domains":
            cli.list_domains()
        elif args.command == "init":
            cli.init_project(args.dir)
        elif args.command == "run":
            result = cli.run_agent(
                domain=args.domain,
                input_text=args.input,
                project_dir=args.project_dir,
                config_file=args.config,
                dry_run=args.dry_run,
                profile=args.profile
            )
            if args.dry_run:
                print("\n🔍 Dry Run Preview:")
                print(json.dumps(result, indent=2))
            else:
                print("\n✅ Complete: Success")
                if args.verbose:
                    print(json.dumps(result, indent=2, default=str))
        elif args.command == "interactive":
            cli.run_agent(domain=args.domain, interactive=True)
        elif args.command == "analyze":
            analysis = cli.analyze_project(args.dir)
            if args.output == "json":
                print(json.dumps(analysis, indent=2))
            else:
                cli._print_analysis_text(analysis)
        elif args.command == "setup":
            cli.setup_claude_code()
        elif args.command == "status":
            cli.show_status()
        elif args.command == "history":
            cli.show_history(limit=args.limit, domain_filter=args.domain)
        elif args.command == "version":
            cli.show_version()
        else:
            parser.print_help()
            return 0

        return 0

    except EnterpriseAgentError as e:
        print(f"\n❌ {e.details.code.name}: {e.details.message}")
        if e.details.recovery_suggestions:
            print("\n💡 Suggestions:")
            for suggestion in e.details.recovery_suggestions:
                print(f"   • {suggestion}")
        if args.verbose:
            print(f"\n🔍 Error Details:\n{e.details.to_json()}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
