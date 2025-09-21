"""Claude Code CLI Provider - Uses Claude Code terminal instead of API."""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess  # nosec B404
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.exceptions import ModelException, ModelTimeoutException
from src.utils.cache import get_model_cache
from src.utils.notifications import notify_authentication_issue, notify_cli_failure
from src.utils.security_audit import audit_authentication, audit_cli_usage
from src.utils.usage_monitor import can_make_claude_request, record_claude_usage

logger = logging.getLogger(__name__)


class ClaudeCodeProvider:
    """Provider that uses Claude Code CLI instead of API for Max subscription users."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Claude Code CLI provider.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.sessions: Dict[str, str] = {}  # Track sessions for context
        self.cache = get_model_cache()

        # Setup persistent session storage
        self.session_file = Path.home() / ".claude" / "enterprise_sessions.json"
        self._load_persistent_sessions()

        self.verify_cli_available()
        self.verify_subscription_auth()

    def verify_cli_available(self) -> bool:
        """Verify Claude Code CLI is installed and available."""
        try:
            result = subprocess.run(  # nosec B603, B607
                ["claude", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Claude Code CLI available: {result.stdout.strip()}")
                audit_cli_usage(
                    "version_check", True, {"version": result.stdout.strip()}
                )
                return True
            else:
                audit_cli_usage("version_check", False, {"error": "CLI not responding"})
                raise ModelException(
                    "Claude Code CLI not found. Please install with: npm install -g @anthropic-ai/claude-code",
                    provider="claude_code",
                )
        except FileNotFoundError:
            notify_authentication_issue("cli_not_found")
            raise ModelException(
                "Claude Code CLI not installed. Please run: npm install -g @anthropic-ai/claude-code",
                provider="claude_code",
            )
        except subprocess.TimeoutExpired:
            notify_cli_failure("version_check", "Command timed out")
            raise ModelTimeoutException(
                "Claude Code CLI verification timed out", provider="claude_code"
            )

    def verify_subscription_auth(self) -> bool:
        """Verify using subscription authentication, not API key."""
        # Check if API key is set (we don't want this)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key and api_key != "STUBBED_FALLBACK":
            notify_authentication_issue(
                "api_key_conflict",
                {"api_key_present": True, "subscription_mode": False},
            )
            logger.warning(
                "ANTHROPIC_API_KEY found! Claude Code will use API instead of subscription. "
                "Run 'unset ANTHROPIC_API_KEY' to use your Max subscription."
            )
            # Optionally remove it programmatically
            if self.config.get("auto_remove_api_key", False):
                del os.environ["ANTHROPIC_API_KEY"]
                logger.info("Removed ANTHROPIC_API_KEY to force subscription usage")

        # Check if logged in with subscription
        try:
            # Try a minimal command to verify auth
            result = subprocess.run(  # nosec B603, B607
                ["claude", "--print", "--model", "haiku", "test"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if "please run `claude login`" in result.stderr.lower():
                notify_authentication_issue("not_logged_in")
                audit_authentication("login_check", False, {"reason": "not_logged_in"})
                logger.warning("Not logged in to Claude Code. Please run: claude login")
                return False

            logger.info("Claude Code subscription authentication verified")
            audit_authentication("login_check", True, {"subscription_mode": True})
            return True

        except Exception as e:
            notify_cli_failure("auth_verification", str(e))
            logger.warning(f"Could not verify subscription auth: {e}")
            return False

    def _map_model_to_cli(self, model: str) -> str:
        """Map internal model names to Claude Code CLI model names.

        Args:
            model: Internal model name (e.g., "claude_sonnet_4")

        Returns:
            CLI model name (e.g., "sonnet")
        """
        model_mapping = {
            "claude_sonnet_4": "sonnet",
            "claude-3-5-sonnet": "sonnet",
            "claude-3-5-sonnet-20241022": "sonnet",
            "claude_opus_4": "opus",
            "claude-3-opus": "opus",
            "claude_haiku": "haiku",
            "claude-3-haiku": "haiku",
        }

        # Check exact match first
        if model in model_mapping:
            return model_mapping[model]

        # Check if model contains known names
        model_lower = model.lower()
        if "sonnet" in model_lower:
            return "sonnet"
        elif "opus" in model_lower:
            return "opus"
        elif "haiku" in model_lower:
            return "haiku"

        # Default to sonnet
        logger.warning(f"Unknown model '{model}', defaulting to 'sonnet'")
        return "sonnet"

    def _load_persistent_sessions(self) -> None:
        """Load persistent sessions from disk."""
        try:
            if self.session_file.exists():
                with open(self.session_file, "r") as f:
                    session_data = json.load(f)

                # Filter out expired sessions
                current_time = time.time()
                retention_hours = self.config.get("session_retention_hours", 5)
                retention_seconds = retention_hours * 3600

                for session_id, session_info in session_data.items():
                    if isinstance(session_info, dict):
                        timestamp = session_info.get("timestamp", 0)
                        if current_time - timestamp < retention_seconds:
                            self.sessions[session_id] = session_info.get(
                                "cli_session_id", session_id
                            )
                    else:
                        # Legacy format
                        self.sessions[session_id] = session_info

                logger.debug(f"Loaded {len(self.sessions)} persistent sessions")

        except Exception as e:
            logger.warning(f"Failed to load persistent sessions: {e}")
            self.sessions = {}

    def _save_persistent_sessions(self) -> None:
        """Save current sessions to disk."""
        try:
            # Ensure directory exists
            self.session_file.parent.mkdir(parents=True, exist_ok=True)

            # Prepare session data with metadata
            current_time = time.time()
            session_data = {}

            for session_id, cli_session_id in self.sessions.items():
                session_data[session_id] = {
                    "cli_session_id": cli_session_id,
                    "timestamp": current_time,
                    "created": current_time,
                }

            with open(self.session_file, "w") as f:
                json.dump(session_data, f, indent=2)

            logger.debug(f"Saved {len(session_data)} sessions to disk")

        except Exception as e:
            logger.warning(f"Failed to save persistent sessions: {e}")

    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        try:
            current_time = time.time()
            retention_hours = self.config.get("session_retention_hours", 5)
            retention_seconds = retention_hours * 3600

            # Load session metadata
            if self.session_file.exists():
                with open(self.session_file, "r") as f:
                    session_data = json.load(f)

                expired_sessions = []
                for session_id, session_info in session_data.items():
                    if isinstance(session_info, dict):
                        timestamp = session_info.get("timestamp", 0)
                        if current_time - timestamp >= retention_seconds:
                            expired_sessions.append(session_id)

                # Remove expired sessions
                for session_id in expired_sessions:
                    if session_id in self.sessions:
                        del self.sessions[session_id]
                    del session_data[session_id]

                # Save cleaned data
                if expired_sessions:
                    with open(self.session_file, "w") as f:
                        json.dump(session_data, f, indent=2)

                    logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")

        except Exception as e:
            logger.warning(f"Failed to cleanup expired sessions: {e}")

    def _enhance_prompt_with_context(
        self,
        prompt: str,
        role: Optional[str] = None,
        operation: Optional[str] = None,
        project_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Enhance prompt with context for better Claude reasoning.

        Args:
            prompt: Original prompt
            role: Role context
            operation: Operation context
            project_context: Project-specific context

        Returns:
            Enhanced prompt optimized for Claude
        """
        enhanced_parts = []

        # Add project-aware template adaptation
        project_template = self._detect_and_adapt_project_template(
            project_context, role, operation
        )
        if project_template:
            enhanced_parts.append(project_template)

        # Add role-specific context
        if role and operation:
            role_context = self._get_role_context(role, operation)
            if role_context:
                enhanced_parts.append(
                    f"<role_context>\n{role_context}\n</role_context>"
                )

        # Add project context
        if project_context:
            context_str = self._format_project_context(project_context)
            if context_str:
                enhanced_parts.append(
                    f"<project_context>\n{context_str}\n</project_context>"
                )

        # Add reasoning enhancement for Claude
        if role in ["Planner", "Reviewer", "Validator"]:
            enhanced_parts.append(
                "<thinking_approach>\n"
                "Think step by step and reason through your approach before providing the response.\n"
                "Consider multiple perspectives and potential edge cases.\n"
                "</thinking_approach>"
            )

        # Add the original prompt
        enhanced_parts.append(f"<task>\n{prompt}\n</task>")

        return "\n\n".join(enhanced_parts)

    def _get_role_context(self, role: str, operation: str) -> str:
        """Get role-specific context to enhance Claude's understanding."""
        role_contexts = {
            "Planner": {
                "decompose": "You are an expert project planner. Break down complex tasks into clear, actionable steps. Consider dependencies, risks, and resource requirements.",
                "analyze": "Analyze the given information systematically. Identify key patterns, potential issues, and opportunities.",
            },
            "Coder": {
                "implement": "You are a senior software engineer. Write clean, maintainable, and well-documented code. Follow best practices and consider security implications.",
                "review": "Review code for correctness, performance, security, and maintainability. Provide specific, actionable feedback.",
            },
            "Reviewer": {
                "review": "You are a code review expert. Evaluate code quality, security, performance, and adherence to best practices. Be thorough but constructive.",
                "validate": "Validate the implementation against requirements. Check for completeness and correctness.",
            },
            "Validator": {
                "validate": "You are a quality assurance expert. Systematically verify that outputs meet requirements and quality standards.",
                "test": "Design comprehensive tests that cover edge cases and validate functionality.",
            },
            "Reflector": {
                "reflect": "You are a process improvement expert. Analyze what worked well and what could be improved. Provide actionable insights.",
                "summarize": "Synthesize key insights and learnings from the process.",
            },
        }

        return role_contexts.get(role, {}).get(operation, "")

    def _format_project_context(self, project_context: Dict[str, Any]) -> str:
        """Format project context for Claude understanding."""
        context_parts = []

        if "tech_stack" in project_context:
            tech_stack = project_context["tech_stack"]
            if isinstance(tech_stack, list):
                context_parts.append(f"Technology Stack: {', '.join(tech_stack)}")
            else:
                context_parts.append(f"Technology Stack: {tech_stack}")

        if "domain" in project_context:
            context_parts.append(f"Domain: {project_context['domain']}")

        if "conventions" in project_context:
            conventions = project_context["conventions"]
            if isinstance(conventions, list):
                context_parts.append(f"Coding Conventions: {'; '.join(conventions)}")
            else:
                context_parts.append(f"Coding Conventions: {conventions}")

        if "requirements" in project_context:
            requirements = project_context["requirements"]
            if isinstance(requirements, list):
                context_parts.append(f"Requirements: {'; '.join(requirements)}")
            else:
                context_parts.append(f"Requirements: {requirements}")

        if "architecture" in project_context:
            context_parts.append(f"Architecture: {project_context['architecture']}")

        return "\n".join(context_parts)

    def _detect_and_adapt_project_template(
        self,
        project_context: Optional[Dict[str, Any]],
        role: Optional[str],
        operation: Optional[str],
    ) -> Optional[str]:
        """Detect project type and adapt templates for Claude's natural strengths."""
        if not project_context:
            return None

        # Detect project characteristics
        tech_stack = project_context.get("tech_stack", [])
        architecture = project_context.get("architecture", "")

        # Enterprise Agent specific adaptations
        if (
            "Enterprise Agent" in str(tech_stack)
            or "multi-agent" in architecture.lower()
        ):
            return self._get_enterprise_agent_template(role, operation)

        # Python project adaptations
        if "Python" in str(tech_stack):
            return self._get_python_project_template(role, operation)

        # Web development adaptations
        if any(
            tech in str(tech_stack).lower()
            for tech in ["react", "vue", "angular", "javascript", "typescript"]
        ):
            return self._get_web_project_template(role, operation)

        # API/Backend service adaptations
        if any(
            tech in str(tech_stack).lower()
            for tech in ["fastapi", "flask", "django", "express"]
        ):
            return self._get_api_project_template(role, operation)

        return None

    def _get_enterprise_agent_template(
        self, role: Optional[str], operation: Optional[str]
    ) -> str:
        """Get template optimized for Enterprise Agent projects."""
        templates = {
            "Planner": {
                "decompose": "You are architecting an enterprise multi-agent system. Focus on role coordination, state management, and validation loops. Consider scalability and maintainability.",
                "analyze": "Analyze the enterprise agent architecture, considering role interactions, data flow, and performance implications.",
            },
            "Coder": {
                "implement": "You are implementing enterprise-grade agent code. Follow the established patterns: role-based design, comprehensive error handling, and structured logging. Ensure type safety and testability.",
                "review": "Review enterprise agent code for adherence to architectural patterns, security considerations, and integration points.",
            },
            "Validator": {
                "validate": "Validate enterprise agent implementation against architectural principles: role separation, state consistency, error resilience, and performance requirements."
            },
            "Reviewer": {
                "review": "Review enterprise agent quality focusing on: architectural compliance, role coordination effectiveness, error handling robustness, and maintainability."
            },
        }

        if role and operation:
            template = templates.get(role, {}).get(operation)
            if template:
                return f"<project_template>\n{template}\n</project_template>"

        return "<project_template>\nYou are working on an enterprise multi-agent system. Focus on role-based architecture, validation loops, and quality assurance.\n</project_template>"

    def _get_python_project_template(
        self, role: Optional[str], operation: Optional[str]
    ) -> str:
        """Get template optimized for Python projects."""
        base_template = "You are working on a Python project. Follow PEP standards, use type hints, implement comprehensive error handling, and ensure code is testable and maintainable."

        if role == "Coder":
            return f"<project_template>\n{base_template} Focus on Pythonic patterns, proper imports, and clear documentation.\n</project_template>"
        elif role == "Reviewer":
            return f"<project_template>\n{base_template} Review for Python best practices, security vulnerabilities, and performance considerations.\n</project_template>"

        return f"<project_template>\n{base_template}\n</project_template>"

    def _get_web_project_template(
        self, role: Optional[str], operation: Optional[str]
    ) -> str:
        """Get template optimized for web development projects."""
        base_template = "You are working on a web development project. Focus on responsive design, accessibility, performance optimization, and security best practices."

        if role == "Coder":
            return f"<project_template>\n{base_template} Ensure cross-browser compatibility and follow modern web standards.\n</project_template>"
        elif role == "Reviewer":
            return f"<project_template>\n{base_template} Review for security vulnerabilities, performance bottlenecks, and accessibility compliance.\n</project_template>"

        return f"<project_template>\n{base_template}\n</project_template>"

    def _get_api_project_template(
        self, role: Optional[str], operation: Optional[str]
    ) -> str:
        """Get template optimized for API/backend projects."""
        base_template = "You are working on an API/backend service. Focus on RESTful design, proper error handling, authentication/authorization, and API documentation."

        if role == "Coder":
            return f"<project_template>\n{base_template} Implement proper validation, logging, and monitoring capabilities.\n</project_template>"
        elif role == "Reviewer":
            return f"<project_template>\n{base_template} Review for security vulnerabilities, performance scalability, and API design consistency.\n</project_template>"

        return f"<project_template>\n{base_template}\n</project_template>"

    def call_model(
        self,
        prompt: str,
        model: str = "sonnet",
        role: Optional[str] = None,
        operation: Optional[str] = None,
        session_id: Optional[str] = None,
        use_cache: bool = True,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        project_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Call Claude Code CLI with the given prompt.

        Args:
            prompt: The prompt to send
            model: Model name (will be mapped to CLI name)
            role: Role context (for logging)
            operation: Operation context (for logging)
            session_id: Optional session ID for context retention
            use_cache: Whether to use cache
            temperature: Temperature setting (not used by CLI directly)
            max_tokens: Max tokens (not directly controllable in CLI)
            project_context: Optional project context for enhanced reasoning
            **kwargs: Additional arguments

        Returns:
            Model response as string
        """
        # Enhance prompt with context for better Claude reasoning
        enhanced_prompt = self._enhance_prompt_with_context(
            prompt, role, operation, project_context
        )

        # Initialize cache_key for potential use in exception handling
        cache_key = None

        # Check cache first (use enhanced prompt for cache key)
        if use_cache:
            cache_key = f"{model}:{role}:{enhanced_prompt[:100]}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for {role}/{operation}")
                return cached

        # Map model name
        cli_model = self._map_model_to_cli(model)

        # Build command
        cmd = [
            "claude",
            "--print",  # Non-interactive mode
            "--model",
            cli_model,
            "--output-format",
            "json",  # Get structured output
        ]

        # Add session management if provided
        if session_id and session_id in self.sessions:
            # Sanitize session ID to prevent injection
            safe_session_id = shlex.quote(self.sessions[session_id])
            cmd.extend(["--resume", safe_session_id])

        # Add fallback model for reliability
        if self.config.get("enable_fallback", True):
            cmd.extend(["--fallback-model", "haiku"])

        # Add allowed tools for automation
        if self.config.get("auto_mode", False):
            cmd.append("--dangerously-skip-permissions")

        # Add the enhanced prompt (already safe as it's passed as argument, not shell interpreted)
        # But we'll ensure it doesn't contain shell metacharacters if it ever gets used in shell mode
        cmd.append(enhanced_prompt)

        try:
            # Check usage limits before making request
            if not can_make_claude_request():
                raise ModelException(
                    "Usage limit reached for Claude Code. Please wait for usage window to reset.",
                    provider="claude_code",
                    model=cli_model,
                )

            logger.debug(f"Executing Claude Code CLI for {role}/{operation}")

            # Execute command with enhanced error handling
            try:
                # Ensure working directory is safe
                work_dir = self.config.get("working_directory", os.getcwd())
                if not os.path.isdir(work_dir):
                    work_dir = os.getcwd()

                result = subprocess.run(  # nosec B603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.get("timeout", 60),
                    cwd=work_dir,
                    shell=False,  # Explicitly disable shell to prevent injection
                )
            except subprocess.TimeoutExpired as e:
                logger.error(f"Claude Code CLI timeout after {e.timeout}s")
                notify_cli_failure(f"{role}/{operation}", f"Timeout after {e.timeout}s")
                raise ModelTimeoutException(
                    f"Claude Code CLI timeout after {e.timeout}s",
                    provider="claude_code",
                    model=cli_model,
                ) from e
            except FileNotFoundError:
                logger.error("Claude Code CLI not found in PATH")
                notify_cli_failure(
                    f"{role}/{operation}", "Claude Code CLI not installed"
                )
                raise ModelException(
                    "Claude Code CLI not found. Please install it first.",
                    provider="claude_code",
                    model=cli_model,
                )
            except Exception as e:
                logger.error(f"Unexpected error executing Claude Code CLI: {e}")
                notify_cli_failure(f"{role}/{operation}", f"Unexpected error: {e}")
                raise ModelException(
                    f"Unexpected error executing Claude Code CLI: {e}",
                    provider="claude_code",
                    model=cli_model,
                ) from e

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"

                # Categorize error for better user guidance
                if "please run `claude login`" in error_msg.lower():
                    notify_authentication_issue("not_logged_in")
                elif (
                    "network" in error_msg.lower() or "connection" in error_msg.lower()
                ):
                    notify_cli_failure(
                        f"{role}/{operation}", f"Network error: {error_msg}"
                    )
                elif (
                    "rate limit" in error_msg.lower()
                    or "usage limit" in error_msg.lower()
                ):
                    notify_cli_failure(
                        f"{role}/{operation}", f"Usage limit reached: {error_msg}"
                    )
                else:
                    notify_cli_failure(f"{role}/{operation}", error_msg)

                raise ModelException(
                    f"Claude Code CLI failed: {error_msg}",
                    provider="claude_code",
                    model=cli_model,
                )

            # Parse response with error handling
            try:
                response = self._parse_cli_response(result.stdout)
            except Exception as e:
                logger.error(f"Failed to parse Claude Code response: {e}")
                # Fallback to raw output if parsing fails
                response = {"text": result.stdout, "response": result.stdout}

            # Store session for context retention
            if session_id and "session_id" in response:
                self.sessions[session_id] = response["session_id"]

            # Extract actual response text with validation
            output = response.get("response", response.get("text", result.stdout))
            if not output or not isinstance(output, str):
                logger.warning("Empty or invalid response from Claude Code CLI")
                output = result.stdout or "No response received"

            # Record successful usage
            record_claude_usage(
                role or "unknown", operation or "call", len(output) // 4
            )

            # Audit successful CLI usage
            audit_cli_usage(
                f"{role}/{operation}",
                True,
                {
                    "model": cli_model,
                    "tokens_estimated": len(output) // 4,
                    "session_id": session_id,
                },
            )

            # Cache the response
            if use_cache and output:
                cache_ttl = 1800 if role == "Planner" else 900  # 30min/15min
                self.cache.set(cache_key, output, ttl=cache_ttl)

            # Store for potential quality feedback
            self._last_response_info = {
                "cache_key": cache_key,
                "role": role,
                "operation": operation,
                "session_id": session_id,
            }

            return output

        except (ModelException, ModelTimeoutException):
            # Re-raise model exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error in call_model: {e}")
            notify_cli_failure(f"{role}/{operation}", f"Unexpected error: {str(e)}")
            raise ModelException(
                f"Claude Code CLI error: {str(e)}",
                provider="claude_code",
                model=cli_model,
                cause=e,
            )

    def _parse_cli_response(self, output: str) -> Dict[str, Any]:
        """Parse Claude Code CLI output.

        Args:
            output: Raw CLI output

        Returns:
            Parsed response dictionary
        """
        if not output:
            return {"text": ""}

        # Try to parse as JSON first
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            # If not JSON, return as text
            return {"text": output}

    def stream_response(
        self,
        prompt: str,
        model: str = "sonnet",
        callback: Optional[callable] = None,
        **kwargs,
    ) -> str:
        """Stream responses from Claude Code CLI.

        Args:
            prompt: The prompt to send
            model: Model name
            callback: Optional callback for streaming chunks
            **kwargs: Additional arguments

        Returns:
            Complete response
        """
        cli_model = self._map_model_to_cli(model)

        cmd = [
            "claude",
            "--print",
            "--model",
            cli_model,
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            prompt,
        ]

        try:
            process = subprocess.Popen(  # nosec B603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            full_response = []

            for line in process.stdout:
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        if "text" in chunk:
                            full_response.append(chunk["text"])
                            if callback:
                                callback(chunk["text"])
                    except json.JSONDecodeError:
                        continue

            process.wait()

            if process.returncode != 0:
                error = process.stderr.read()
                raise ModelException(
                    f"Streaming failed: {error}", provider="claude_code"
                )

            return "".join(full_response)

        except Exception as e:
            raise ModelException(
                f"Stream error: {str(e)}", provider="claude_code", cause=e
            )

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session for context retention.

        Args:
            session_id: Optional session ID, generates one if not provided

        Returns:
            Session ID
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # Clean up expired sessions first
        self._cleanup_expired_sessions()

        # Create new session
        self.sessions[session_id] = session_id

        # Save to persistent storage
        self._save_persistent_sessions()

        logger.debug(f"Created new session: {session_id}")
        return session_id

    def clear_session(self, session_id: str) -> None:
        """Clear a session.

        Args:
            session_id: Session ID to clear
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

            # Update persistent storage
            self._save_persistent_sessions()

            logger.debug(f"Cleared session: {session_id}")

    def provide_quality_feedback(
        self, score: float, feedback: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Provide quality feedback for the last response to improve future outputs.

        Args:
            score: Quality score between 0.0 and 1.0
            feedback: Optional detailed feedback dictionary

        Returns:
            True if feedback was successfully recorded
        """
        if not hasattr(self, "_last_response_info") or not self._last_response_info:
            logger.warning("No recent response to provide feedback for")
            return False

        cache_key = self._last_response_info.get("cache_key")
        if cache_key:
            self.cache.update_quality(cache_key, score, feedback)
            logger.debug(
                f"Updated quality score {score} for {self._last_response_info.get('role')}/{self._last_response_info.get('operation')}"
            )
            return True

        return False

    def get_quality_insights(self) -> Dict[str, Any]:
        """Get quality insights from cached responses for learning.

        Returns:
            Quality insights dictionary
        """
        insights = self.cache.get_quality_insights()
        low_quality_patterns = self.cache.get_low_quality_patterns()

        return {
            **insights,
            "low_quality_patterns": low_quality_patterns,
            "recommendations": self._generate_quality_recommendations(
                insights, low_quality_patterns
            ),
        }

    def _generate_quality_recommendations(
        self, insights: Dict[str, Any], patterns: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on quality patterns."""
        recommendations = []

        # Analyze quality metrics
        avg_quality = insights.get("average_quality", 0.0)
        if avg_quality < 0.7:
            recommendations.append("Consider using more specific prompts and examples")

        if insights.get("low_quality_items", 0) > 5:
            recommendations.append(
                "Review prompt templates for common failure patterns"
            )

        # Analyze patterns
        if patterns:
            common_issues = {}
            for pattern in patterns:
                feedback = pattern.get("feedback", {})
                if isinstance(feedback, dict):
                    for issue_type in feedback.keys():
                        common_issues[issue_type] = common_issues.get(issue_type, 0) + 1

            if common_issues:
                top_issue = max(common_issues.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Focus on improving {top_issue[0]} (appears in {top_issue[1]} low-quality responses)"
                )

        return recommendations

    def get_usage_info(self) -> Dict[str, Any]:
        """Get usage information (placeholder for Max plan tracking).

        Returns:
            Usage information dictionary
        """
        # Claude Code doesn't provide direct usage APIs
        # This would need to track manually or parse from Claude web
        return {
            "plan": "Max 20x",
            "estimated_prompts_remaining": "200-800 per 5 hours",
            "note": "Usage tracked through Claude web interface",
        }


# Singleton instance for reuse
_provider_instance: Optional[ClaudeCodeProvider] = None


def get_claude_code_provider(
    config: Optional[Dict[str, Any]] = None
) -> ClaudeCodeProvider:
    """Get or create Claude Code provider instance.

    Args:
        config: Optional configuration

    Returns:
        ClaudeCodeProvider instance
    """
    global _provider_instance

    if _provider_instance is None:
        _provider_instance = ClaudeCodeProvider(config)

    return _provider_instance


def reset_claude_code_provider() -> None:
    """Reset the global Claude Code provider instance for testing."""
    global _provider_instance
    _provider_instance = None


__all__ = [
    "ClaudeCodeProvider",
    "get_claude_code_provider",
    "reset_claude_code_provider",
]
