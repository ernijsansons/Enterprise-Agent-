"""Claude Code CLI wrapper utilities."""
from __future__ import annotations

import json
import logging
import subprocess  # nosec B404
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Output format options for Claude Code CLI."""

    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


class PermissionMode(Enum):
    """Permission modes for Claude Code CLI."""

    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    BYPASS_PERMISSIONS = "bypassPermissions"
    PLAN = "plan"


@dataclass
class ClaudeCommand:
    """Represents a Claude Code CLI command."""

    prompt: str
    model: str = "sonnet"
    output_format: OutputFormat = OutputFormat.JSON
    print_mode: bool = True
    session_id: Optional[str] = None
    resume_session: Optional[str] = None
    continue_last: bool = False
    fork_session: bool = False
    allowed_tools: Optional[List[str]] = None
    disallowed_tools: Optional[List[str]] = None
    permission_mode: Optional[PermissionMode] = None
    fallback_model: Optional[str] = None
    add_directories: Optional[List[Path]] = None
    timeout: int = 60
    verbose: bool = False
    debug: Optional[str] = None

    def build_command(self) -> List[str]:
        """Build the command list for subprocess.

        Returns:
            List of command arguments
        """
        cmd = ["claude"]

        # Add prompt if not continuing/resuming
        if not self.continue_last and not self.resume_session:
            if self.print_mode:
                cmd.append("--print")

        # Model selection
        cmd.extend(["--model", self.model])

        # Output format
        if self.print_mode:
            cmd.extend(["--output-format", self.output_format.value])

        # Session management
        if self.continue_last:
            cmd.append("--continue")
        elif self.resume_session:
            cmd.extend(["--resume", self.resume_session])

        if self.fork_session:
            cmd.append("--fork-session")

        if self.session_id:
            cmd.extend(["--session-id", self.session_id])

        # Permissions
        if self.permission_mode:
            cmd.extend(["--permission-mode", self.permission_mode.value])

        # Tools
        if self.allowed_tools:
            cmd.extend(["--allowed-tools", " ".join(self.allowed_tools)])

        if self.disallowed_tools:
            cmd.extend(["--disallowed-tools", " ".join(self.disallowed_tools)])

        # Fallback model
        if self.fallback_model:
            cmd.extend(["--fallback-model", self.fallback_model])

        # Additional directories
        if self.add_directories:
            for directory in self.add_directories:
                cmd.extend(["--add-dir", str(directory)])

        # Debug/verbose
        if self.verbose:
            cmd.append("--verbose")

        if self.debug:
            cmd.extend(["--debug", self.debug])

        # Add the prompt last
        if self.prompt:
            cmd.append(self.prompt)

        return cmd


class ClaudeCodeCLI:
    """Wrapper for Claude Code CLI operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize CLI wrapper.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.default_timeout = self.config.get("timeout", 60)
        self.working_directory = self.config.get("working_directory", Path.cwd())

    def execute(
        self,
        command: ClaudeCommand,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a Claude Code command.

        Args:
            command: ClaudeCommand instance
            stream_callback: Optional callback for streaming output

        Returns:
            Dictionary with response data
        """
        cmd_list = command.build_command()

        logger.debug(f"Executing command: {' '.join(cmd_list)}")

        try:
            if command.output_format == OutputFormat.STREAM_JSON and stream_callback:
                return self._execute_streaming(
                    cmd_list, stream_callback, command.timeout
                )
            else:
                return self._execute_blocking(cmd_list, command.timeout)

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {command.timeout}s")
            return {
                "error": "timeout",
                "message": f"Command timed out after {command.timeout}s",
            }

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"error": "execution_failed", "message": str(e)}

    def _execute_blocking(self, cmd_list: List[str], timeout: int) -> Dict[str, Any]:
        """Execute command and wait for completion.

        Args:
            cmd_list: Command arguments
            timeout: Timeout in seconds

        Returns:
            Parsed response dictionary
        """
        result = subprocess.run(  # nosec B603
            cmd_list,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.working_directory,
        )

        if result.returncode != 0:
            logger.error(f"Command failed: {result.stderr}")
            return {
                "error": "command_failed",
                "message": result.stderr,
                "returncode": result.returncode,
            }

        return self._parse_output(result.stdout)

    def _execute_streaming(
        self, cmd_list: List[str], callback: Callable[[str], None], timeout: int
    ) -> Dict[str, Any]:
        """Execute command with streaming output.

        Args:
            cmd_list: Command arguments
            callback: Callback for streaming chunks
            timeout: Timeout in seconds

        Returns:
            Complete response dictionary
        """
        process = subprocess.Popen(  # nosec B603
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self.working_directory,
        )

        chunks = []
        stdout_queue = Queue()
        stderr_queue = Queue()

        # Thread to read stdout
        def read_stdout():
            for line in process.stdout:
                stdout_queue.put(line)
            stdout_queue.put(None)  # Signal completion

        # Thread to read stderr
        def read_stderr():
            for line in process.stderr:
                stderr_queue.put(line)
            stderr_queue.put(None)

        stdout_thread = threading.Thread(target=read_stdout)
        stderr_thread = threading.Thread(target=read_stderr)

        stdout_thread.start()
        stderr_thread.start()

        # Process output with timeout
        import time

        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                process.kill()
                raise subprocess.TimeoutExpired(cmd_list, timeout)

            # Check stdout
            if not stdout_queue.empty():
                line = stdout_queue.get()
                if line is None:
                    break

                # Parse streaming JSON
                try:
                    chunk = json.loads(line.strip())
                    chunks.append(chunk)

                    # Call callback with text content
                    if "text" in chunk:
                        callback(chunk["text"])

                except json.JSONDecodeError:
                    # Non-JSON output
                    chunks.append({"text": line.strip()})
                    callback(line.strip())

        stdout_thread.join()
        stderr_thread.join()

        # Wait for process completion
        process.wait()

        if process.returncode != 0:
            stderr_content = []
            while not stderr_queue.empty():
                line = stderr_queue.get()
                if line:
                    stderr_content.append(line)

            return {
                "error": "command_failed",
                "message": "".join(stderr_content),
                "returncode": process.returncode,
            }

        # Combine chunks into final response
        full_text = "".join(chunk.get("text", "") for chunk in chunks)

        return {"response": full_text, "chunks": chunks, "streaming": True}

    def _parse_output(self, output: str) -> Dict[str, Any]:
        """Parse CLI output.

        Args:
            output: Raw CLI output

        Returns:
            Parsed response dictionary
        """
        if not output:
            return {"response": "", "empty": True}

        # Try to parse as JSON
        try:
            data = json.loads(output)
            return data
        except json.JSONDecodeError:
            # Return as plain text
            return {"response": output, "format": "text"}

    def execute_simple(
        self, prompt: str, model: str = "sonnet", session_id: Optional[str] = None
    ) -> str:
        """Execute a simple command and return text response.

        Args:
            prompt: The prompt
            model: Model to use
            session_id: Optional session ID

        Returns:
            Text response
        """
        command = ClaudeCommand(
            prompt=prompt,
            model=model,
            output_format=OutputFormat.TEXT,
            session_id=session_id,
            print_mode=True,
        )

        result = self.execute(command)

        if "error" in result:
            raise RuntimeError(f"Command failed: {result['message']}")

        return result.get("response", "")

    def continue_conversation(self, prompt: str, model: str = "sonnet") -> str:
        """Continue the most recent conversation.

        Args:
            prompt: New prompt
            model: Model to use

        Returns:
            Text response
        """
        command = ClaudeCommand(
            prompt=prompt,
            model=model,
            continue_last=True,
            output_format=OutputFormat.TEXT,
        )

        result = self.execute(command)
        return result.get("response", "")


class SessionManager:
    """Manages Claude Code CLI sessions for context retention."""

    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, str] = {}
        self.cli = ClaudeCodeCLI()

    def create_session(self, session_id: str) -> str:
        """Create a new session.

        Args:
            session_id: Unique session identifier

        Returns:
            Session ID
        """
        self.sessions[session_id] = session_id
        logger.info(f"Created session: {session_id}")
        return session_id

    def execute_in_session(
        self, session_id: str, prompt: str, model: str = "sonnet"
    ) -> str:
        """Execute a command in a specific session.

        Args:
            session_id: Session ID
            prompt: The prompt
            model: Model to use

        Returns:
            Text response
        """
        if session_id not in self.sessions:
            self.create_session(session_id)

        command = ClaudeCommand(
            prompt=prompt,
            model=model,
            session_id=session_id,
            output_format=OutputFormat.TEXT,
        )

        result = self.cli.execute(command)
        return result.get("response", "")

    def fork_session(self, original_session_id: str, new_session_id: str) -> str:
        """Fork an existing session.

        Args:
            original_session_id: Original session ID
            new_session_id: New session ID

        Returns:
            New session ID
        """
        if original_session_id not in self.sessions:
            raise ValueError(f"Session {original_session_id} not found")

        command = ClaudeCommand(
            prompt="",
            resume_session=original_session_id,
            fork_session=True,
            session_id=new_session_id,
            output_format=OutputFormat.JSON,
        )

        result = self.cli.execute(command)

        if "error" not in result:
            self.sessions[new_session_id] = new_session_id
            logger.info(f"Forked session {original_session_id} -> {new_session_id}")

        return new_session_id

    def clear_session(self, session_id: str) -> None:
        """Clear a session.

        Args:
            session_id: Session ID to clear
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")


__all__ = [
    "ClaudeCommand",
    "ClaudeCodeCLI",
    "SessionManager",
    "OutputFormat",
    "PermissionMode",
]
