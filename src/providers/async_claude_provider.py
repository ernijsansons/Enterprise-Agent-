"""Async Claude Code CLI Provider for improved performance."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.exceptions import ModelException, ModelTimeoutException, RateLimitExceeded
from src.utils.async_cache import get_async_model_cache
from src.utils.circuit_breaker import get_circuit_breaker_registry, CircuitBreakerError
from src.utils.notifications import notify_authentication_issue, notify_cli_failure
from src.utils.rate_limiter import get_rate_limiter
from src.utils.security_audit import audit_authentication, audit_cli_usage
from src.utils.usage_monitor import can_make_claude_request, record_claude_usage

logger = logging.getLogger(__name__)


class AsyncClaudeCodeProvider:
    """Async version of Claude Code CLI provider for better performance."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize async Claude Code CLI provider.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.sessions: Dict[str, str] = {}
        self.cache = get_async_model_cache()

        # Setup persistent session storage
        self.session_file = Path.home() / ".claude" / "enterprise_sessions.json"

        # Get circuit breaker and rate limiter
        self.rate_limiter = get_rate_limiter()
        cb_registry = get_circuit_breaker_registry()
        self.circuit_breaker = cb_registry.get_breaker("claude_code_cli")

        logger.info("Async Claude Code provider initialized")

    async def call_model(
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
        """Async call to Claude Code CLI.

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
        # Check rate limits
        if not self.rate_limiter.acquire("claude_code", 1):
            status = self.rate_limiter.get_status("claude_code")
            wait_time = status.get("time_until_token", 1.0)
            raise RateLimitExceeded(
                f"Rate limit exceeded for Claude Code. Wait {wait_time:.1f} seconds.",
                retry_after=wait_time,
                key="claude_code",
            )

        # Check cache first
        if use_cache:
            cached = await self.cache.get_response(model, prompt, max_tokens or 8192, role)
            if cached:
                logger.debug(f"Async cache hit for {role}/{operation}")
                return cached

        # Build command
        cli_model = self._map_model_to_cli(model)
        cmd = [
            "claude",
            "--print",
            "--model",
            cli_model,
            "--output-format",
            "json",
        ]

        # Add session management if provided
        if session_id and session_id in self.sessions:
            safe_session_id = shlex.quote(self.sessions[session_id])
            cmd.extend(["--resume", safe_session_id])

        # Add the prompt
        cmd.append(prompt)

        try:
            # Execute with circuit breaker protection
            result = await self._execute_claude_cli_async(cmd)

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                logger.error(f"Claude CLI error: {error_msg}")
                raise ModelException(
                    f"Claude Code CLI error: {error_msg}",
                    provider="claude_code",
                    model=cli_model,
                )

            # Parse response
            output = self._parse_cli_response(result.stdout)
            response_text = output.get("text", result.stdout)

            # Cache successful response
            if use_cache and response_text:
                ttl = 1800 if role == "Planner" else 900  # Longer TTL for planning
                await self.cache.cache_response(
                    model, prompt, response_text, max_tokens or 8192, role, ttl
                )

            return response_text

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker blocked call: {e}")
            notify_cli_failure(f"{role}/{operation}", f"Circuit breaker: {e}")
            raise ModelException(
                f"Claude Code service unavailable: {e}",
                provider="claude_code",
                model=cli_model,
            ) from e

        except Exception as e:
            logger.error(f"Async Claude CLI error: {e}")
            raise ModelException(
                f"Async Claude CLI error: {e}",
                provider="claude_code",
                model=cli_model,
            ) from e

    async def _execute_claude_cli_async(self, cmd: List[str]) -> asyncio.subprocess.Process:
        """Execute Claude CLI command asynchronously.

        Args:
            cmd: Command to execute

        Returns:
            Subprocess result

        Raises:
            ModelTimeoutException: If command times out
            ModelException: If command fails
        """
        try:
            # Ensure working directory is safe
            work_dir = self.config.get("working_directory", os.getcwd())
            if not os.path.isdir(work_dir):
                work_dir = os.getcwd()

            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
            )

            # Wait for completion with timeout
            timeout = self.config.get("timeout", 60)
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill the process if it times out
                process.kill()
                await process.wait()
                raise ModelTimeoutException(
                    f"Claude Code CLI timeout after {timeout}s",
                    provider="claude_code",
                )

            # Create a result object similar to subprocess.CompletedProcess
            class AsyncResult:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout.decode() if stdout else ""
                    self.stderr = stderr.decode() if stderr else ""

            return AsyncResult(process.returncode, stdout, stderr)

        except FileNotFoundError:
            logger.error("Claude Code CLI not found in PATH")
            raise ModelException(
                "Claude Code CLI not found. Please install it first.",
                provider="claude_code",
            )
        except Exception as e:
            logger.error(f"Error executing async Claude Code CLI: {e}")
            raise ModelException(
                f"Error executing async Claude Code CLI: {e}",
                provider="claude_code",
            ) from e

    def _map_model_to_cli(self, model: str) -> str:
        """Map internal model names to Claude Code CLI model names."""
        model_mapping = {
            "claude_sonnet_4": "sonnet",
            "claude-3-5-sonnet": "sonnet",
            "claude-3-5-sonnet-20241022": "sonnet",
            "claude_opus_4": "opus",
            "claude-3-opus": "opus",
            "claude_haiku": "haiku",
            "claude-3-haiku": "haiku",
        }

        if model in model_mapping:
            return model_mapping[model]

        model_lower = model.lower()
        if "sonnet" in model_lower:
            return "sonnet"
        elif "opus" in model_lower:
            return "opus"
        elif "haiku" in model_lower:
            return "haiku"

        logger.warning(f"Unknown model '{model}', defaulting to 'sonnet'")
        return "sonnet"

    def _parse_cli_response(self, output: str) -> Dict[str, Any]:
        """Parse Claude Code CLI output."""
        if not output:
            return {"text": ""}

        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"text": output}

    async def stream_response(
        self,
        prompt: str,
        model: str = "sonnet",
        callback: Optional[callable] = None,
        **kwargs,
    ) -> str:
        """Stream responses from Claude Code CLI (async version).

        Args:
            prompt: The prompt to send
            model: Model name
            callback: Optional callback for streaming chunks
            **kwargs: Additional arguments

        Returns:
            Complete response
        """
        # For now, implement as regular call
        # True streaming would require real-time subprocess communication
        response = await self.call_model(prompt, model, **kwargs)

        if callback:
            # Simulate streaming by chunking the response
            chunk_size = 50
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                await callback(chunk)
                await asyncio.sleep(0.01)  # Small delay to simulate streaming

        return response

    async def batch_call_models(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 3,
    ) -> List[str]:
        """Execute multiple model calls concurrently.

        Args:
            requests: List of request dictionaries with call_model parameters
            max_concurrent: Maximum concurrent requests

        Returns:
            List of responses in same order as requests
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_call(request):
            async with semaphore:
                return await self.call_model(**request)

        # Execute all requests concurrently with concurrency limit
        tasks = [bounded_call(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error strings
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch request {i} failed: {response}")
                results.append(f"Error: {response}")
            else:
                results.append(response)

        return results

    async def warm_cache(self, common_prompts: List[Dict[str, Any]]) -> None:
        """Warm cache with common prompts.

        Args:
            common_prompts: List of prompt dictionaries to pre-cache
        """
        logger.info(f"Warming cache with {len(common_prompts)} common prompts")

        async def generate_response(prompt_config):
            return await self.call_model(**prompt_config)

        # Create cache warming tasks
        warming_tasks = []
        for prompt_config in common_prompts:
            cache_key = f"{prompt_config.get('model', 'sonnet')}:{prompt_config['prompt'][:50]}"
            warming_tasks.append((cache_key, lambda pc=prompt_config: generate_response(pc)))

        # Use cache warming functionality
        await self.cache.cache.warm_cache(warming_tasks)

    async def get_stats(self) -> Dict[str, Any]:
        """Get async provider statistics.

        Returns:
            Dictionary with provider stats
        """
        cache_stats = await self.cache.get_stats()
        return {
            "provider": "async_claude_code",
            "cache": cache_stats,
            "rate_limiter": self.rate_limiter.get_status("claude_code"),
            "circuit_breaker": self.circuit_breaker.get_stats(),
        }


# Global instance
_async_provider: Optional[AsyncClaudeCodeProvider] = None


def get_async_claude_provider(
    config: Optional[Dict[str, Any]] = None
) -> AsyncClaudeCodeProvider:
    """Get or create async Claude Code provider instance.

    Args:
        config: Optional configuration

    Returns:
        AsyncClaudeCodeProvider instance
    """
    global _async_provider

    if _async_provider is None:
        _async_provider = AsyncClaudeCodeProvider(config)

    return _async_provider


__all__ = [
    "AsyncClaudeCodeProvider",
    "get_async_claude_provider",
]