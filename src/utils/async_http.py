"""Async HTTP client with retry and timeout handling."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from src.utils.circuit_breaker import CircuitBreakerError, get_circuit_breaker_registry
from src.utils.rate_limiter import RateLimitExceeded, get_rate_limiter

logger = logging.getLogger(__name__)


class AsyncHTTPClient:
    """Async HTTP client with resilience patterns."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        user_agent: str = "Enterprise-Agent/1.0",
    ):
        """Initialize async HTTP client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
            user_agent: User agent string
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError(
                "aiohttp is required for async HTTP. Install with: pip install aiohttp"
            )

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.user_agent = user_agent

        # Session will be created lazily
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

        # Get circuit breaker and rate limiter
        self.rate_limiter = get_rate_limiter()
        self.circuit_registry = get_circuit_breaker_registry()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    connector = aiohttp.TCPConnector(
                        limit=100,  # Connection pool limit
                        limit_per_host=30,  # Per-host limit
                        ttl_dns_cache=300,  # DNS cache TTL
                        use_dns_cache=True,
                    )

                    headers = {
                        "User-Agent": self.user_agent,
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip, deflate",
                    }

                    self._session = aiohttp.ClientSession(
                        timeout=timeout,
                        connector=connector,
                        headers=headers,
                        json_serialize=json.dumps,
                    )

        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        params: Optional[Dict[str, str]] = None,
        rate_limit_key: Optional[str] = None,
        circuit_breaker_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make async HTTP request with resilience patterns.

        Args:
            method: HTTP method
            url: Request URL
            headers: Optional headers
            json_data: JSON data to send
            data: Raw data to send
            params: URL parameters
            rate_limit_key: Key for rate limiting
            circuit_breaker_key: Key for circuit breaker
            **kwargs: Additional aiohttp arguments

        Returns:
            Response data dictionary

        Raises:
            RateLimitExceeded: If rate limited
            CircuitBreakerError: If circuit breaker is open
            aiohttp.ClientError: For HTTP errors
        """
        # Check rate limit
        if rate_limit_key and not self.rate_limiter.acquire(rate_limit_key, 1):
            status = self.rate_limiter.get_status(rate_limit_key)
            wait_time = status.get("time_until_token", 1.0)
            raise RateLimitExceeded(
                f"Rate limit exceeded for '{rate_limit_key}'. Wait {wait_time:.1f}s.",
                retry_after=wait_time,
                key=rate_limit_key,
            )

        # Get circuit breaker if specified
        circuit_breaker = None
        if circuit_breaker_key:
            circuit_breaker = self.circuit_registry.get_breaker(circuit_breaker_key)

        async def make_request():
            session = await self._get_session()

            # Merge headers
            request_headers = {}
            if headers:
                request_headers.update(headers)

            # Make the request
            async with session.request(
                method=method,
                url=url,
                headers=request_headers,
                json=json_data,
                data=data,
                params=params,
                **kwargs,
            ) as response:
                # Read response
                response_text = await response.text()

                # Parse JSON if possible
                try:
                    response_data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    response_data = {"text": response_text}

                # Check for HTTP errors
                if response.status >= 400:
                    error_msg = response_data.get("error", response_text)
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=error_msg,
                    )

                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "data": response_data,
                    "text": response_text,
                }

        # Execute with circuit breaker if specified
        if circuit_breaker:
            try:
                return circuit_breaker.call(make_request)
            except CircuitBreakerError:
                raise
        else:
            return await make_request()

    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make POST request."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make PUT request."""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    async def batch_requests(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 10,
    ) -> List[Dict[str, Any]]:
        """Execute multiple HTTP requests concurrently.

        Args:
            requests: List of request dictionaries
            max_concurrent: Maximum concurrent requests

        Returns:
            List of response dictionaries
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_request(request_config):
            async with semaphore:
                try:
                    return await self.request(**request_config)
                except Exception as e:
                    logger.error(f"Batch request failed: {e}")
                    return {
                        "error": str(e),
                        "status": 0,
                        "data": {},
                        "text": "",
                    }

        tasks = [bounded_request(req) for req in requests]
        return await asyncio.gather(*tasks)


class AsyncOpenAIClient:
    """Async OpenAI API client."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            base_url: API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.http_client = AsyncHTTPClient(user_agent="Enterprise-Agent-OpenAI/1.0")

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create chat completion.

        Args:
            model: Model name
            messages: Chat messages
            max_tokens: Maximum tokens
            temperature: Temperature setting
            **kwargs: Additional parameters

        Returns:
            Chat completion response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        response = await self.http_client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json_data=data,
            rate_limit_key="openai_api",
            circuit_breaker_key="openai_api",
        )

        return response["data"]

    async def close(self):
        """Close HTTP client."""
        await self.http_client.close()


class AsyncAnthropicClient:
    """Async Anthropic API client."""

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com"):
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            base_url: API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.http_client = AsyncHTTPClient(user_agent="Enterprise-Agent-Anthropic/1.0")

    async def messages_create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create message completion.

        Args:
            model: Model name
            messages: Messages
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Message response
        """
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs,
        }

        response = await self.http_client.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json_data=data,
            rate_limit_key="anthropic_api",
            circuit_breaker_key="anthropic_api",
        )

        return response["data"]

    async def close(self):
        """Close HTTP client."""
        await self.http_client.close()


# Global clients
_http_client: Optional[AsyncHTTPClient] = None
_openai_client: Optional[AsyncOpenAIClient] = None
_anthropic_client: Optional[AsyncAnthropicClient] = None


def get_async_http_client() -> AsyncHTTPClient:
    """Get global async HTTP client."""
    global _http_client
    if _http_client is None:
        _http_client = AsyncHTTPClient()
    return _http_client


async def cleanup_http_clients():
    """Clean up all HTTP clients."""
    global _http_client, _openai_client, _anthropic_client

    if _http_client:
        await _http_client.close()
        _http_client = None

    if _openai_client:
        await _openai_client.close()
        _openai_client = None

    if _anthropic_client:
        await _anthropic_client.close()
        _anthropic_client = None


__all__ = [
    "AsyncHTTPClient",
    "AsyncOpenAIClient",
    "AsyncAnthropicClient",
    "get_async_http_client",
    "cleanup_http_clients",
    "AIOHTTP_AVAILABLE",
]
