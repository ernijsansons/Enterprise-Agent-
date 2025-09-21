"""Async cache manager with LRU eviction and cache warming."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.timestamp + self.ttl

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.timestamp


class AsyncLRUCache:
    """Thread-safe async LRU cache with TTL support."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        max_memory_mb: int = 100,
    ):
        """Initialize async LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock()  # For sync access
        self._total_memory = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Cache warming
        self._warming_tasks: Set[asyncio.Task] = set()
        self._popular_keys: Dict[str, int] = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                await self._remove_key(key)
                self._misses += 1
                return None

            # Update access statistics
            entry.access_count += 1
            entry.last_access = time.time()

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            self._hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (uses default if None)
            force: Force set even if memory limit exceeded

        Returns:
            True if set successfully, False if rejected
        """
        async with self._lock:
            ttl = ttl or self.default_ttl
            now = time.time()

            # Calculate size
            size_bytes = self._estimate_size(value)

            # Check memory limits
            if not force and self._total_memory + size_bytes > self.max_memory_bytes:
                # Try to evict some entries first
                await self._evict_to_fit(size_bytes)

                # Check again
                if self._total_memory + size_bytes > self.max_memory_bytes:
                    logger.warning(f"Cache memory limit exceeded, rejecting key: {key}")
                    return False

            # Remove old entry if exists
            if key in self._cache:
                await self._remove_key(key)

            # Create new entry
            entry = CacheEntry(
                value=value,
                timestamp=now,
                ttl=ttl,
                access_count=1,
                last_access=now,
                size_bytes=size_bytes,
            )

            # Add to cache
            self._cache[key] = entry
            self._total_memory += size_bytes

            # Track popularity for warming
            self._popular_keys[key] = self._popular_keys.get(key, 0) + 1

            # Evict if over size limit
            if len(self._cache) > self.max_size:
                await self._evict_lru()

            return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            if key in self._cache:
                await self._remove_key(key)
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._total_memory = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    async def _remove_key(self, key: str) -> None:
        """Remove key and update memory tracking."""
        if key in self._cache:
            entry = self._cache[key]
            self._total_memory -= entry.size_bytes
            del self._cache[key]

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)  # Remove oldest
            self._evictions += 1
            logger.debug(f"Evicted LRU cache entry: {key}")

    async def _evict_to_fit(self, needed_bytes: int) -> None:
        """Evict entries to make room for new entry."""
        while self._cache and self._total_memory + needed_bytes > self.max_memory_bytes:
            await self._evict_lru()

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if isinstance(value, str):
                return len(value.encode("utf-8"))
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value).encode("utf-8"))
            elif isinstance(value, bytes):
                return len(value)
            else:
                return len(str(value).encode("utf-8"))
        except Exception:
            return 1024  # Default estimate

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "memory_mb": round(self._total_memory / (1024 * 1024), 2),
                "max_memory_mb": round(self.max_memory_bytes / (1024 * 1024), 2),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
                "evictions": self._evictions,
                "popular_keys": list(
                    sorted(
                        self._popular_keys.items(), key=lambda x: x[1], reverse=True
                    )[:10]
                ),
            }

    async def warm_cache(self, keys_and_generators: List[Tuple[str, callable]]) -> None:
        """Warm cache with frequently accessed data.

        Args:
            keys_and_generators: List of (key, async_generator_function) tuples
        """
        logger.info(f"Starting cache warming for {len(keys_and_generators)} keys")

        async def warm_key(key: str, generator: callable):
            try:
                value = await generator()
                await self.set(key, value, force=True)
                logger.debug(f"Warmed cache key: {key}")
            except Exception as e:
                logger.warning(f"Failed to warm cache key '{key}': {e}")

        # Create warming tasks
        tasks = [
            asyncio.create_task(warm_key(key, gen)) for key, gen in keys_and_generators
        ]

        # Track tasks for cleanup
        self._warming_tasks.update(tasks)

        # Wait for completion
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # Clean up completed tasks
            for task in tasks:
                self._warming_tasks.discard(task)

        logger.info("Cache warming completed")

    def get_popular_keys(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most popular cache keys.

        Args:
            limit: Maximum number of keys to return

        Returns:
            List of (key, access_count) tuples
        """
        with self._thread_lock:
            return sorted(self._popular_keys.items(), key=lambda x: x[1], reverse=True)[
                :limit
            ]

    # Sync methods for compatibility
    def get_sync(self, key: str) -> Optional[Any]:
        """Synchronous get method."""
        with self._thread_lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                return None

            entry.access_count += 1
            entry.last_access = time.time()
            self._cache.move_to_end(key)
            return entry.value

    def set_sync(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Synchronous set method."""
        with self._thread_lock:
            ttl = ttl or self.default_ttl
            size_bytes = self._estimate_size(value)

            # Simple eviction for sync mode
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size_bytes,
            )

            self._cache[key] = entry
            return True


class ModelResponseCache:
    """Specialized cache for model responses."""

    def __init__(self, max_size: int = 500, default_ttl: float = 1800.0):
        """Initialize model response cache.

        Args:
            max_size: Maximum number of cached responses
            default_ttl: Default TTL for responses
        """
        self.cache = AsyncLRUCache(max_size=max_size, default_ttl=default_ttl)

    def _make_key(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        role: Optional[str] = None,
    ) -> str:
        """Create cache key for model response.

        Args:
            model: Model name
            prompt: Input prompt
            max_tokens: Max tokens limit
            role: Optional role context

        Returns:
            Cache key string
        """
        # Create stable hash of prompt content
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        # Include role in key for context-specific caching
        key_parts = [model, prompt_hash, str(max_tokens)]
        if role:
            key_parts.append(role)

        return ":".join(key_parts)

    async def get_response(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        role: Optional[str] = None,
    ) -> Optional[str]:
        """Get cached model response.

        Args:
            model: Model name
            prompt: Input prompt
            max_tokens: Max tokens limit
            role: Optional role context

        Returns:
            Cached response or None
        """
        key = self._make_key(model, prompt, max_tokens, role)
        return await self.cache.get(key)

    async def cache_response(
        self,
        model: str,
        prompt: str,
        response: str,
        max_tokens: int,
        role: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> bool:
        """Cache model response.

        Args:
            model: Model name
            prompt: Input prompt
            response: Model response
            max_tokens: Max tokens limit
            role: Optional role context
            ttl: Cache TTL (uses default if None)

        Returns:
            True if cached successfully
        """
        key = self._make_key(model, prompt, max_tokens, role)
        return await self.cache.set(key, response, ttl)

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return await self.cache.get_stats()

    # Sync compatibility methods
    def get_response_sync(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        role: Optional[str] = None,
    ) -> Optional[str]:
        """Sync version of get_response."""
        key = self._make_key(model, prompt, max_tokens, role)
        return self.cache.get_sync(key)

    def cache_response_sync(
        self,
        model: str,
        prompt: str,
        response: str,
        max_tokens: int,
        role: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> bool:
        """Sync version of cache_response."""
        key = self._make_key(model, prompt, max_tokens, role)
        return self.cache.set_sync(key, response, ttl)


# Global instances
_model_cache: Optional[ModelResponseCache] = None
_general_cache: Optional[AsyncLRUCache] = None


def get_async_model_cache() -> ModelResponseCache:
    """Get global async model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelResponseCache()
    return _model_cache


def get_async_cache() -> AsyncLRUCache:
    """Get global async cache instance."""
    global _general_cache
    if _general_cache is None:
        _general_cache = AsyncLRUCache()
    return _general_cache


__all__ = [
    "CacheEntry",
    "AsyncLRUCache",
    "ModelResponseCache",
    "get_async_model_cache",
    "get_async_cache",
]
