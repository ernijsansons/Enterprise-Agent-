"""Caching utilities with TTL support."""
from __future__ import annotations

import hashlib
import json
import logging
import pickle  # nosec B403
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with expiration and quality tracking."""

    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    hits: int = 0
    quality_score: Optional[float] = None
    feedback: Optional[Dict[str, Any]] = None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def access(self) -> Any:
        """Access the cached value and increment hit count."""
        self.hits += 1
        return self.value

    def update_quality(
        self, score: float, feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update quality metrics for this cached item."""
        self.quality_score = score
        if feedback:
            self.feedback = feedback


class TTLCache:
    """Thread-safe cache with time-to-live support."""

    def __init__(
        self,
        default_ttl: float = 300,  # 5 minutes default
        max_size: Optional[int] = 1000,
        cleanup_interval: float = 60,  # Cleanup every minute
    ):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            self._maybe_cleanup()

            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return default

            if entry.is_expired():
                del self._cache[key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return default

            self._stats["hits"] += 1
            return entry.access()

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self._lock:
            if self.max_size and len(self._cache) >= self.max_size:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                value=value, ttl=ttl if ttl is not None else self.default_ttl
            )

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def _maybe_cleanup(self) -> None:
        """Perform cleanup if needed."""
        if time.time() - self._last_cleanup > self.cleanup_interval:
            self._cleanup()
            self._last_cleanup = time.time()

    def _cleanup(self) -> None:
        """Remove expired entries."""
        expired = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired.append(key)

        for key in expired:
            del self._cache[key]
            self._stats["expirations"] += 1

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find entry with oldest timestamp and lowest hits
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].timestamp, -self._cache[k].hits),
        )

        del self._cache[lru_key]
        self._stats["evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                **self._stats,
                "size": len(self._cache),
                "hit_rate": hit_rate,
            }

    def update_quality(
        self, key: str, score: float, feedback: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update quality metrics for a cached item."""
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                entry.update_quality(score, feedback)

    def get_quality_insights(self) -> Dict[str, Any]:
        """Get quality insights from cached items."""
        with self._lock:
            quality_scores = []
            high_quality_items = 0
            low_quality_items = 0

            for entry in self._cache.values():
                if entry.quality_score is not None:
                    quality_scores.append(entry.quality_score)
                    if entry.quality_score >= 0.8:
                        high_quality_items += 1
                    elif entry.quality_score < 0.5:
                        low_quality_items += 1

            avg_quality = (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            )

            return {
                "average_quality": avg_quality,
                "high_quality_items": high_quality_items,
                "low_quality_items": low_quality_items,
                "total_rated_items": len(quality_scores),
                "unrated_items": len(self._cache) - len(quality_scores),
            }

    def get_low_quality_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in low-quality cached items for learning."""
        with self._lock:
            low_quality_patterns = []

            for key, entry in self._cache.items():
                if entry.quality_score is not None and entry.quality_score < 0.6:
                    pattern = {
                        "key": key[:100],  # Truncate for privacy
                        "score": entry.quality_score,
                        "hits": entry.hits,
                        "feedback": entry.feedback,
                        "timestamp": entry.timestamp,
                    }
                    low_quality_patterns.append(pattern)

            # Sort by score (worst first) and limit to 10
            def get_score(x: Dict[str, Any]) -> float:
                score = x.get("score", 0)
                return float(score) if isinstance(score, (int, float, str)) else 0.0

            return sorted(low_quality_patterns, key=get_score)[:10]


class ModelResponseCache(TTLCache):
    """Specialized cache for model responses."""

    def __init__(
        self,
        default_ttl: float = 600,  # 10 minutes for model responses
        max_size: int = 500,
        **kwargs,
    ):
        super().__init__(default_ttl=default_ttl, max_size=max_size, **kwargs)

    @staticmethod
    def _create_key(
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Create cache key from model parameters."""
        key_data = {
            "model": model,
            "prompt": prompt[:500],  # Truncate very long prompts
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get_response(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Get cached model response."""
        key = self._create_key(model, prompt, temperature, max_tokens)
        return self.get(key)

    def cache_response(
        self,
        model: str,
        prompt: str,
        response: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache model response."""
        key = self._create_key(model, prompt, temperature, max_tokens)
        self.set(key, response, ttl)


def cached(
    ttl: Optional[float] = None,
    cache_instance: Optional[TTLCache] = None,
    key_func: Optional[Callable] = None,
):
    """Decorator for caching function results."""
    if cache_instance is None:
        cache_instance = TTLCache(default_ttl=ttl or 300)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

            # Try to get from cache
            result = cache_instance.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            logger.debug(f"Cached result for {func.__name__}")

            return result

        # Attach cache instance to wrapper for testing/debugging
        wrapper._cache = cache_instance  # type: ignore[attr-defined]
        return wrapper

    return decorator


class DiskCache:
    """Persistent disk-based cache."""

    def __init__(
        self,
        cache_dir: str = ".cache",
        default_ttl: float = 3600,  # 1 hour default
        serializer: str = "pickle",  # "pickle" or "json"
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_ttl = default_ttl
        self.serializer = serializer

    def _get_path(self, key: str) -> Path:
        """Get file path for cache key."""
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from disk cache."""
        path = self._get_path(key)

        if not path.exists():
            return default

        try:
            with open(path, "rb" if self.serializer == "pickle" else "r") as f:
                if self.serializer == "pickle":
                    data = pickle.load(f)  # nosec B301
                else:
                    data = json.load(f)

            # Check expiration
            if data.get("expires") and data["expires"] < time.time():
                path.unlink()  # Delete expired file
                return default

            return data["value"]

        except Exception as exc:
            logger.warning(f"Failed to read cache {key}: {exc}")
            return default

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in disk cache."""
        path = self._get_path(key)
        ttl = ttl if ttl is not None else self.default_ttl

        data = {
            "value": value,
            "timestamp": time.time(),
            "expires": time.time() + ttl if ttl else None,
        }

        try:
            with open(path, "wb" if self.serializer == "pickle" else "w") as f:
                if self.serializer == "pickle":
                    pickle.dump(data, f)
                else:
                    json.dump(data, f)

        except Exception as exc:
            logger.warning(f"Failed to write cache {key}: {exc}")

    def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        path = self._get_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Clear all cache files."""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()


# Global cache instances
_model_cache = None
_general_cache = None


def get_model_cache() -> ModelResponseCache:
    """Get global model response cache."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelResponseCache()
    return _model_cache


def get_cache() -> TTLCache:
    """Get global general cache."""
    global _general_cache
    if _general_cache is None:
        _general_cache = TTLCache()
    return _general_cache


__all__ = [
    "CacheEntry",
    "TTLCache",
    "ModelResponseCache",
    "cached",
    "DiskCache",
    "get_model_cache",
    "get_cache",
]
