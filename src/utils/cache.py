"""Caching utilities with TTL support and configurable controls."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle  # nosec B403
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    enabled: bool = True
    default_ttl: float = 300  # 5 minutes
    max_size: Optional[int] = 1000
    cleanup_interval: float = 60  # 1 minute
    adaptive_ttl: bool = False  # Enable adaptive TTL based on quality
    quality_threshold: float = 0.8  # Minimum quality for extended TTL
    high_quality_ttl_multiplier: float = 2.0  # Multiply TTL for high-quality items
    low_quality_ttl_multiplier: float = 0.5  # Reduce TTL for low-quality items
    persistence_enabled: bool = False  # Enable disk persistence
    persistence_path: str = ".cache"
    compression_enabled: bool = False  # Enable compression for large values
    compression_threshold: int = 1024  # Compress values larger than this (bytes)
    warmup_enabled: bool = False  # Enable cache warming
    eviction_policy: str = "lru"  # "lru", "lfu", "ttl"
    metrics_enabled: bool = True  # Enable detailed metrics collection

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CacheConfig":
        """Create CacheConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create CacheConfig from environment variables."""
        return cls(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            default_ttl=float(os.getenv("CACHE_DEFAULT_TTL", "300")),
            max_size=int(os.getenv("CACHE_MAX_SIZE", "1000"))
            if os.getenv("CACHE_MAX_SIZE")
            else None,
            cleanup_interval=float(os.getenv("CACHE_CLEANUP_INTERVAL", "60")),
            adaptive_ttl=os.getenv("CACHE_ADAPTIVE_TTL", "false").lower() == "true",
            quality_threshold=float(os.getenv("CACHE_QUALITY_THRESHOLD", "0.8")),
            high_quality_ttl_multiplier=float(
                os.getenv("CACHE_HIGH_QUALITY_TTL_MULTIPLIER", "2.0")
            ),
            low_quality_ttl_multiplier=float(
                os.getenv("CACHE_LOW_QUALITY_TTL_MULTIPLIER", "0.5")
            ),
            persistence_enabled=os.getenv("CACHE_PERSISTENCE_ENABLED", "false").lower()
            == "true",
            persistence_path=os.getenv("CACHE_PERSISTENCE_PATH", ".cache"),
            compression_enabled=os.getenv("CACHE_COMPRESSION_ENABLED", "false").lower()
            == "true",
            compression_threshold=int(os.getenv("CACHE_COMPRESSION_THRESHOLD", "1024")),
            warmup_enabled=os.getenv("CACHE_WARMUP_ENABLED", "false").lower() == "true",
            eviction_policy=os.getenv("CACHE_EVICTION_POLICY", "lru").lower(),
            metrics_enabled=os.getenv("CACHE_METRICS_ENABLED", "true").lower()
            == "true",
        )


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
    """Thread-safe cache with time-to-live support and configurable controls."""

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        default_ttl: Optional[float] = None,
        max_size: Optional[int] = None,
        cleanup_interval: Optional[float] = None,
    ):
        # Use provided config or create default
        self.config = config or CacheConfig()

        # Override config with explicit parameters if provided
        if default_ttl is not None:
            self.config.default_ttl = default_ttl
        if max_size is not None:
            self.config.max_size = max_size
        if cleanup_interval is not None:
            self.config.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._access_order: List[str] = []  # For LRU tracking
        self._access_frequency: Dict[str, int] = {}  # For LFU tracking

        # Enhanced statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "adaptive_ttl_adjustments": 0,
            "compression_saves": 0,
            "persistence_ops": 0,
        }

        # Initialize persistence if enabled
        if self.config.persistence_enabled:
            self._persistence_path = Path(self.config.persistence_path)
            self._persistence_path.mkdir(exist_ok=True)
            self._load_persistent_cache()

        # Initialize compression if enabled
        if self.config.compression_enabled:
            try:
                import zlib

                self._compression = zlib
            except ImportError:
                logger.warning("zlib not available, disabling compression")
                self.config.compression_enabled = False

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with configurable behavior."""
        if not self.config.enabled:
            return default

        with self._lock:
            self._maybe_cleanup()

            entry = self._cache.get(key)

            if entry is None:
                if self.config.metrics_enabled:
                    self._stats["misses"] += 1
                return default

            if entry.is_expired():
                self._remove_entry(key)
                if self.config.metrics_enabled:
                    self._stats["expirations"] += 1
                    self._stats["misses"] += 1
                return default

            # Update access tracking for eviction policies
            self._update_access_tracking(key)

            if self.config.metrics_enabled:
                self._stats["hits"] += 1

            # Decompress value if needed
            value = entry.access()
            if self.config.compression_enabled and isinstance(value, bytes):
                try:
                    value = self._compression.decompress(value).decode("utf-8")
                except Exception:
                    pass  # Return compressed value if decompression fails

            return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        quality_score: Optional[float] = None,
    ) -> None:
        """Set value in cache with configurable behavior."""
        if not self.config.enabled:
            return

        with self._lock:
            # Handle eviction if cache is full
            if self.config.max_size and len(self._cache) >= self.config.max_size:
                self._evict_entry()

            # Calculate adaptive TTL if enabled
            effective_ttl = ttl if ttl is not None else self.config.default_ttl
            if self.config.adaptive_ttl and quality_score is not None:
                effective_ttl = self._calculate_adaptive_ttl(
                    effective_ttl, quality_score
                )

            # Compress value if enabled and meets threshold
            stored_value = value
            if self.config.compression_enabled and isinstance(value, str):
                value_bytes = value.encode("utf-8")
                if len(value_bytes) > self.config.compression_threshold:
                    try:
                        stored_value = self._compression.compress(value_bytes)
                        if self.config.metrics_enabled:
                            self._stats["compression_saves"] += 1
                    except Exception:
                        stored_value = value  # Use original if compression fails

            # Create cache entry
            entry = CacheEntry(value=stored_value, ttl=effective_ttl)
            if quality_score is not None:
                entry.update_quality(quality_score)

            self._cache[key] = entry
            self._update_access_tracking(key)

            # Persist if enabled
            if self.config.persistence_enabled:
                self._persist_entry(key, entry)

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

    def _calculate_adaptive_ttl(self, base_ttl: float, quality_score: float) -> float:
        """Calculate adaptive TTL based on quality score."""
        if quality_score >= self.config.quality_threshold:
            adjusted_ttl = base_ttl * self.config.high_quality_ttl_multiplier
        else:
            adjusted_ttl = base_ttl * self.config.low_quality_ttl_multiplier

        if self.config.metrics_enabled:
            self._stats["adaptive_ttl_adjustments"] += 1

        return adjusted_ttl

    def _update_access_tracking(self, key: str) -> None:
        """Update access tracking for eviction policies."""
        # Update LRU tracking
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        # Update LFU tracking
        self._access_frequency[key] = self._access_frequency.get(key, 0) + 1

    def _remove_entry(self, key: str) -> None:
        """Remove entry and clean up tracking."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
        if key in self._access_frequency:
            del self._access_frequency[key]

    def _evict_entry(self) -> None:
        """Evict entry based on configured policy."""
        if not self._cache:
            return

        if self.config.eviction_policy == "lru":
            self._evict_lru()
        elif self.config.eviction_policy == "lfu":
            self._evict_lfu()
        elif self.config.eviction_policy == "ttl":
            self._evict_ttl()
        else:
            self._evict_lru()  # Default to LRU

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order[0]
            self._remove_entry(lru_key)
            if self.config.metrics_enabled:
                self._stats["evictions"] += 1

    def _evict_lfu(self) -> None:
        """Evict least frequently used entry."""
        if not self._access_frequency:
            return

        lfu_key = min(
            self._access_frequency.keys(), key=lambda k: self._access_frequency[k]
        )
        self._remove_entry(lfu_key)
        if self.config.metrics_enabled:
            self._stats["evictions"] += 1

    def _evict_ttl(self) -> None:
        """Evict entry with shortest remaining TTL."""
        if not self._cache:
            return

        current_time = time.time()
        ttl_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].timestamp + (self._cache[k].ttl or 0))
            - current_time,
        )
        self._remove_entry(ttl_key)
        if self.config.metrics_enabled:
            self._stats["evictions"] += 1

    def _persist_entry(self, key: str, entry: CacheEntry) -> None:
        """Persist cache entry to disk."""
        try:
            persist_path = (
                self._persistence_path
                / f"{hashlib.sha256(key.encode()).hexdigest()}.cache"
            )
            with open(persist_path, "wb") as f:
                pickle.dump({"key": key, "entry": entry}, f)
            if self.config.metrics_enabled:
                self._stats["persistence_ops"] += 1
        except Exception as e:
            logger.warning(f"Failed to persist cache entry {key}: {e}")

    def _load_persistent_cache(self) -> None:
        """Load persistent cache entries from disk."""
        try:
            for cache_file in self._persistence_path.glob("*.cache"):
                try:
                    with open(cache_file, "rb") as f:
                        data = pickle.load(f)
                        key = data["key"]
                        entry = data["entry"]
                        if not entry.is_expired():
                            self._cache[key] = entry
                            self._update_access_tracking(key)
                        else:
                            cache_file.unlink()  # Remove expired file
                except Exception:
                    cache_file.unlink()  # Remove corrupted file
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")

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
