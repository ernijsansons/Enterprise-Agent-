"""Async memory storage with batch processing and vector operations."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pinecone

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None

from src.utils.async_cache import get_async_cache

logger = logging.getLogger(__name__)


@dataclass
class AsyncMemoryRecord:
    """Memory record with async support."""

    value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "vector": self.vector,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access": self.last_access.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AsyncMemoryRecord:
        """Create from dictionary."""
        return cls(
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            vector=data.get("vector"),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_access=datetime.fromisoformat(
                data.get("last_access", data["timestamp"])
            ),
        )


class AsyncMemoryStore:
    """Async memory store with batch processing and vector operations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize async memory store.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.stores: Dict[str, Dict[str, AsyncMemoryRecord]] = {}
        self.retention_days = int(self.config.get("retention_days", 30))

        # Async cache for frequently accessed data
        self.cache = get_async_cache()

        # Vector store setup
        self.enable_vectors = self.config.get("enable_vectors", False)
        self.vector_dimension = self.config.get("vector_dimension", 384)

        # Batch processing settings
        self.batch_size = self.config.get("batch_size", 100)
        self.batch_timeout = self.config.get("batch_timeout", 5.0)
        self._pending_operations: List[Tuple[str, str, str, Any]] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None

        # Pinecone setup
        self.pinecone_index = None
        if (
            self.config.get("storage", "memory").startswith("hybrid")
            and PINECONE_AVAILABLE
        ):
            self._init_pinecone()

        logger.info("Async memory store initialized")

    def _init_pinecone(self) -> None:
        """Initialize Pinecone vector database."""
        try:
            import os

            api_key = os.getenv("PINECONE_API_KEY")
            if api_key:
                from pinecone import Pinecone

                pc = Pinecone(api_key=api_key)
                index_name = self.config.get("pinecone_index", "memory-index")
                self.pinecone_index = pc.Index(index_name)
                logger.info("Pinecone vector store initialized")
        except Exception as e:
            logger.warning(f"Pinecone initialization failed: {e}")
            self.pinecone_index = None

    async def store(
        self,
        level: str,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        vector: Optional[List[float]] = None,
    ) -> None:
        """Store value in memory.

        Args:
            level: Storage level
            key: Storage key
            value: Value to store
            metadata: Optional metadata
            vector: Optional vector representation
        """
        if level not in self.stores:
            self.stores[level] = {}

        record = AsyncMemoryRecord(
            value=value,
            metadata=metadata or {},
            vector=vector,
        )

        self.stores[level][key] = record

        # Cache frequently accessed data
        cache_key = f"{level}:{key}"
        await self.cache.set(cache_key, record.to_dict(), ttl=3600.0)

        # Add to batch processing queue for vector operations
        if self.enable_vectors or self.pinecone_index:
            await self._add_to_batch("store", level, key, record)

        logger.debug(f"Stored {level}:{key}")

    async def retrieve(
        self,
        level: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Retrieve value from memory.

        Args:
            level: Storage level
            key: Storage key
            default: Default value if not found

        Returns:
            Stored value or default
        """
        # Check cache first
        cache_key = f"{level}:{key}"
        cached = await self.cache.get(cache_key)
        if cached:
            # Update access statistics
            if level in self.stores and key in self.stores[level]:
                record = self.stores[level][key]
                record.access_count += 1
                record.last_access = datetime.utcnow()
            return cached["value"]

        # Check memory store
        if level in self.stores and key in self.stores[level]:
            record = self.stores[level][key]
            record.access_count += 1
            record.last_access = datetime.utcnow()

            # Update cache
            await self.cache.set(cache_key, record.to_dict(), ttl=3600.0)
            return record.value

        return default

    async def batch_store(
        self,
        operations: List[Tuple[str, str, Any, Optional[Dict[str, Any]]]],
    ) -> None:
        """Store multiple values in batch.

        Args:
            operations: List of (level, key, value, metadata) tuples
        """
        for level, key, value, metadata in operations:
            await self.store(level, key, value, metadata)

        logger.info(f"Batch stored {len(operations)} items")

    async def batch_retrieve(
        self,
        keys: List[Tuple[str, str]],
        default: Any = None,
    ) -> List[Any]:
        """Retrieve multiple values in batch.

        Args:
            keys: List of (level, key) tuples
            default: Default value for missing keys

        Returns:
            List of retrieved values
        """
        tasks = [self.retrieve(level, key, default) for level, key in keys]
        results = await asyncio.gather(*tasks)
        return results

    async def search(
        self,
        level: str,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[str, Any, float]]:
        """Search for similar items.

        Args:
            level: Storage level to search
            query: Search query
            limit: Maximum results
            similarity_threshold: Minimum similarity score

        Returns:
            List of (key, value, similarity_score) tuples
        """
        if level not in self.stores:
            return []

        results = []

        # Simple text matching for now
        query_lower = query.lower()
        for key, record in self.stores[level].items():
            value_str = str(record.value).lower()
            if query_lower in value_str:
                # Simple similarity score based on query presence
                score = len(query_lower) / len(value_str) if value_str else 0.0
                if score >= similarity_threshold:
                    results.append((key, record.value, score))

        # Sort by similarity score
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    async def vector_search(
        self,
        query_vector: List[float],
        level: str = "default",
        limit: int = 10,
        min_score: float = 0.7,
    ) -> List[Tuple[str, Any, float]]:
        """Search using vector similarity.

        Args:
            query_vector: Query vector
            level: Storage level
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of (key, value, score) tuples
        """
        if not self.enable_vectors:
            return []

        results = []

        # Search in Pinecone if available
        if self.pinecone_index:
            try:
                response = self.pinecone_index.query(
                    vector=query_vector,
                    top_k=limit,
                    include_metadata=True,
                    namespace=level,
                )

                for match in response.matches:
                    if match.score >= min_score:
                        key = match.metadata.get("key", match.id)
                        value = match.metadata.get("value")
                        results.append((key, value, match.score))

            except Exception as e:
                logger.warning(f"Pinecone search failed: {e}")

        # Fallback to local vector search
        if not results and level in self.stores and NUMPY_AVAILABLE:
            query_np = np.array(query_vector)

            for key, record in self.stores[level].items():
                if record.vector:
                    record_np = np.array(record.vector)
                    similarity = np.dot(query_np, record_np) / (
                        np.linalg.norm(query_np) * np.linalg.norm(record_np)
                    )
                    if similarity >= min_score:
                        results.append((key, record.value, float(similarity)))

            # Sort by similarity
            results.sort(key=lambda x: x[2], reverse=True)
            results = results[:limit]

        return results

    async def _add_to_batch(
        self,
        operation: str,
        level: str,
        key: str,
        data: Any,
    ) -> None:
        """Add operation to batch processing queue."""
        async with self._batch_lock:
            self._pending_operations.append((operation, level, key, data))

            # Start batch processing task if not running
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._process_batch())

    async def _process_batch(self) -> None:
        """Process batch operations."""
        await asyncio.sleep(self.batch_timeout)

        async with self._batch_lock:
            if not self._pending_operations:
                return

            operations = self._pending_operations[: self.batch_size]
            self._pending_operations = self._pending_operations[self.batch_size :]

        # Process vector operations
        vector_operations = []
        for operation, level, key, data in operations:
            if operation == "store" and isinstance(data, AsyncMemoryRecord):
                if data.vector or self.enable_vectors:
                    vector_operations.append((level, key, data))

        if vector_operations and self.pinecone_index:
            await self._batch_vector_upsert(vector_operations)

        # Continue processing if more operations pending
        if self._pending_operations:
            self._batch_task = asyncio.create_task(self._process_batch())

    async def _batch_vector_upsert(
        self,
        operations: List[Tuple[str, str, AsyncMemoryRecord]],
    ) -> None:
        """Batch upsert vectors to Pinecone."""
        try:
            upsert_data = []
            for level, key, record in operations:
                if record.vector:
                    upsert_data.append(
                        {
                            "id": f"{level}:{key}",
                            "values": record.vector,
                            "metadata": {
                                "level": level,
                                "key": key,
                                "value": str(record.value)[
                                    :1000
                                ],  # Truncate for metadata
                                "timestamp": record.timestamp.isoformat(),
                            },
                        }
                    )

            if upsert_data:
                self.pinecone_index.upsert(vectors=upsert_data, namespace="memory")
                logger.debug(f"Batch upserted {len(upsert_data)} vectors")

        except Exception as e:
            logger.error(f"Batch vector upsert failed: {e}")

    async def prune(self) -> int:
        """Remove old entries and clean up.

        Returns:
            Number of entries removed
        """
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        removed_count = 0

        for level in self.stores:
            keys_to_remove = []
            for key, record in self.stores[level].items():
                if record.timestamp < cutoff:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.stores[level][key]
                # Remove from cache
                cache_key = f"{level}:{key}"
                await self.cache.delete(cache_key)
                removed_count += 1

        logger.info(f"Pruned {removed_count} old memory entries")
        return removed_count

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics.

        Returns:
            Statistics dictionary
        """
        total_records = sum(len(store) for store in self.stores.values())
        cache_stats = await self.cache.get_stats()

        stats = {
            "total_records": total_records,
            "levels": {level: len(store) for level, store in self.stores.items()},
            "cache": cache_stats,
            "pending_batch_operations": len(self._pending_operations),
            "vector_enabled": self.enable_vectors,
            "pinecone_available": self.pinecone_index is not None,
        }

        return stats

    async def close(self) -> None:
        """Clean up resources."""
        # Cancel batch processing task
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Process any remaining batch operations
        if self._pending_operations:
            await self._process_batch()

        logger.info("Async memory store closed")


# Global instance
_async_memory_store: Optional[AsyncMemoryStore] = None


def get_async_memory_store(config: Optional[Dict[str, Any]] = None) -> AsyncMemoryStore:
    """Get global async memory store instance.

    Args:
        config: Optional configuration

    Returns:
        AsyncMemoryStore instance
    """
    global _async_memory_store
    if _async_memory_store is None:
        _async_memory_store = AsyncMemoryStore(config or {})
    return _async_memory_store


__all__ = [
    "AsyncMemoryRecord",
    "AsyncMemoryStore",
    "get_async_memory_store",
]
