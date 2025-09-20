"""Memory storage abstraction."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

try:  # Optional dependency
    import pinecone  # type: ignore
except ImportError:  # pragma: no cover
    pinecone = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MemoryStore:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        types = self.config.get("types", ["session"])
        self.stores: Dict[str, Dict[str, MemoryRecord]] = {level: {} for level in types}
        self.retention_days = int(self.config.get("retention_days", 30))

        self.pinecone_index = None
        if self.config.get("storage", "memory").startswith("hybrid") and pinecone:
            api_key = os.getenv("PINECONE_API_KEY")
            if api_key:
                try:
                    # Updated Pinecone API v3+ syntax
                    from pinecone import Pinecone

                    pc = Pinecone(api_key=api_key)
                    index_name = self.config.get("pinecone_index", "memory-index")
                    self.pinecone_index = pc.Index(index_name)
                except (
                    Exception
                ) as exc:  # pragma: no cover - handle missing index gracefully
                    logger.warning("Pinecone index initialisation failed: %s", exc)
                    self.pinecone_index = None

    # ---------------------------------------------------------------- store/retrieve
    def store(self, level: str, key: str, value: Any) -> None:
        store = self.stores.setdefault(level, {})
        store[key] = MemoryRecord(value=value)
        if self.pinecone_index:
            self._upsert_vector(level, key, value)

    def retrieve(self, level: str, key: str, default: Any = None) -> Optional[Any]:
        record = self.stores.get(level, {}).get(key)
        return record.value if record else default

    def prune(self) -> None:
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        for level, store in self.stores.items():
            self.stores[level] = {
                key: record
                for key, record in store.items()
                if record.timestamp > cutoff
            }

    # ----------------------------------------------------------------- vectors
    def _upsert_vector(self, level: str, key: str, value: Any) -> None:
        if not self.pinecone_index:
            return

        # Generate proper vector dimensions based on the embedding model
        # Standard dimensions for common embedding models:
        # - OpenAI text-embedding-3-large: 3072
        # - OpenAI text-embedding-3-small: 1536
        # - OpenAI text-embedding-ada-002: 1536
        # - Cohere embed-v3: 1024
        # - Default fallback: 768 (common for many models)

        try:
            # Try to get actual embeddings if integration is available
            from src.tools.integrations import embed_code

            if isinstance(value, str):
                embedding_result = embed_code(value)
                vector = embedding_result.get("vector")
                if vector and isinstance(vector, list):
                    # Use actual embedding vector
                    metadata = {
                        "level": level,
                        "key": key,
                        "model": embedding_result.get("model", "unknown"),
                        "text_preview": value[:200] if value else "",
                    }
                else:
                    # Fallback to default dimensions if embedding fails
                    vector = self._generate_placeholder_vector(value)
                    metadata = {
                        "level": level,
                        "key": key,
                        "text_preview": value[:200] if value else "",
                    }
            else:
                # For non-string values, generate placeholder vector
                vector = self._generate_placeholder_vector(str(value))
                metadata = {"level": level, "key": key, "type": type(value).__name__}

            # Ensure vector has correct dimensions for the index
            vector = self._ensure_vector_dimensions(vector)

            # Upsert to Pinecone with proper error handling
            self.pinecone_index.upsert([(f"{level}_{key}", vector, metadata)])
            logger.debug(
                f"Upserted vector for {level}/{key} with {len(vector)} dimensions"
            )

        except ImportError:
            # If integrations module is not available, use placeholder
            vector = self._generate_placeholder_vector(str(value))
            metadata = {"level": level, "key": key}
            try:
                self.pinecone_index.upsert([(f"{level}_{key}", vector, metadata)])
            except Exception as exc:
                logger.debug(f"Vector upsert failed for {level}/{key}: {exc}")
        except Exception as exc:
            logger.debug(f"Vector generation/upsert failed for {level}/{key}: {exc}")

    def _generate_placeholder_vector(
        self, text: str, dimensions: int = 768
    ) -> list[float]:
        """Generate a deterministic placeholder vector from text."""
        import hashlib

        # Create a deterministic hash-based vector
        hash_obj = hashlib.sha256(text.encode("utf-8"))
        hash_bytes = hash_obj.digest()

        # Convert hash to float vector with requested dimensions
        vector = []
        for i in range(dimensions):
            # Use different parts of the hash for each dimension
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1] range
            value = (hash_bytes[byte_idx] / 127.5) - 1.0
            vector.append(value)

        return vector

    def _ensure_vector_dimensions(self, vector: list[float]) -> list[float]:
        """Ensure vector matches expected dimensions for the index."""
        expected_dim = self.config.get("vector_dimensions", 768)

        if len(vector) == expected_dim:
            return vector
        elif len(vector) < expected_dim:
            # Pad with zeros if vector is too short
            return vector + [0.0] * (expected_dim - len(vector))
        else:
            # Truncate if vector is too long
            return vector[:expected_dim]


__all__ = ["MemoryStore"]
