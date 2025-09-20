from datetime import datetime, timedelta

from src.memory import MemoryStore


def test_memory_store_basic_store_retrieve():
    store = MemoryStore(
        {"types": ["session"], "storage": "memory", "retention_days": 30}
    )
    store.store("session", "key", "value")
    assert store.retrieve("session", "key") == "value"


def test_memory_store_prune():
    store = MemoryStore(
        {"types": ["session"], "storage": "memory", "retention_days": 1}
    )
    store.store("session", "old", "value")
    # Manually age record
    store.stores["session"]["old"].timestamp = datetime.utcnow() - timedelta(days=2)
    store.prune()
    assert "old" not in store.stores["session"]


def test_memory_store_handles_unknown_level():
    store = MemoryStore({"types": ["session"], "storage": "memory"})
    assert store.retrieve("missing", "key") is None
    store.store("custom", "key", "value")
    assert store.retrieve("custom", "key") == "value"
