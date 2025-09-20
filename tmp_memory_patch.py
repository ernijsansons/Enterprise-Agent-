try:
    from src.memory import MemoryStore
except ImportError:  # pragma: no cover
    from src.memory.storage import MemoryStore
