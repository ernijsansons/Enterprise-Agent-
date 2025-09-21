"""Concurrency utilities for thread-safe operations."""
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ThreadSafeDict(dict):
    """Thread-safe dictionary implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            return super().__getitem__(key)

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            super().__delitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return super().get(key, default)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        with self._lock:
            return super().setdefault(key, default)

    def update(self, *args, **kwargs) -> None:
        with self._lock:
            super().update(*args, **kwargs)


class ExecutionManager:
    """Manages parallel execution with proper resource management."""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._active_tasks: ThreadSafeDict = ThreadSafeDict()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def start(self) -> None:
        """Start the executor."""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
                logger.debug(f"Started executor with {self.max_workers} workers")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        with self._lock:
            if self._executor:
                self._executor.shutdown(wait=wait)
                self._executor = None
                self._active_tasks.clear()
                logger.debug("Executor shutdown complete")

    def submit(
        self, fn: Callable, *args, task_id: Optional[str] = None, **kwargs
    ) -> Any:
        """Submit a task for execution."""
        if not callable(fn):
            raise ValueError("First argument must be callable")

        with self._lock:
            if self._executor is None:
                self.start()

            # Ensure executor is available
            if self._executor is None:
                raise RuntimeError("Failed to start executor")

            try:
                future = self._executor.submit(fn, *args, **kwargs)

                if task_id:
                    self._active_tasks[task_id] = future

                return future
            except Exception as exc:
                logger.error(f"Failed to submit task: {exc}")
                raise

    def map_parallel(
        self, fn: Callable, items: List[Any], timeout: Optional[float] = None
    ) -> List[Any]:
        """Execute function on items in parallel and return results in order."""
        if not items:
            return []

        with self._lock:
            if self._executor is None:
                self.start()

            # Ensure executor is available
            if self._executor is None:
                raise RuntimeError("Failed to start executor")

        futures = []
        try:
            # Submit all tasks first
            futures = [self._executor.submit(fn, item) for item in items]
            results = []

            # Wait for results with proper error handling
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except Exception as exc:
                    # Cancel remaining futures on error
                    for j, remaining_future in enumerate(futures):
                        if j > i and not remaining_future.done():
                            remaining_future.cancel()
                    logger.error(f"Error in parallel execution for item {i}: {exc}")
                    raise RuntimeError(
                        f"Parallel execution failed on item {i}: {exc}"
                    ) from exc

            return results

        except Exception as exc:
            # Ensure all futures are cancelled on any error
            for future in futures:
                if not future.done():
                    future.cancel()
            logger.error(f"Failed to execute parallel tasks: {exc}")
            raise

    def wait_for_tasks(
        self, task_ids: Optional[List[str]] = None, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Wait for specific tasks or all tasks to complete."""
        if task_ids:
            futures = {
                tid: self._active_tasks.get(tid)
                for tid in task_ids
                if tid in self._active_tasks
            }
        else:
            futures = dict(self._active_tasks)

        results = {}
        for task_id, future in futures.items():
            if future:
                try:
                    results[task_id] = future.result(timeout=timeout)
                except Exception as exc:
                    results[task_id] = {"error": str(exc)}
                    logger.error(f"Task {task_id} failed: {exc}")

        return results


@contextmanager
def synchronized_state(lock: threading.Lock = None):
    """Context manager for synchronized state access."""
    if lock is None:
        lock = threading.Lock()

    acquired = False
    try:
        acquired = lock.acquire(timeout=30)  # 30 second timeout to prevent deadlocks
        if not acquired:
            raise RuntimeError("Failed to acquire lock within timeout")
        yield
    finally:
        if acquired:
            lock.release()


class ThreadSafeMeta(type):
    """Metaclass to automatically create instance-specific locks."""

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._instance_lock = threading.RLock()
        return instance


def thread_safe(lock_attr: str = None):
    """Decorator to make a method thread-safe.

    Args:
        lock_attr: Name of the lock attribute on the instance (default: '_instance_lock')
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Use instance lock or create a new one
            if lock_attr:
                lock = getattr(self, lock_attr, None)
                if lock is None:
                    raise AttributeError(
                        f"Lock attribute '{lock_attr}' not found on instance"
                    )
            else:
                # Try to get instance lock, create if doesn't exist
                if not hasattr(self, "_instance_lock"):
                    self._instance_lock = threading.RLock()
                lock = self._instance_lock

            with lock:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


class AsyncExecutor:
    """Manages async execution for model calls."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Ensure event loop exists."""
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                try:
                    self._loop = asyncio.get_running_loop()
                except RuntimeError:
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
            return self._loop

    async def run_async(self, coro):
        """Run a coroutine."""
        return await coro

    def run_sync(self, coro):
        """Run a coroutine synchronously."""
        loop = self._ensure_loop()

        if loop.is_running():
            # If loop is already running, schedule the coroutine
            future = asyncio.ensure_future(coro, loop=loop)
            return future
        else:
            # If loop is not running, run until complete
            return loop.run_until_complete(coro)

    async def gather_async(self, *coros, return_exceptions: bool = False):
        """Gather multiple coroutines."""
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)

    def gather_sync(self, *coros, return_exceptions: bool = False):
        """Gather multiple coroutines synchronously."""
        loop = self._ensure_loop()

        if loop.is_running():
            future = asyncio.gather(*coros, return_exceptions=return_exceptions)
            return future
        else:
            return loop.run_until_complete(
                asyncio.gather(*coros, return_exceptions=return_exceptions)
            )


# Global execution manager instance
_global_executor = None
_executor_lock = threading.Lock()


def get_executor(max_workers: int = 4) -> ExecutionManager:
    """Get or create global executor instance."""
    global _global_executor

    with _executor_lock:
        if _global_executor is None:
            _global_executor = ExecutionManager(max_workers=max_workers)
            _global_executor.start()
        return _global_executor


def shutdown_executor() -> None:
    """Shutdown global executor."""
    global _global_executor

    with _executor_lock:
        if _global_executor:
            _global_executor.shutdown()
            _global_executor = None


__all__ = [
    "ThreadSafeDict",
    "ExecutionManager",
    "synchronized_state",
    "thread_safe",
    "AsyncExecutor",
    "get_executor",
    "shutdown_executor",
]
