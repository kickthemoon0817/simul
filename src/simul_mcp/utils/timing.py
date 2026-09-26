"""
Timing utilities for Isaac Sim MCP Server.

This module provides timing, performance measurement, rate limiting,
and timeout utilities for asynchronous operations.
"""

import time
import asyncio
import inspect
import functools
from typing import Any, Callable, Optional, Dict
from contextlib import contextmanager
from collections import defaultdict, deque

from ..logging import get_logger

logger = get_logger(__name__)


class Timer:
    """
    A simple timer class for measuring elapsed time.

    Can be used as a context manager or manually started/stopped.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or "Timer"
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed_time = None
        logger.debug(f"{self.name} started")

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")

        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        logger.debug(f"{self.name} completed in {self.elapsed_time:.3f}s")
        return self.elapsed_time

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    @property
    def is_running(self) -> bool:
        """Check if timer is currently running."""
        return self.start_time is not None and self.end_time is None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class RateLimiter:
    """
    Token bucket rate limiter for controlling request rates.
    """

    def __init__(self, rate: float, burst: int = 1):
        """
        Initialize rate limiter.

        Args:
            rate: Requests per second
            burst: Maximum burst size (token bucket capacity)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        # Monotonic, so a wall-clock step backwards cannot stall refills.
        self.last_update = time.monotonic()

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def seconds_until_available(self, tokens: int = 1) -> float:
        """
        Return how long a caller must wait before ``tokens`` can be acquired.

        Spends nothing, so a caller can consult several buckets and only draw
        from all of them once every one has room.

        Args:
            tokens: Number of tokens the caller wants

        Returns:
            0.0 when the tokens are available now, else the wait in seconds
        """
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        return (tokens - self.tokens) / self.rate

    def _refill(self) -> None:
        """Add the tokens earned since the last update, up to the burst size."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def wait_for_token(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available.

        Args:
            tokens: Number of tokens to wait for
        """
        while not self.acquire(tokens):
            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(min(wait_time, 0.1))  # Cap wait time


class PerformanceMonitor:
    """
    Monitor performance metrics for functions and operations.
    """

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "recent_times": deque(maxlen=100),  # Keep last 100 measurements
            }
        )

    def record(self, name: str, duration: float) -> None:
        """
        Record a performance measurement.

        Args:
            name: Operation name
            duration: Duration in seconds
        """
        metrics = self.metrics[name]
        metrics["count"] += 1
        metrics["total_time"] += duration
        metrics["min_time"] = min(metrics["min_time"], duration)
        metrics["max_time"] = max(metrics["max_time"], duration)
        metrics["recent_times"].append(duration)

    def get_stats(self, name: str) -> Dict[str, Any]:
        """
        Get performance statistics for an operation.

        Args:
            name: Operation name

        Returns:
            Dictionary with performance statistics
        """
        if name not in self.metrics:
            return {}

        metrics = self.metrics[name]
        recent_times = list(metrics["recent_times"])

        stats = {
            "count": metrics["count"],
            "total_time": metrics["total_time"],
            "average_time": metrics["total_time"] / metrics["count"]
            if metrics["count"] > 0
            else 0,
            "min_time": metrics["min_time"]
            if metrics["min_time"] != float("inf")
            else 0,
            "max_time": metrics["max_time"],
        }

        if recent_times:
            stats["recent_average"] = sum(recent_times) / len(recent_times)
            stats["recent_min"] = min(recent_times)
            stats["recent_max"] = max(recent_times)

        return stats

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all monitored operations."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure operation duration."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record(name, duration)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(name: Optional[str] = None):
    """
    Decorator to monitor function performance.

    Args:
        name: Custom name for the operation (defaults to function name)
    """

    def decorator(func: Callable) -> Callable:
        operation_name = name or f"{func.__module__}.{func.__name__}"

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with performance_monitor.measure(operation_name):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with performance_monitor.measure(operation_name):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
