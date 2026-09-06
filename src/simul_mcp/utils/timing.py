"""
Timing utilities for Isaac Sim MCP Server.

This module provides timing, performance measurement, rate limiting,
and timeout utilities for asynchronous operations.
"""

import time
import asyncio
import inspect
import functools
from typing import Any, Callable, Optional, Dict, cast
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


def measure_time(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to measure function execution time.

    Args:
        func: Function to decorate (when used without parentheses)
        name: Custom name for the timer (defaults to function name)

    Usage:
        @measure_time
        def my_function():
            pass

        @measure_time(name="Custom Timer")
        def my_function():
            pass
    """

    def decorator(f: Callable) -> Callable:
        timer_name = name or f.__name__

        if inspect.iscoroutinefunction(f):

            @functools.wraps(f)
            async def async_wrapper(*args, **kwargs):
                with Timer(timer_name):
                    return await f(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(f)
            def sync_wrapper(*args, **kwargs):
                with Timer(timer_name):
                    return f(*args, **kwargs)

            return sync_wrapper

    if func is None:
        # Called with parentheses: @measure_time(name="...")
        return decorator
    else:
        # Called without parentheses: @measure_time
        return decorator(func)


async def timeout_after(seconds: float, coro):
    """
    Run a coroutine with a timeout.

    Args:
        seconds: Timeout in seconds
        coro: Coroutine to run

    Returns:
        Coroutine result

    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Operation timed out after {seconds}s")
        raise


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
        self.last_update = time.time()

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
        now = time.time()
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


def rate_limiter(rate: float, burst: int = 1):
    """
    Decorator to apply rate limiting to a function.

    Args:
        rate: Requests per second
        burst: Maximum burst size
    """
    limiter = RateLimiter(rate, burst)

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                await limiter.wait_for_token()
                return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                while not limiter.acquire():
                    time.sleep(0.01)  # Small sleep to avoid busy waiting
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


class Debouncer:
    """
    Debouncer to prevent rapid successive calls to a function.
    """

    def __init__(self, delay: float):
        """
        Initialize debouncer.

        Args:
            delay: Delay in seconds before function is called
        """
        self.delay = delay
        self.last_call_time = 0
        self.timer_handle: Optional[asyncio.Handle] = None

    def __call__(self, func: Callable) -> Callable:
        """Make the debouncer callable as a decorator."""
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.debounce_async(func, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.debounce_sync(func, *args, **kwargs)

            return sync_wrapper

    def debounce_sync(self, func: Callable, *args, **kwargs):
        """Debounce a synchronous function."""
        current_time = time.time()

        if current_time - self.last_call_time >= self.delay:
            self.last_call_time = current_time
            return func(*args, **kwargs)
        else:
            logger.debug(f"Debounced call to {func.__name__}")
            return None

    async def debounce_async(self, func: Callable, *args, **kwargs):
        """Debounce an asynchronous function."""
        # Cancel previous timer if it exists
        if self.timer_handle:
            self.timer_handle.cancel()

        # Create new timer
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def call_func():
            if not future.cancelled():
                try:
                    result = func(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        # Schedule the coroutine
                        task = loop.create_task(result)

                        def _complete_task(task_result: asyncio.Task) -> None:
                            exception: BaseException | None = task_result.exception()
                            if exception is None:
                                future.set_result(task_result.result())
                            else:
                                future.set_exception(cast(BaseException, exception))

                        task.add_done_callback(_complete_task)
                    else:
                        future.set_result(result)
                except Exception as e:
                    future.set_exception(e)

        self.timer_handle = loop.call_later(self.delay, call_func)

        try:
            return await future
        except asyncio.CancelledError:
            logger.debug(f"Debounced async call to {func.__name__} was cancelled")
            return None


def debounce(delay: float):
    """
    Decorator to debounce function calls.

    Args:
        delay: Delay in seconds before function is called
    """
    return Debouncer(delay)


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
