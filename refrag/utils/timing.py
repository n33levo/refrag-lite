"""Timing utilities for measuring TTFT and throughput."""

import time
from contextlib import contextmanager
from typing import Optional

import torch


class Timer:
    """Simple timer for measuring elapsed time.

    Example:
        >>> timer = Timer()
        >>> time.sleep(0.1)
        >>> elapsed = timer.elapsed()
        >>> elapsed >= 0.1
        True
    """

    def __init__(self):
        """Initialize timer."""
        self.start_time = time.perf_counter()
        self.end_time: Optional[float] = None

    def reset(self) -> None:
        """Reset timer."""
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self) -> float:
        """Stop timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        self.end_time = time.perf_counter()
        return self.elapsed()

    def elapsed(self) -> float:
        """Get elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time


@contextmanager
def measure_ttft(device: str = "cuda"):
    """Context manager to measure Time-to-First-Token.

    Args:
        device: Device to synchronize ("cuda", "mps", or "cpu")

    Yields:
        Timer object

    Example:
        >>> with measure_ttft() as timer:
        ...     # Generate first token
        ...     pass
        >>> ttft = timer.elapsed()
    """
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

    timer = Timer()

    try:
        yield timer
    finally:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        timer.stop()


def measure_throughput(
    num_tokens: int,
    elapsed_time: float,
) -> float:
    """Compute throughput in tokens per second.

    Args:
        num_tokens: Number of tokens generated
        elapsed_time: Elapsed time in seconds

    Returns:
        Throughput in tokens/second

    Example:
        >>> throughput = measure_throughput(100, 2.0)
        >>> throughput
        50.0
    """
    if elapsed_time <= 0:
        return 0.0
    return num_tokens / elapsed_time


def get_vram_usage() -> dict:
    """Get current VRAM usage.

    Returns:
        Dictionary with VRAM stats in MB

    Example:
        >>> vram = get_vram_usage()
        >>> "allocated_mb" in vram
        True
    """
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
        }

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "peak_allocated_mb": peak,
    }


def reset_peak_memory() -> None:
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class SpeedBenchmark:
    """Benchmark for measuring generation speed.

    Example:
        >>> benchmark = SpeedBenchmark(device="cuda")
        >>> benchmark.start()
        >>> # Generate tokens
        >>> benchmark.record_token()  # First token
        >>> benchmark.record_token()  # Second token
        >>> stats = benchmark.get_stats()
        >>> "ttft" in stats
        True
    """

    def __init__(self, device: str = "cuda"):
        """Initialize benchmark.

        Args:
            device: Device to synchronize
        """
        self.device = device
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.num_tokens = 0

    def start(self) -> None:
        """Start benchmark."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            reset_peak_memory()
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()

        self.start_time = time.perf_counter()
        self.first_token_time = None
        self.end_time = None
        self.num_tokens = 0

    def record_token(self) -> None:
        """Record a generated token."""
        if self.first_token_time is None:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.synchronize()
            self.first_token_time = time.perf_counter()

        self.num_tokens += 1

    def stop(self) -> None:
        """Stop benchmark."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.end_time = time.perf_counter()

    def get_stats(self) -> dict:
        """Get benchmark statistics.

        Returns:
            Dictionary with timing stats

        Example:
            >>> benchmark = SpeedBenchmark()
            >>> benchmark.start()
            >>> benchmark.record_token()
            >>> benchmark.stop()
            >>> stats = benchmark.get_stats()
        """
        if self.start_time is None:
            return {}

        stats = {}

        # Time to first token
        if self.first_token_time is not None:
            stats["ttft"] = self.first_token_time - self.start_time

        # Total time
        end = self.end_time or time.perf_counter()
        stats["total_time"] = end - self.start_time

        # Throughput (excluding TTFT)
        if self.first_token_time is not None and self.num_tokens > 1:
            remaining_time = end - self.first_token_time
            remaining_tokens = self.num_tokens - 1
            if remaining_time > 0:
                stats["throughput"] = remaining_tokens / remaining_time

        # Number of tokens
        stats["num_tokens"] = self.num_tokens

        # VRAM
        stats.update(get_vram_usage())

        return stats
