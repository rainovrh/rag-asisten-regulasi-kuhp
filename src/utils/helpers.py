"""Helper utilities."""

import time
from functools import wraps
from typing import Any, Callable

from src.utils.logger import get_logger

logger = get_logger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time.
    
    Example:
        >>> @timer
        ... def slow_function():
        ...     time.sleep(1)
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} executed in {elapsed:.2f}ms")
        return result
    
    return wrapper


def batch_process(items: list[Any], batch_size: int = 10) -> list[list[Any]]:
    """Split items into batches.
    
    Args:
        items: List of items to batch.
        batch_size: Maximum items per batch.
    
    Returns:
        List of batches.
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Input text.
        max_length: Maximum character length.
        suffix: Suffix to append if truncated.
    
    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
