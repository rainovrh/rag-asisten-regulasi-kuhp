"""Input validation utilities."""

import re
from pathlib import Path

from src.utils.exceptions import ConfigurationError


def validate_file_path(path: Path | str, must_exist: bool = True) -> Path:
    """Validate file path.
    
    Args:
        path: File path to validate.
        must_exist: If True, raises error if file doesn't exist.
    
    Returns:
        Validated Path object.
    
    Raises:
        ConfigurationError: If path is invalid.
    """
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise ConfigurationError(f"File not found: {path_obj}")
    
    return path_obj


def validate_query(query: str) -> str:
    """Validate user query.
    
    Args:
        query: User input query.
    
    Returns:
        Cleaned query string.
    
    Raises:
        ValueError: If query is empty or too long.
    """
    query = query.strip()
    
    if not query:
        raise ValueError("Query cannot be empty")
    
    if len(query) > 1000:
        raise ValueError("Query too long (max 1000 characters)")
    
    return query


def validate_top_k(top_k: int, max_value: int = 100) -> int:
    """Validate top_k parameter.
    
    Args:
        top_k: Number of results to retrieve.
        max_value: Maximum allowed value.
    
    Returns:
        Validated top_k value.
    
    Raises:
        ValueError: If top_k is out of range.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    
    if top_k > max_value:
        raise ValueError(f"top_k cannot exceed {max_value}")
    
    return top_k


def validate_threshold(threshold: float) -> float:
    """Validate similarity threshold.
    
    Args:
        threshold: Similarity threshold value.
    
    Returns:
        Validated threshold value.
    
    Raises:
        ValueError: If threshold is out of range.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")
    
    return threshold
