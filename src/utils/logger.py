"""Logging configuration for the application."""

from pathlib import Path
from loguru import logger
import sys

from src.config.settings import settings


def setup_logger(log_level: str = "INFO") -> None:
    """Configure application logger with console and file handlers.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
    )
    
    # File handler
    log_path = settings.log_dir / "app_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_path),
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        level="DEBUG",
    )
    
    logger.info(f"Logger initialized at level {log_level}")


def get_logger(name: str):
    """Get a logger instance with the given name.
    
    Args:
        name: Module name for the logger.
    
    Returns:
        Logger instance.
    """
    return logger.bind(name=name)
