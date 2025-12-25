"""Centralized logging configuration for the neuro_foundation package.

Provides consistent logging across all modules with file and console output,
proper formatting, and configurable log levels.

Usage:
    from olfactory_modeling.utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.debug("Detailed debug information")
    logger.warning("Something unusual happened")
    logger.error("An error occurred", exc_info=True)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Global registry to avoid duplicate handlers
_CONFIGURED_LOGGERS = set()


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    console: bool = True,
    format_string: Optional[str] = None,
) -> None:
    """Configure root logger with file and/or console handlers.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path. If None and log_dir provided, auto-generates timestamped filename
        log_dir: Directory for log files. Creates if doesn't exist
        console: Whether to log to console (stdout/stderr)
        format_string: Custom format string. If None, uses default detailed format
        
    Example:
        >>> # Basic setup - console only
        >>> setup_logging(log_level="INFO")
        
        >>> # Production setup - console + file
        >>> setup_logging(
        ...     log_level="INFO",
        ...     log_dir=Path("logs"),
        ...     console=True
        ... )
        
        >>> # Debug mode - verbose console + file
        >>> setup_logging(
        ...     log_level="DEBUG",
        ...     log_file=Path("logs/debug_session.log"),
        ...     console=True
        ... )
    """
    # Clear existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    # Default format: timestamp - logger name - level - message
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        if log_file is None:
            # Auto-generate timestamped filename
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"olfactory_modeling_{timestamp}.log"
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_file}")


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance for a module.
    
    Args:
        name: Logger name (typically __name__ of the module)
        level: Optional override for log level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
        
        >>> debug_logger = get_logger(__name__, level="DEBUG")
        >>> debug_logger.debug("Detailed information")
    """
    logger = logging.getLogger(name)
    
    # Set level if specified
    if level:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
    
    # If root logger has no handlers, set up basic console logging
    if not logging.getLogger().handlers:
        setup_logging(log_level="INFO", console=True)
    
    return logger


def log_function_call(logger: logging.Logger, level: str = "DEBUG"):
    """Decorator to log function calls with arguments and return values.
    
    Useful for debugging complex pipelines and tracking execution flow.
    
    Args:
        logger: Logger instance to use
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        
    Example:
        >>> logger = get_logger(__name__)
        >>> 
        >>> @log_function_call(logger, level="DEBUG")
        ... def process_data(x, y, normalize=True):
        ...     return x + y
        >>> 
        >>> result = process_data(1, 2, normalize=False)
        # Logs: "Calling process_data(x=1, y=2, normalize=False)"
        # Logs: "process_data returned 3"
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            args_repr = ", ".join(repr(a) for a in args)
            kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            all_args = ", ".join(filter(None, [args_repr, kwargs_repr]))
            
            log_method = getattr(logger, level.lower())
            log_method(f"Calling {func.__name__}({all_args})")
            
            try:
                result = func(*args, **kwargs)
                log_method(f"{func.__name__} returned {result!r}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


# Convenience function for quick setup in scripts
def quick_setup(verbose: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    """Quick logging setup for scripts and notebooks.
    
    Args:
        verbose: If True, use DEBUG level. Otherwise INFO
        log_file: Optional file to log to
        
    Returns:
        Root logger instance
        
    Example:
        >>> logger = quick_setup(verbose=True, log_file=Path("script.log"))
        >>> logger.info("Script started")
    """
    level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level=level, log_file=log_file, console=True)
    return logging.getLogger()
