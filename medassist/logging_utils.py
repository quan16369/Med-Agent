"""
Logging Setup
Configurable logging for development, testing, and production
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
):
    """
    Setup application logging
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for console only)
        format_string: Custom log format
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for a specific module"""
    return logging.getLogger(name)


class LogContext:
    """Context manager for structured logging"""
    
    def __init__(self, logger: logging.Logger, context: str, **kwargs):
        self.logger = logger
        self.context = context
        self.metadata = kwargs
    
    def __enter__(self):
        metadata_str = " ".join(f"{k}={v}" for k, v in self.metadata.items())
        self.logger.info(f"Starting {self.context} {metadata_str}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"Failed {self.context}: {exc_val}")
        else:
            self.logger.info(f"Completed {self.context}")
        return False
