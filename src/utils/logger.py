"""
Structured logging utility with file rotation, multiple log levels, and performance logging.

This module provides a centralized logging system with:
- JSON-formatted structured logs
- File rotation (both size-based and time-based)
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Performance monitoring decorators
- Context managers for operation tracking
"""

import logging
import logging.handlers
import json
import time
import functools
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: LogRecord instance

        Returns:
            JSON-formatted log string
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add custom fields from extra parameter
        if hasattr(record, 'custom_fields'):
            log_data.update(record.custom_fields)

        return json.dumps(log_data)


class PerformanceFormatter(logging.Formatter):
    """Custom formatter for performance logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format performance log record."""
        if hasattr(record, 'duration'):
            return f"{record.levelname} - {record.getMessage()} - Duration: {record.duration:.4f}s"
        return super().format(record)


class Logger:
    """
    Enhanced logger with structured logging, file rotation, and performance tracking.
    """

    _instances: Dict[str, logging.Logger] = {}

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        json_format: bool = True,
        console_output: bool = True,
        file_output: bool = True,
    ):
        """
        Initialize logger with specified configuration.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
            json_format: Use JSON formatting for structured logs
            console_output: Enable console output
            file_output: Enable file output
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.json_format = json_format
        self.console_output = console_output
        self.file_output = file_output

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger with handlers and formatters.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        logger.propagate = False

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            if self.json_format:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler with rotation (size-based)
        if self.file_output:
            log_file = self.log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.level)
            if self.json_format:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            # Time-based rotating file handler (daily)
            timed_log_file = self.log_dir / f"{self.name}_daily.log"
            timed_handler = logging.handlers.TimedRotatingFileHandler(
                timed_log_file,
                when='midnight',
                interval=1,
                backupCount=30  # Keep 30 days of logs
            )
            timed_handler.setLevel(self.level)
            if self.json_format:
                timed_handler.setFormatter(JSONFormatter())
            else:
                timed_handler.setFormatter(file_formatter)
            logger.addHandler(timed_handler)

        # Performance logger
        perf_log_file = self.log_dir / f"{self.name}_performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(PerformanceFormatter())

        # Add performance handler to a separate logger
        perf_logger = logging.getLogger(f"{self.name}.performance")
        perf_logger.setLevel(logging.INFO)
        perf_logger.propagate = False
        perf_logger.handlers.clear()
        perf_logger.addHandler(perf_handler)

        return logger

    @classmethod
    def get_logger(cls, name: str, **kwargs) -> 'Logger':
        """
        Get or create a logger instance (singleton pattern per name).

        Args:
            name: Logger name
            **kwargs: Additional configuration parameters

        Returns:
            Logger instance
        """
        if name not in cls._instances:
            cls._instances[name] = cls(name, **kwargs)
        return cls._instances[name]

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra={'custom_fields': kwargs})

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra={'custom_fields': kwargs})

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra={'custom_fields': kwargs})

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra={'custom_fields': kwargs})

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra={'custom_fields': kwargs})

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, extra={'custom_fields': kwargs})

    def log_performance(self, operation: str, duration: float, **kwargs):
        """
        Log performance metrics.

        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional metrics
        """
        perf_logger = logging.getLogger(f"{self.name}.performance")
        metrics = {'operation': operation, 'duration': duration}
        metrics.update(kwargs)

        extra = {'custom_fields': metrics, 'duration': duration}
        perf_logger.info(f"Performance: {operation}", extra=extra)

    @contextmanager
    def log_context(self, operation: str, level: int = logging.INFO, **kwargs):
        """
        Context manager for logging operation start/end with timing.

        Args:
            operation: Operation name
            level: Log level
            **kwargs: Additional context fields

        Yields:
            Dictionary to add dynamic fields during operation
        """
        start_time = time.time()
        context_data = {}

        self.logger.log(
            level,
            f"Starting: {operation}",
            extra={'custom_fields': kwargs}
        )

        try:
            yield context_data
            duration = time.time() - start_time

            log_data = {'duration': duration, 'status': 'success'}
            log_data.update(kwargs)
            log_data.update(context_data)

            self.logger.log(
                level,
                f"Completed: {operation}",
                extra={'custom_fields': log_data}
            )
            self.log_performance(operation, duration, **context_data)

        except Exception as e:
            duration = time.time() - start_time

            log_data = {
                'duration': duration,
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            }
            log_data.update(kwargs)
            log_data.update(context_data)

            self.logger.error(
                f"Failed: {operation}",
                extra={'custom_fields': log_data},
                exc_info=True
            )
            self.log_performance(operation, duration, status='failed', **context_data)
            raise


def log_performance(logger_name: str = None):
    """
    Decorator to log function execution time and performance.

    Args:
        logger_name: Name of logger to use (defaults to function's module)

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            log_name = logger_name or func.__module__
            logger = Logger.get_logger(log_name)

            # Log function call
            func_name = f"{func.__module__}.{func.__qualname__}"
            logger.debug(f"Calling function: {func_name}", args=str(args), kwargs=str(kwargs))

            # Execute and time function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.log_performance(
                    func_name,
                    duration,
                    status='success'
                )
                logger.debug(f"Function completed: {func_name}", duration=duration)

                return result

            except Exception as e:
                duration = time.time() - start_time

                logger.log_performance(
                    func_name,
                    duration,
                    status='failed',
                    error=str(e),
                    error_type=type(e).__name__
                )
                logger.exception(f"Function failed: {func_name}", duration=duration)
                raise

        return wrapper
    return decorator


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    **kwargs
) -> Logger:
    """
    Convenience function to get a logger instance.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        **kwargs: Additional configuration parameters

    Returns:
        Logger instance
    """
    return Logger.get_logger(name, log_dir=log_dir, level=level, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = get_logger("example", level=logging.DEBUG)

    # Basic logging
    logger.debug("This is a debug message")
    logger.info("Application started", version="1.0.0", environment="production")
    logger.warning("This is a warning", threshold=0.95, current_value=0.97)
    logger.error("An error occurred", error_code=500)

    # Context manager for operations
    with logger.log_context("data_processing", data_size=1000):
        time.sleep(0.1)  # Simulate work
        logger.info("Processing data...")

    # Performance decorator
    @log_performance("example")
    def slow_function(n: int):
        """Example function to demonstrate performance logging."""
        time.sleep(0.05)
        return sum(range(n))

    result = slow_function(1000)
    logger.info("Function result", result=result)

    # Exception logging
    try:
        raise ValueError("Example exception")
    except ValueError:
        logger.exception("Caught an exception", context="example")

    print(f"\nLog files created in: {logger.log_dir}")
