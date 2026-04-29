"""
Logging configuration for the ExecutionCoordinator system.
Provides hierarchical logging following Python logging best practices.

This module uses standard Python logging practices:
- Hierarchical loggers using dot notation (parent.child)
- Module-level configuration functions
- Proper logger inheritance and propagation
- Standard logging levels and formatting
"""

import logging
import sys
from typing import Dict, Optional

# Logger hierarchy - follows Python module naming conventions
LOGGERS = {
    "main": "execution_coordinator",
    "seed": "execution_coordinator.seed",
    "parallel": "execution_coordinator.parallel",
    "record": "execution_coordinator.record",
}


def setup_execution_logging(
    levels: Optional[Dict[str, int]] = None,
    format_string: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
    disable_existing: bool = True,
) -> None:
    """
    Configure logging for the execution coordinator system.

    This follows Python logging best practices by:
    - Using hierarchical logger names
    - Allowing logger inheritance through propagation
    - Using a single handler to avoid duplication

    Args:
        levels: Dictionary mapping logger type to level (e.g. {'main': logging.INFO})
        format_string: Log message format (None for default)
        handler: Custom handler (None for default console handler)
        disable_existing: Whether to clear existing root logger handlers
    """
    if levels is None:
        levels = {
            "main": logging.INFO,
            "seed": logging.INFO,
            "parallel": logging.INFO,
            "record": logging.WARNING,
        }

    if format_string is None:
        format_string = "[%(levelname)s] %(name)s: %(message)s"

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_string))

    # Clean up existing configuration if requested
    if disable_existing:
        # Clear root logger to prevent duplicate messages
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    # Configure parent logger first
    parent_logger = logging.getLogger(LOGGERS["main"])
    parent_logger.setLevel(levels.get("main", logging.INFO))
    parent_logger.handlers.clear()
    parent_logger.addHandler(handler)
    parent_logger.propagate = False

    # Configure child loggers - they inherit from parent but can have different levels
    for logger_type in ["seed", "parallel", "record"]:
        child_logger = logging.getLogger(LOGGERS[logger_type])
        child_logger.setLevel(levels.get(logger_type, logging.INFO))
        # Child loggers don't need their own handlers - they inherit from parent
        # But we set propagate=False to avoid duplicate messages
        child_logger.propagate = False
        child_logger.handlers.clear()
        child_logger.addHandler(handler)


def get_logger(logger_type: str = "main") -> logging.Logger:
    """
    Get a logger by type. This is the Pythonic way to get loggers.

    Args:
        logger_type: Type of logger ('main', 'seed', 'parallel', 'record')

    Returns:
        Logger instance

    Raises:
        ValueError: If logger_type is not valid
    """
    if logger_type not in LOGGERS:
        raise ValueError(
            f"Unknown logger type: {logger_type}. Valid types: {list(LOGGERS.keys())}"
        )

    return logging.getLogger(LOGGERS[logger_type])


# Convenience functions - more Pythonic than class methods
def get_main_logger() -> logging.Logger:
    """Get the main execution coordinator logger."""
    return get_logger("main")


def get_seed_logger() -> logging.Logger:
    """Get the seed-specific logger."""
    return get_logger("seed")


def get_parallel_logger() -> logging.Logger:
    """Get the parallel execution logger."""
    return get_logger("parallel")


def get_record_logger() -> logging.Logger:
    """Get the record/save operations logger."""
    return get_logger("record")


def set_logger_level(logger_type: str, level: int) -> None:
    """
    Set the level for a specific logger type.

    Args:
        logger_type: Type of logger to modify
        level: New logging level (e.g. logging.DEBUG, logging.INFO)
    """
    logger = get_logger(logger_type)
    logger.setLevel(level)


def disable_logger(logger_type: str) -> None:
    """Disable a specific logger by setting level to CRITICAL."""
    set_logger_level(logger_type, logging.CRITICAL)


def enable_logger(logger_type: str, level: int = logging.INFO) -> None:
    """Enable a specific logger by setting it to the specified level."""
    set_logger_level(logger_type, level)


# Preset configurations for common use cases
class LoggingPresets:
    """Common logging configurations following Python conventions."""

    @staticmethod
    def development() -> None:
        """Development: All loggers enabled at INFO level."""
        setup_execution_logging(
            {
                "main": logging.INFO,
                "seed": logging.INFO,
                "parallel": logging.INFO,
                "record": logging.INFO,
            }
        )

    @staticmethod
    def production() -> None:
        """Production: Minimal logging to reduce noise."""
        setup_execution_logging(
            {
                "main": logging.INFO,
                "seed": logging.WARNING,
                "parallel": logging.WARNING,
                "record": logging.ERROR,
            }
        )

    @staticmethod
    def quiet() -> None:
        """Quiet: Only warnings and errors."""
        setup_execution_logging(
            {
                "main": logging.WARNING,
                "seed": logging.CRITICAL,
                "parallel": logging.CRITICAL,
                "record": logging.ERROR,
            }
        )

    @staticmethod
    def debug() -> None:
        """Debug: Verbose logging with function names and line numbers."""
        setup_execution_logging(
            levels={
                "main": logging.DEBUG,
                "seed": logging.DEBUG,
                "parallel": logging.DEBUG,
                "record": logging.DEBUG,
            },
            format_string="[%(levelname)s] %(name)s (%(funcName)s:%(lineno)d): %(message)s",
        )


# For backward compatibility and convenience
def configure_execution_logging(preset: str = "development") -> None:
    """
    Configure logging using a preset name.

    Args:
        preset: One of 'development', 'production', 'quiet', 'debug'
    """
    preset_methods = {
        "development": LoggingPresets.development,
        "production": LoggingPresets.production,
        "quiet": LoggingPresets.quiet,
        "debug": LoggingPresets.debug,
    }

    if preset not in preset_methods:
        raise ValueError(
            f"Unknown preset: {preset}. Valid presets: {list(preset_methods.keys())}"
        )

    preset_methods[preset]()


class InPlaceLogger(logging.Logger):
    """Custom logger that immediately uses in-place logging handler for same line logging."""

    def __init__(self, name: str):
        super().__init__(name, level=logging.INFO)
        self.addHandler(LogInPlaceHandler())
        self.propagate = False


class LogInPlaceHandler(logging.StreamHandler):
    """Custom handler to overwrite the same terminal line for progress updates."""

    def __init__(self):
        super().__init__(stream=sys.stdout)
        formatter = logging.Formatter("%(levelname)s:%(message)s")
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        sys.stdout.write(f"{msg}    \r")  # Overwrite the line with some padding
        sys.stdout.flush()
