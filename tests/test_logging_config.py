"""Tests for logging configuration infrastructure."""

import logging
import tempfile
from pathlib import Path

import pytest

from olfactory_modeling.utils.logging_config import (
    setup_logging,
    get_logger,
    quick_setup,
)


class TestLoggingSetup:
    """Test logging configuration functions."""
    
    def test_setup_logging_console_only(self):
        """Test basic console-only logging setup."""
        setup_logging(log_level="INFO", console=True)
        logger = logging.getLogger()
        assert logger.level == logging.INFO
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_level="DEBUG", log_file=log_file, console=False)
            
            logger = logging.getLogger()
            logger.info("Test message")
            
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content
    
    def test_setup_logging_auto_filename(self):
        """Test automatic log filename generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            setup_logging(log_level="INFO", log_dir=log_dir, console=False)
            
            log_files = list(log_dir.glob("neuro_foundation_*.log"))
            assert len(log_files) == 1
    
    def test_get_logger(self):
        """Test getting a configured logger."""
        setup_logging(log_level="INFO")
        logger = get_logger(__name__)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__
    
    def test_get_logger_with_level_override(self):
        """Test logger with custom level."""
        setup_logging(log_level="INFO")
        debug_logger = get_logger(__name__, level="DEBUG")
        
        assert debug_logger.level == logging.DEBUG
    
    def test_quick_setup_verbose(self):
        """Test quick setup in verbose mode."""
        logger = quick_setup(verbose=True)
        assert logger.level == logging.DEBUG
    
    def test_quick_setup_with_file(self):
        """Test quick setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "quick.log"
            logger = quick_setup(verbose=False, log_file=log_file)
            logger.info("Quick setup test")
            
            assert log_file.exists()
            assert "Quick setup test" in log_file.read_text()


class TestLogLevels:
    """Test that different log levels work correctly."""
    
    def test_debug_level_captures_all(self):
        """Test DEBUG level captures all messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "debug.log"
            setup_logging(log_level="DEBUG", log_file=log_file, console=False)
            logger = get_logger(__name__)
            
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            
            content = log_file.read_text()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content
    
    def test_info_level_filters_debug(self):
        """Test INFO level filters out DEBUG messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "info.log"
            setup_logging(log_level="INFO", log_file=log_file, console=False)
            logger = get_logger(__name__)
            
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            
            content = log_file.read_text()
            assert "Debug message" not in content
            assert "Info message" in content
            assert "Warning message" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
