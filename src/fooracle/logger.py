import logging
import os

log_level = os.getenv("LOG_LEVEL", "INFO")

numeric_level = logging.getLevelName(log_level.upper())
if not isinstance(numeric_level, int):
    raise ValueError(f"Invalid log level: {log_level}")

logging.basicConfig(level=numeric_level)
LOGGER = logging.getLogger(__name__)
