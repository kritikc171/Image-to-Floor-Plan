"""Logging configuration: structured, request-safe (no secrets)."""

from __future__ import annotations

import logging
import sys


class RequestIdFilter(logging.Filter):
    """Injects a request id attribute so every record formats safely."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the application logger ``floorplan``."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s")
    )
    handler.addFilter(RequestIdFilter())
    logger = logging.getLogger("floorplan")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


logger = configure_logging()
