"""Logging setup. Call setup_logging() once at the top of each script."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure root logger with sensible defaults.

    Parameters
    ----------
    level : log level name (DEBUG, INFO, WARNING, ERROR).
    log_file : optional file path to also write logs to.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    for noisy in ("urllib3", "fsspec", "asyncio", "h5py"):
        logging.getLogger(noisy).setLevel(logging.WARNING)