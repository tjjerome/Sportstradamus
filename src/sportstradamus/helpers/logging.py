"""Structured JSON logging for CLI entry points.

Each CLI calls :func:`get_logger` once at startup. The returned
:class:`logging.Logger` writes one JSON record per line to
``logs/{YYYY-MM-DD}/{cli_name}.jsonl`` (rotating at 50 MB) and mirrors
``WARNING``/``ERROR`` records to ``stderr`` in plain text.

Stdlib ``logging`` only — no third-party logging libraries.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Rotation cap for the JSONL file handler. 50 MB matches the spec; the
# default 1 backup is enough for daily rotation since the date stamp in
# the path already partitions logs by day.
_MAX_BYTES = 50 * 1024 * 1024
_BACKUP_COUNT = 5

# Fields that ``logging.LogRecord`` always carries. Anything not in this
# set is treated as a caller-supplied ``extra=`` and serialized into the
# JSON record.
_RESERVED_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line.

    Emits the fixed fields ``ts``, ``level``, ``module``, ``message`` and
    folds any caller-supplied ``extra={...}`` keys into the same record.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serialize ``record`` as a JSON line."""
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="microseconds"),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _log_dir() -> Path:
    """Return ``logs/{YYYY-MM-DD}/`` rooted at the current working directory."""
    return Path("logs") / datetime.now().strftime("%Y-%m-%d")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given CLI name.

    The logger writes JSON lines to ``logs/{YYYY-MM-DD}/{name}.jsonl`` with
    rotation at 50 MB, and additionally writes ``WARNING``/``ERROR`` records
    in plain text to ``stderr``. Repeated calls with the same ``name``
    return the same logger without stacking handlers.

    Args:
        name: CLI identifier used as both the logger name and the log
            file basename (e.g. ``"confer"``, ``"meditate"``).

    Returns:
        A ``logging.Logger`` ready for use.
    """
    logger = logging.getLogger(f"sportstradamus.cli.{name}")
    if getattr(logger, "_sportstradamus_configured", False):
        return logger

    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_dir / f"{name}.jsonl",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger._sportstradamus_configured = True  # type: ignore[attr-defined]
    return logger
