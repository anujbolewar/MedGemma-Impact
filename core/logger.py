"""
core/logger.py — JSONL structured logger for NeuroWeave Sentinel.

NWSLogger writes one JSON object per line to logs/nws_{date}.jsonl,
rotating automatically each day. WARN/ERROR/CRITICAL are also mirrored
to Python stdlib logging (stderr). Thread-safe via threading.Lock.

Usage::

    from core.logger import get_logger
    log = get_logger()
    log.info("input", "gaze_frame", {"x": 0.5, "y": 0.4})
    log.perf("llm", "inference_done", latency_ms=1240.5, data={"tokens": 22})
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── stdlib mirror logger (stderr for WARN+) ──────────────────
_stdlib = logging.getLogger("nws")
if not _stdlib.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))
    _stdlib.addHandler(_handler)
_stdlib.setLevel(logging.DEBUG)
_stdlib.propagate = False

# ── Log directory (relative to project root) ─────────────────
_LOG_DIR = Path("logs")

# ── Singleton storage ─────────────────────────────────────────
_instance: Optional["NWSLogger"] = None
_instance_lock = threading.Lock()


class NWSLogger:
    """
    Singleton JSONL structured logger for NeuroWeave Sentinel.

    Each call to an log method appends a single JSON line to
    ``logs/nws_{YYYY-MM-DD}.jsonl``. A new file is opened automatically
    when the calendar date changes.

    Fields written per entry:

    .. code-block:: json

        {
          "timestamp_iso": "2026-02-25T01:20:49.123456+00:00",
          "level": "INFO",
          "phase": "llm",
          "event": "inference_done",
          "data": {"tokens": 22},
          "latency_ms": 1240.5
        }

    ``latency_ms`` is omitted when ``None``.

    Do not instantiate directly — use :func:`get_logger`.
    """

    def __init__(self) -> None:
        """Open the log file for today and write the startup entry."""
        self._lock = threading.Lock()
        self._file: Optional[Any] = None
        self._current_date: str = ""
        self._open_file()
        self._write_startup()

    # ──────────────────────────────────────────
    # Public logging methods
    # ──────────────────────────────────────────

    def info(self, phase: str, event: str, data: Optional[dict] = None) -> None:
        """
        Write an INFO-level structured log entry.

        Args:
            phase: System phase or subsystem (e.g. ``'llm'``, ``'gaze'``).
            event: Short event identifier (e.g. ``'frame_received'``).
            data: Optional dict of additional key-value context.
        """
        self._write("INFO", phase, event, data)

    def warn(self, phase: str, event: str, data: Optional[dict] = None) -> None:
        """
        Write a WARN-level entry and mirror to stderr via stdlib logging.

        Args:
            phase: System phase.
            event: Short event identifier.
            data: Optional context dict.
        """
        self._write("WARN", phase, event, data)
        _stdlib.warning("[%s] %s | %s", phase, event, data or {})

    def error(self, phase: str, event: str, data: Optional[dict] = None) -> None:
        """
        Write an ERROR-level entry and mirror to stderr via stdlib logging.

        Args:
            phase: System phase.
            event: Short event identifier.
            data: Optional context dict.
        """
        self._write("ERROR", phase, event, data)
        _stdlib.error("[%s] %s | %s", phase, event, data or {})

    def critical(self, phase: str, event: str, data: Optional[dict] = None) -> None:
        """
        Write a CRITICAL-level entry and mirror to stderr via stdlib logging.

        Args:
            phase: System phase.
            event: Short event identifier.
            data: Optional context dict.
        """
        self._write("CRITICAL", phase, event, data)
        _stdlib.critical("[%s] %s | %s", phase, event, data or {})

    def perf(
        self,
        phase: str,
        event: str,
        latency_ms: float,
        data: Optional[dict] = None,
    ) -> None:
        """
        Write a PERF-level entry for latency tracking.

        Args:
            phase: Subsystem the measurement belongs to (e.g. ``'llm'``).
            event: What was measured (e.g. ``'inference_done'``).
            latency_ms: Measured latency in milliseconds.
            data: Optional additional context dict.
        """
        self._write("PERF", phase, event, data, latency_ms=latency_ms)

    def flush(self) -> None:
        """
        Flush the underlying file buffer immediately.

        Call this before process exit or after critical events to ensure
        no log entries are lost in the OS buffer.
        """
        with self._lock:
            if self._file and not self._file.closed:
                self._file.flush()

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _write(
        self,
        level: str,
        phase: str,
        event: str,
        data: Optional[dict],
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Serialise and append one JSON line to the log file.

        Performs daily rotation check on every write (cheap date compare).
        Thread-safe via ``self._lock``.

        Args:
            level: Log level string.
            phase: Subsystem phase.
            event: Event identifier.
            data: Context dict (may be None).
            latency_ms: Optional latency value.
        """
        now = datetime.now(tz=timezone.utc)
        record: dict[str, Any] = {
            "timestamp_iso": now.isoformat(),
            "level": level,
            "phase": phase,
            "event": event,
            "data": data or {},
        }
        if latency_ms is not None:
            record["latency_ms"] = round(latency_ms, 3)

        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            self._rotate_if_needed(now)
            if self._file and not self._file.closed:
                self._file.write(line + "\n")
                self._file.flush()

    def _rotate_if_needed(self, now: datetime) -> None:
        """
        Open a new log file if the calendar date has changed.

        Called inside ``self._lock`` — do not call from outside.

        Args:
            now: Current UTC datetime.
        """
        today = now.strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._file and not self._file.closed:
                self._file.close()
            self._current_date = today
            log_path = _LOG_DIR / f"nws_{today}.jsonl"
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._file = open(log_path, "a", encoding="utf-8", buffering=1)  # noqa: WPS515

    def _open_file(self) -> None:
        """Open the log file for today's date (called once on init)."""
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            self._rotate_if_needed(now)

    def _write_startup(self) -> None:
        """Write a startup entry with Python version, platform, and timestamp."""
        self.info(
            phase="system",
            event="startup",
            data={
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": platform.node(),
                "timestamp_local": datetime.now().isoformat(),
            },
        )


# ──────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────

def get_logger() -> NWSLogger:
    """
    Return the singleton :class:`NWSLogger` instance.

    Thread-safe: the first call creates the instance; subsequent calls
    return the same object without acquiring the creation lock.

    Returns:
        The application-wide :class:`NWSLogger`.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NWSLogger()
    return _instance
