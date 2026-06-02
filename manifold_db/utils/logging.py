"""
Structured logging for manifold database.

Provides configurable logging setup, execution timing utilities, and a
performance tracker for monitoring query latency and throughput.
"""

from __future__ import annotations

import functools
import json
import logging
import statistics
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_format: bool = False,
) -> None:
    """Configure the root logger for the manifold database package.

    Parameters
    ----------
    level : str
        Logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).
    log_file : str or None
        Optional file path for a file handler. If *None*, logs only to stderr.
    json_format : bool
        If *True*, emit structured JSON log records instead of plain text.
    """
    root = logging.getLogger("manifold_db")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on repeated calls
    root.handlers.clear()

    if json_format:
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # Stream handler (stderr)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``manifold_db`` namespace.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    return logging.getLogger(f"manifold_db.{name}")


class _JsonFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


# ---------------------------------------------------------------------------
# log_timer context manager
# ---------------------------------------------------------------------------


@contextmanager
def log_timer(
    operation: str,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> Generator[None, None, None]:
    """Context manager that logs the wall-clock duration of a block.

    Parameters
    ----------
    operation : str
        Human-readable label for the timed operation.
    logger : logging.Logger or None
        Logger to use; falls back to ``get_logger("timer")``.
    level : int
        Logging level for the duration message.

    Yields
    ------
    None

    Example
    -------
    >>> with log_timer("atlas build", logger=log):
    ...     build_atlas(data)
    """
    _logger = logger or get_logger("timer")
    start = time.perf_counter()
    _logger.log(level, "START  %s", operation)
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.log(level, "FINISH %s  [%.2f ms]", operation, elapsed_ms)


# ---------------------------------------------------------------------------
# log_execution decorator
# ---------------------------------------------------------------------------


def log_execution(
    func: Any | None = None,
    *,
    level: int = logging.DEBUG,
) -> Any:
    """Decorator that logs entry, exit, and timing of a function call.

    Can be used bare ``@log_execution`` or with options ``@log_execution(level=logging.INFO)``.

    Parameters
    ----------
    func : callable or None
        The decorated function (passed automatically when used without arguments).
    level : int
        Logging level.

    Returns
    -------
    callable
        Wrapped function.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _logger = get_logger(fn.__module__)
            qualname = f"{fn.__module__}.{fn.__qualname__}"
            _logger.log(level, "→ ENTER  %s", qualname)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                _logger.log(level, "← EXIT   %s  [%.2f ms]", qualname, elapsed_ms)
                return result
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                _logger.log(
                    logging.ERROR,
                    "✗ ERROR  %s  [%.2f ms] %s",
                    qualname,
                    elapsed_ms,
                    exc,
                    exc_info=True,
                )
                raise

        return wrapper

    # Support both @log_execution and @log_execution(level=...)
    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------


@dataclass
class _QueryRecord:
    """Single query performance record."""

    chart_id: str
    query_type: str
    duration_ms: float
    n_results: int
    timestamp: float = field(default_factory=time.time)


class PerformanceTracker:
    """Tracks and reports query performance metrics.

    Thread-safe (single-thread assumption is fine for CPython's GIL on
    standard deployments). Keeps a bounded ring buffer of recent queries and
    maintains aggregated statistics.

    Parameters
    ----------
    max_history : int
        Maximum number of recent queries to retain in memory.
    """

    def __init__(self, max_history: int = 10_000) -> None:
        self._max_history = max_history
        self._history: list[_QueryRecord] = []
        self._logger = get_logger("performance")

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_query(
        self,
        chart_id: str,
        query_type: str,
        duration_ms: float,
        n_results: int,
    ) -> None:
        """Record a completed query.

        Parameters
        ----------
        chart_id : str
            Identifier of the chart that handled the query.
        query_type : str
            Type of query (e.g. ``'geodesic'``, ``'tangent'``, ``'cross_modal'``).
        duration_ms : float
            Wall-clock duration in milliseconds.
        n_results : int
            Number of results returned.
        """
        record = _QueryRecord(
            chart_id=chart_id,
            query_type=query_type,
            duration_ms=duration_ms,
            n_results=n_results,
        )
        self._history.append(record)
        # Trim to max_history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    # ------------------------------------------------------------------
    # Aggregated statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return aggregated performance statistics.

        Returns
        -------
        dict
            Keys:
            * ``total_queries`` – total number recorded
            * ``total_duration_ms`` – cumulative query time
            * ``avg_duration_ms`` – mean query duration
            * ``median_duration_ms`` – median query duration
            * ``p95_duration_ms`` – 95th-percentile duration
            * ``min_duration_ms`` – fastest query
            * ``max_duration_ms`` – slowest query
            * ``queries_per_second`` – throughput estimate
            * ``by_type`` – per-query-type breakdown
        """
        if not self._history:
            return {
                "total_queries": 0,
                "total_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
                "median_duration_ms": 0.0,
                "p95_duration_ms": 0.0,
                "min_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "queries_per_second": 0.0,
                "by_type": {},
            }

        durations = [r.duration_ms for r in self._history]
        sorted_d = sorted(durations)
        total_duration = sum(durations)
        total_time_span = (
            self._history[-1].timestamp - self._history[0].timestamp
        ) or 1e-9

        by_type: dict[str, dict[str, Any]] = {}
        for r in self._history:
            bucket = by_type.setdefault(r.query_type, {"count": 0, "total_ms": 0.0})
            bucket["count"] += 1
            bucket["total_ms"] += r.duration_ms
        for qt, info in by_type.items():
            info["avg_ms"] = info["total_ms"] / info["count"]

        return {
            "total_queries": len(self._history),
            "total_duration_ms": round(total_duration, 3),
            "avg_duration_ms": round(statistics.mean(durations), 3),
            "median_duration_ms": round(statistics.median(durations), 3),
            "p95_duration_ms": round(sorted_d[int(len(sorted_d) * 0.95)], 3),
            "min_duration_ms": round(min(durations), 3),
            "max_duration_ms": round(max(durations), 3),
            "queries_per_second": round(len(self._history) / total_time_span, 2),
            "by_type": by_type,
        }

    # ------------------------------------------------------------------
    # Recent queries
    # ------------------------------------------------------------------

    def recent_queries(self, n: int = 100) -> list[dict[str, Any]]:
        """Return the most recent *n* query records as plain dicts.

        Parameters
        ----------
        n : int
            Number of records to return.

        Returns
        -------
        list[dict]
            Each dict contains ``chart_id``, ``query_type``,
            ``duration_ms``, ``n_results``, and ``timestamp``.
        """
        slice_ = self._history[-n:]
        return [
            {
                "chart_id": r.chart_id,
                "query_type": r.query_type,
                "duration_ms": round(r.duration_ms, 3),
                "n_results": r.n_results,
                "timestamp": r.timestamp,
            }
            for r in slice_
        ]

    def reset(self) -> None:
        """Clear all recorded queries."""
        self._history.clear()
