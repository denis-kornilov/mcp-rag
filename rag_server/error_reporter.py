from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path


class ErrorReporter:
    """Logs exceptions to system log and debug.log with required context."""

    def __init__(self, logger_name: str, debug_log_path: Path | None = None) -> None:
        self._logger = logging.getLogger(logger_name)
        self._debug_log_path = debug_log_path or (Path(__file__).resolve().parents[2] / "debug.log")

    def warn(
        self,
        *,
        stage: str,
        symbol: str = "-",
        tf: str = "-",
        ts: str = "-",
        message: str = "",
        exc: Exception | None = None,
    ) -> None:
        line = self._format_line(level="WARN", stage=stage, symbol=symbol, tf=tf, ts=ts, message=message, exc=exc)
        self._logger.warning(line)
        self._append_debug(line, exc)

    def error(
        self,
        *,
        stage: str,
        symbol: str = "-",
        tf: str = "-",
        ts: str = "-",
        message: str = "",
        exc: Exception | None = None,
    ) -> None:
        line = self._format_line(level="ERROR", stage=stage, symbol=symbol, tf=tf, ts=ts, message=message, exc=exc)
        self._logger.error(line)
        self._append_debug(line, exc)

    def _format_line(
        self,
        *,
        level: str,
        stage: str,
        symbol: str,
        tf: str,
        ts: str,
        message: str,
        exc: Exception | None,
    ) -> str:
        exc_part = f" exc={type(exc).__name__}:{exc}" if exc is not None else ""
        return (
            f"[{level}] symbol={symbol} tf={tf} ts={ts} stage={stage} "
            f"message={message}{exc_part}"
        )

    def _append_debug(self, line: str, exc: Exception | None) -> None:
        self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._debug_log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")
            if exc is not None:
                tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=5)).strip()
                if tb:
                    fh.write(tb + "\n")
