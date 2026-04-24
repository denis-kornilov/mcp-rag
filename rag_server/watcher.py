"""File system watcher — triggers incremental ingest on file changes.

Uses watchdog library (must be installed: pip install watchdog).
Changes are debounced: ingest fires only after DEBOUNCE_S seconds of silence.
Gracefully no-ops if watchdog is not installed.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Set

from .error_reporter import ErrorReporter
from .settings import settings

logger = logging.getLogger("rag_server.watcher")
error_reporter = ErrorReporter("rag_server.watcher")

DEBOUNCE_S = 4.0


class _DebounceHandler:
    def __init__(self, root: Path, collection: str, exts: Set[str], ctx=None) -> None:
        self._root = root
        self._collection = collection
        self._exts = exts
        self._pending: Set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        # Capture project context at watcher startup so debounce thread can use it
        import contextvars  # noqa: PLC0415
        self._ctx = ctx or contextvars.copy_context()

    def on_path(self, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() not in self._exts:
            return
        try:
            rel = str(p.relative_to(self._root))
        except ValueError:
            return
        with self._lock:
            self._pending.add(rel)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_S, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            paths = list(self._pending)
            self._pending.clear()
            self._timer = None
        if not paths:
            return
        # Run ingest inside captured project context
        self._ctx.run(self._do_ingest, paths)

    def _do_ingest(self, paths: list) -> None:
        logger.info("[watcher] Incremental ingest: %d file(s) changed", len(paths))
        try:
            from .ingest_ops import ingest_paths  # noqa: PLC0415
            ingest_paths(rel_paths=paths, root=self._root, collection=self._collection)
            from .hybrid_search import invalidate  # noqa: PLC0415
            invalidate()
        except Exception as exc:
            error_reporter.warn(stage="watcher_flush", message="ingest_paths failed", exc=exc)


def start_watcher(root: Path, collection: str = "code", context=None) -> bool:
    """Start watchdog observer in a daemon thread.

    Returns True if started successfully, False if watchdog is not installed.
    Install with: pip install watchdog
    """
    try:
        from watchdog.observers import Observer  # noqa: PLC0415
        from watchdog.events import FileSystemEventHandler, FileSystemEvent  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "[watcher] watchdog not installed — FS watcher disabled. "
            "Run: pip install watchdog"
        )
        return False

    raw_exts = getattr(settings, "auto_ingest_extensions", "")
    exts = {e.strip().lower() for e in raw_exts.split(",") if e.strip()}
    if not exts:
        exts = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml"}

    debouncer = _DebounceHandler(root, collection, exts, ctx=context)

    class _Adapter(FileSystemEventHandler):
        def on_modified(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                debouncer.on_path(event.src_path)

        def on_created(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                debouncer.on_path(event.src_path)

        def on_moved(self, event: FileSystemEvent) -> None:
            if not event.is_directory:
                debouncer.on_path(getattr(event, "dest_path", event.src_path))

    observer = Observer()
    observer.schedule(_Adapter(), str(root), recursive=True)
    observer.daemon = True
    observer.start()
    logger.info("[watcher] FS watcher started on %s (collection=%s)", root, collection)
    return True
