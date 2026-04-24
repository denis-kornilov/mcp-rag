"""Per-request project context.

For stdio: set once at startup from PROJECT_KEY env var or .mcp-rag file.
For HTTP:  set per-request by ProjectKeyASGIMiddleware from X-Project-Key header.

Uses ContextVar for async/per-request isolation (HTTP mode) plus a module-level
fallback dict for raw threading.Thread workers (e.g. _run_action in gateway).
Both are updated together in set_project(), so threading.Thread callers always
see the last set_project() value.
"""
from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Any, Dict

_chroma_path: ContextVar[str | None] = ContextVar("project_chroma_path", default=None)
_project_root: ContextVar[str | None] = ContextVar("project_root", default=None)
_project_key: ContextVar[str | None] = ContextVar("project_key", default=None)

# Module-level fallback — visible to all threads in the same process.
_fallback: Dict[str, str] = {"chroma_path": "", "project_root": "", "key": ""}
_fallback_lock = threading.Lock()


def set_project(*, chroma_path: str, project_root: str, key: str = "") -> None:
    _chroma_path.set(chroma_path)
    _project_root.set(project_root)
    _project_key.set(key)
    with _fallback_lock:
        _fallback["chroma_path"] = chroma_path
        _fallback["project_root"] = project_root
        _fallback["key"] = key


def get_chroma_path() -> str | None:
    return _chroma_path.get() or _fallback.get("chroma_path") or None


def get_project_root() -> str | None:
    return _project_root.get() or _fallback.get("project_root") or None


def get_project_key() -> str | None:
    return _project_key.get() or _fallback.get("key") or None


def current_project() -> Dict[str, Any]:
    with _fallback_lock:
        fb = dict(_fallback)
    return {
        "key": _project_key.get() or fb["key"],
        "chroma_path": _chroma_path.get() or fb["chroma_path"],
        "project_root": _project_root.get() or fb["project_root"],
    }
