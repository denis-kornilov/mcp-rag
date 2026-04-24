"""Multi-project registry — maps project keys to isolated data directories.

Each project gets a UUID key and its own subdirectory under SERVER_DATA_ROOT:
    {server_data_root}/{key}/chroma_db/
    {server_data_root}/{key}/_manifests/   (via chroma_path/_manifests)

The registry is persisted in {server_data_root}/projects.json.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .error_reporter import ErrorReporter

error_reporter = ErrorReporter("rag_server.project_manager")

_LOCK = threading.Lock()
_instance: "ProjectManager | None" = None


class ProjectManager:
    def __init__(self, server_data_root: str | Path) -> None:
        self._root = Path(server_data_root).resolve()
        self._manifest = self._root / "projects.json"
        self._root.mkdir(parents=True, exist_ok=True)
        self._projects: Dict[str, Dict[str, Any]] = self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self._manifest.exists():
            return {}
        try:
            return json.loads(self._manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            error_reporter.warn(stage="pm_load", message="failed to load projects.json", exc=exc)
            return {}

    def _save(self) -> None:
        self._manifest.write_text(
            json.dumps(self._projects, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def register(self, hint: str = "", key: str = "") -> Dict[str, Any]:
        """Create a new project. Returns its full config including the key.

        If *key* is provided it is used as-is (and as the subdirectory name).
        Otherwise a UUID hex is generated.
        Name is derived from the last component of hint (project folder name).
        """
        with _LOCK:
            key = key.strip() or uuid.uuid4().hex
            project_dir = self._root / key
            project_dir.mkdir(parents=True, exist_ok=True)
            derived_name = Path(hint.strip()).name if hint.strip() else key[:8]
            entry: Dict[str, Any] = {
                "key": key,
                "name": derived_name,
                "hint": hint.strip(),
                "chroma_path": str(project_dir / "chroma_db"),
                "project_root": hint.strip() or str(project_dir),
                "created_at": time.time(),
            }
            self._projects[key] = entry
            self._save()
            return dict(entry)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._projects.get(key)

    def list_all(self) -> List[Dict[str, Any]]:
        return [
            {"key": k, "name": v["name"], "hint": v.get("hint", ""), "created_at": v.get("created_at")}
            for k, v in self._projects.items()
        ]

    def delete(self, key: str) -> bool:
        with _LOCK:
            if key not in self._projects:
                return False
            del self._projects[key]
            self._save()
            return True


def get_manager() -> ProjectManager:
    """Return the global ProjectManager singleton (initialised lazily)."""
    global _instance
    if _instance is not None:
        return _instance
    with _LOCK:
        if _instance is not None:
            return _instance
        from .settings import settings  # noqa: PLC0415
        root = str(getattr(settings, "server_data_root", str(Path(__file__).resolve().parent / "rag_data")))
        _instance = ProjectManager(root)
    return _instance
