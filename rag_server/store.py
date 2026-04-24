from __future__ import annotations

import threading
from typing import Dict

from .error_reporter import ErrorReporter
from .settings import settings

_clients: Dict[str, object] = {}
_clients_lock = threading.Lock()
error_reporter = ErrorReporter("rag_server.store")


def _resolve_chroma_path() -> str:
    from .project_context import get_chroma_path  # noqa: PLC0415
    return get_chroma_path() or settings.chroma_path


def _get_client(chroma_path: str | None = None):
    persist_dir = chroma_path or _resolve_chroma_path()

    if persist_dir in _clients:
        return _clients[persist_dir]

    with _clients_lock:
        if persist_dir in _clients:
            return _clients[persist_dir]

        try:
            import numpy as np  # type: ignore
            if not hasattr(np, "float_"):
                setattr(np, "float_", np.float64)
            if not hasattr(np, "int_"):
                setattr(np, "int_", np.int64)
            if not hasattr(np, "uint"):
                setattr(np, "uint", np.uint64)
        except Exception as exc:
            error_reporter.warn(stage="store_numpy_compat", message="numpy compatibility shim failed", exc=exc)

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        try:
            client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as exc:
            error_reporter.warn(
                stage="store_persistent_client_init",
                message="fallback to chromadb.Client with ChromaSettings",
                exc=exc,
            )
            client = chromadb.Client(
                ChromaSettings(is_persistent=True, persist_directory=persist_dir, anonymized_telemetry=False)
            )
        _clients[persist_dir] = client
        return client


def get_collection(name: str = "docs", chroma_path: str | None = None):
    client = _get_client(chroma_path)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def reset_collection(name: str = "docs", chroma_path: str | None = None):
    client = _get_client(chroma_path)
    try:
        client.delete_collection(name)
    except Exception as exc:
        error_reporter.warn(
            stage="store_reset_collection_delete",
            message=f"delete_collection ignored for name={name}",
            exc=exc,
        )
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
