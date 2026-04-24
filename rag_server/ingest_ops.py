from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import pathspec
from .chunker import chunk_file
from .embeddings import embed_texts
from .error_reporter import ErrorReporter
from .settings import settings
from .store import get_collection, reset_collection


class StopIngestion(Exception):
    """Raised by progress_cb to request a graceful stop of the ingest loop.

    full_ingest catches this, saves the manifest for already-processed files,
    and returns with mode='stopped' so the caller can report progress to date.
    """


def _get_default_exts() -> set[str]:
    raw = getattr(settings, "auto_ingest_extensions", "")
    if not raw:
        return {
            ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
            ".js", ".ts", ".tsx", ".css", ".html",
        }
    return {ext.strip() for ext in raw.split(",") if ext.strip()}
SPECIAL_FILENAMES = {".env", ".env.example", ".env.sample", ".env.template"}
INDEX_RESOURCES_FILENAME = "rag_index_resources.txt"

# The mcp_rag_package directory: app/ lives one level below, package root is parents[1].
# We resolve it once at import time so we can exclude it from any walk regardless
# of how PROJECT_ROOT is configured.
_PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]


def _data_roots() -> frozenset[Path]:
    """Return the set of resolved data directories that must never be ingested.

    Covers:
    - The entire mcp_rag_data parent directory (chroma_db + all model caches live there)
    - Each individual configured path as a fallback if they live elsewhere
    Called lazily so settings are fully loaded before we read them.
    """
    candidates: list[str | None] = [
        settings.chroma_path,
        # model caches — read from env directly (no longer in rag_server settings)
        os.environ.get("HF_HOME"),
        os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
        os.environ.get("TORCH_HOME"),
    ]
    # Also add the parent of chroma_path — that's typically the top-level data
    # dir (e.g. mcp_rag_data) which contains ALL model caches and the vector DB.
    if settings.chroma_path:
        candidates.append(str(Path(settings.chroma_path).parent))

    project_root = Path(settings.project_root).resolve()
    roots: set[Path] = set()
    for raw in candidates:
        if not raw:
            continue
        try:
            p = Path(raw).resolve()
            # If the configured path doesn't exist yet, walk up to the first
            # existing ancestor — so we still exclude a not-yet-created chroma_db.
            while not p.exists() and p != p.parent:
                p = p.parent
            # Never add the project root itself as a data root
            if p == project_root:
                continue
            roots.add(p)
        except Exception:
            pass
    return frozenset(roots)

# Directories that are never worth indexing: VCS internals, caches, build
# artefacts, virtual-envs, large generated data dirs.
SKIP_DIRS: frozenset[str] = frozenset({
    # VCS
    ".git", ".hg", ".svn",
    # Python
    "__pycache__", ".mypy_cache", ".ruff_cache", ".dmypy",
    ".pytest_cache", ".hypothesis", "*.egg-info",
    # JS / TS
    "node_modules", ".next", ".nuxt", ".turbo",
    # Virtual environments
    "venv", ".venv", "env", "ENV", ".env.d",
    # Build / dist
    "dist", "build", "out", "target",
    # IDE
    ".idea", ".vscode",
    # RAG data — never index your own vector store or model cache
    "chroma_db", "mcp_rag_data",
})

ProgressCb = Callable[[str, Dict[str, Any]], None] | None
error_reporter = ErrorReporter("rag_server.ingest_ops")


def _parse_allowlist_patterns(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [part.strip().strip("/") for part in raw.split(",") if part.strip().strip("/")]


def _is_allowed_file(path: Path, exts: set[str]) -> bool:
    return path.suffix.lower() in exts or path.name.lower() in SPECIAL_FILENAMES


def _should_skip_dir(name: str, parent: Path, data_roots: frozenset[Path] | None = None) -> bool:
    """Return True if this directory should be excluded from all walks.

    Checks (in order):
    1. Name-based deny-list (SKIP_DIRS, *.egg-info)
    2. RAG package directory (never index our own code)
    3. Configured data directories: chroma_db, model caches — never index the
       knowledge base or model weights regardless of where they live on disk.
    """
    if name in SKIP_DIRS or name.endswith(".egg-info"):
        return True
    try:
        resolved = (parent / name).resolve()
        if resolved == _PACKAGE_ROOT:
            return True
        roots = data_roots if data_roots is not None else _data_roots()
        for data_root in roots:
            # Exclude if the candidate IS a data root or is inside one
            if resolved == data_root or data_root in resolved.parents:
                return True
    except OSError:
        pass
    return False


class GitIgnoreFilter:
    """Handles .gitignore logic using pathspec."""
    def __init__(self, root: Path):
        self.root = root
        self.spec = None
        ignore_file = root / ".gitignore"
        if ignore_file.exists():
            try:
                lines = ignore_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                self.spec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
            except Exception as exc:
                error_reporter.warn(
                    stage="load_gitignore",
                    symbol=str(ignore_file),
                    message="failed to parse .gitignore",
                    exc=exc
                )

    def is_ignored(self, path: Path) -> bool:
        if self.spec is None:
            return False
        try:
            rel_path = path.relative_to(self.root).as_posix()
            return self.spec.match_file(rel_path)
        except Exception:
            return False


def _iter_tree_files(base: Path, exts: set[str], ignore_filter: GitIgnoreFilter | None = None) -> Iterable[Path]:
    dr = _data_roots()
    for dirpath, dirs, filenames in os.walk(base):
        parent = Path(dirpath)
        dirs[:] = [d for d in dirs if not _should_skip_dir(d, parent, dr)]
        for fn in filenames:
            path = parent / fn
            if ignore_filter and ignore_filter.is_ignored(path):
                continue
            if path.is_file() and _is_allowed_file(path, exts):
                yield path


def _iter_pattern_matches(root: Path, pattern: str, exts: set[str], ignore_filter: GitIgnoreFilter | None = None) -> Iterable[Path]:
    target = root / pattern
    if target.exists():
        if target.is_file():
            if ignore_filter and ignore_filter.is_ignored(target):
                return
            if _is_allowed_file(target, exts):
                yield target
            return
        if target.is_dir():
            yield from _iter_tree_files(target, exts, ignore_filter=ignore_filter)
            return
    # Patterns without path separators (e.g. "*.py") only match top-level with
    # glob(); use rglob() so they recurse through subdirectories.
    glob_fn = root.rglob if ("/" not in pattern and "**" not in pattern) else root.glob
    for match in glob_fn(pattern):
        if ignore_filter and ignore_filter.is_ignored(match):
            continue
        if match.is_file():
            if _is_allowed_file(match, exts):
                yield match
            continue
        if match.is_dir():
            yield from _iter_tree_files(match, exts, ignore_filter=ignore_filter)


def _iter_root_files(root: Path, exts: set[str], ignore_filter: GitIgnoreFilter | None = None) -> Iterable[Path]:
    dr = _data_roots()
    for dirpath, dirs, filenames in os.walk(root):
        parent = Path(dirpath)
        dirs[:] = [d for d in dirs if not _should_skip_dir(d, parent, dr)]
        for fn in filenames:
            path = parent / fn
            if ignore_filter and ignore_filter.is_ignored(path):
                continue
            if path.is_file() and _is_allowed_file(path, exts):
                yield path


def iter_files(root: Path, exts: set[str], includes: List[str] | None = None, limit: int | None = None) -> Iterable[Path]:
    allowlist = _parse_allowlist_patterns(getattr(settings, "auto_ingest_allowlist", "")) if includes is None else includes
    ignore_filter = GitIgnoreFilter(root)
    iterator = _iter_root_files(root, exts, ignore_filter=ignore_filter) if not allowlist else (
        path for pattern in allowlist for path in _iter_pattern_matches(root, pattern, exts, ignore_filter=ignore_filter)
    )
    seen: set[str] = set()
    remaining = limit
    for path in iterator:
        try:
            rel = path.relative_to(root).as_posix()
        except Exception as exc:
            error_reporter.warn(
                stage="iter_files_relative_to_root",
                symbol=path.as_posix(),
                message=f"failed to compute relative path for root={root}",
                exc=exc,
            )
            continue
        if rel in seen:
            continue
        seen.add(rel)
        yield path
        if remaining is not None:
            remaining -= 1
            if remaining <= 0:
                return


def _progress(cb: ProgressCb, stage: str, **payload: Any) -> None:
    if cb:
        cb(stage, payload)


def load_index_resources(root: Path) -> List[str]:
    manifest_path = root / INDEX_RESOURCES_FILENAME
    if not manifest_path.exists() or not manifest_path.is_file():
        return []
    entries: List[str] = []
    for raw_line in manifest_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def scan_files_preview(
    root: Path,
    exts: set[str] | None = None,
    includes: List[str] | None = None,
    limit_files: int | None = None,
) -> Dict[str, Any]:
    """Return a summary of files that would be ingested — without reading or embedding anything.

    Used by the MCP scan_project tool so the LLM can report scope to the user
    before starting a potentially long ingest job.
    """
    exts = exts or set(_get_default_exts())
    allowlist = includes if includes is not None else _parse_allowlist_patterns(getattr(settings, "auto_ingest_allowlist", ""))
    files = list(iter_files(root, exts, includes=allowlist or None, limit=limit_files))

    by_ext: Dict[str, int] = {}
    total_bytes = 0
    for fp in files:
        ext = fp.suffix.lower() or "(no ext)"
        by_ext[ext] = by_ext.get(ext, 0) + 1
        try:
            total_bytes += fp.stat().st_size
        except OSError:
            pass

    return {
        "root": str(root),
        "allowlist": allowlist,
        "files_found": len(files),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1_048_576, 2),
        "by_extension": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])),
        "sample_paths": [fp.relative_to(root).as_posix() for fp in files[:20]],
        "skipped_dirs": sorted(SKIP_DIRS),
    }


def _manifest_dir() -> Path:
    from .project_context import get_chroma_path  # noqa: PLC0415
    base = Path(get_chroma_path() or settings.chroma_path)
    out = base / "_manifests"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _manifest_path(collection: str) -> Path:
    safe = hashlib.sha1(collection.encode("utf-8")).hexdigest()[:12]
    return _manifest_dir() / f"{collection}-{safe}.json"


def load_manifest(collection: str) -> Dict[str, Dict[str, int]]:
    path = _manifest_path(collection)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        error_reporter.warn(
            stage="manifest_load_parse",
            symbol=collection,
            message=f"failed to parse manifest at {path}",
            exc=exc,
        )
        return {}
    files = data.get("files")
    return files if isinstance(files, dict) else {}


def save_manifest(collection: str, files: Dict[str, Dict[str, int]]) -> None:
    path = _manifest_path(collection)
    payload = {
        "collection": collection,
        "files": files,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pause_state_path(collection: str) -> Path:
    safe = hashlib.sha1(collection.encode("utf-8")).hexdigest()[:12]
    return _manifest_dir() / f"{collection}-{safe}.paused.json"


def save_pause_state(collection: str, files_done: int, files_total: int, chunks_written: int) -> None:
    """Mark that a full ingest was paused mid-way. Consumed by sync_project on next run."""
    import time as _time
    path = _pause_state_path(collection)
    payload = {
        "collection": collection,
        "paused": True,
        "files_done": files_done,
        "files_total": files_total,
        "chunks_written": chunks_written,
        "paused_at": _time.time(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pause_state(collection: str) -> Dict[str, Any] | None:
    """Return pause state if the last full ingest was stopped, else None."""
    path = _pause_state_path(collection)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data.get("paused") else None
    except Exception:
        return None


def clear_pause_state(collection: str) -> None:
    """Remove pause marker after a successful resume."""
    path = _pause_state_path(collection)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def snapshot_project(root: Path, exts: set[str], includes: List[str] | None = None, limit: int | None = None) -> Dict[str, Dict[str, int]]:
    state: Dict[str, Dict[str, int]] = {}
    for fp in iter_files(root, exts, includes=includes, limit=limit):
        try:
            stat = fp.stat()
            rel = fp.relative_to(root).as_posix()
            state[rel] = {
                "mtime_ns": int(stat.st_mtime_ns),
                "size": int(stat.st_size),
            }
        except Exception as exc:
            error_reporter.warn(
                stage="snapshot_project_stat",
                symbol=fp.as_posix(),
                message="failed to collect file stat for snapshot",
                exc=exc,
            )
            continue
    return state


def plan_ingest_work(
    root: Path,
    exts: set[str] | None = None,
    includes: List[str] | None = None,
    limit_files: int | None = None,
    max_file_bytes: int | None = None,
    progress_cb: ProgressCb = None,
) -> Dict[str, Any]:
    exts = exts or set(_get_default_exts())
    max_file_bytes = max_file_bytes or int(getattr(settings, "auto_ingest_max_file_bytes", 800_000))
    files = list(iter_files(root, exts, includes=includes, limit=limit_files))
    rel_paths: List[str] = []
    files_scanned = 0
    files_skipped = 0
    chunks_total = 0

    for fp in files:
        files_scanned += 1
        rel_path = fp.relative_to(root).as_posix()
        _progress(progress_cb, "scan_file", index=files_scanned, total=len(files), path=rel_path)
        try:
            if not fp.exists() or not fp.is_file():
                files_skipped += 1
                continue
            if fp.stat().st_size > max_file_bytes:
                files_skipped += 1
                continue
            data = fp.read_text(encoding="utf-8", errors="ignore")
            rel_paths.append(rel_path)
            chunks_total += len(chunk_file(Path(rel_path), data))
        except Exception as exc:
            error_reporter.warn(
                stage="plan_ingest_work_scan_file",
                symbol=rel_path,
                message="file skipped during plan build",
                exc=exc,
            )
            files_skipped += 1
            continue

    report = {
        "files_scanned": files_scanned,
        "files_planned": len(rel_paths),
        "files_skipped": files_skipped,
        "chunks_total": chunks_total,
        "rel_paths": sorted(rel_paths),
    }
    _progress(progress_cb, "scan_plan_complete", **report)
    return report


def _prepare_single_file(rel_path: str, root: Path, collection: str, max_file_bytes: int) -> List[Dict[str, Any]]:
    fp = root / rel_path
    items = []
    try:
        if not fp.exists() or not fp.is_file():
            return []
        if fp.stat().st_size > max_file_bytes:
            return []
        data = fp.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_file(Path(rel_path), data)
        for ch in chunks:
            ch_meta = ch.get("metadata", {}).copy()
            ch_meta.setdefault("repo_root", str(root))
            ch_meta.setdefault("project", root.name)
            chunk_type = str(ch_meta.get("type", "file"))
            symbol = str(ch_meta.get("symbol", ""))
            start = str(ch_meta.get("start_line", ""))
            end = str(ch_meta.get("end_line", ""))
            key = f"{collection}:{rel_path}:{chunk_type}:{symbol}:{start}:{end}"
            did = hashlib.sha1(key.encode("utf-8")).hexdigest()
            items.append({
                "text": ch["text"],
                "metadata": ch_meta,
                "collection": collection,
                "id": did,
            })
    except Exception as exc:
        error_reporter.warn(
            stage="prepare_single_file",
            symbol=rel_path,
            message="file skipped during chunk preparation",
            exc=exc,
        )
    return items


def _build_items_for_files(
    rel_paths: List[str],
    root: Path,
    collection: str,
    max_file_bytes: int,
    progress_cb: ProgressCb = None,
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    stats = {
        "files_seen": 0,
        "files_loaded": 0,
        "files_skipped": 0,
        "chunks_prepared": 0,
    }
    
    # Use ThreadPoolExecutor for faster file I/O and chunking
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    max_workers = min(32, (os.cpu_count() or 4) * 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map file processing tasks
        future_to_path = {
            executor.submit(_prepare_single_file, path, root, collection, max_file_bytes): path 
            for path in rel_paths
        }
        
        for idx, future in enumerate(future_to_path, start=1):
            stats["files_seen"] += 1
            rel_path = future_to_path[future]
            try:
                file_items = future.result()
                if file_items:
                    items.extend(file_items)
                    stats["files_loaded"] += 1
                    stats["chunks_prepared"] += len(file_items)
                    _progress(
                        progress_cb,
                        "prepare_file",
                        index=idx,
                        total=len(rel_paths),
                        path=rel_path,
                        chunks_in_file=len(file_items),
                    )
                else:
                    stats["files_skipped"] += 1
            except Exception as exc:
                error_reporter.warn(
                    stage="build_items_for_files_future",
                    symbol=rel_path,
                    message="future failed during chunk preparation",
                    exc=exc,
                )
                stats["files_skipped"] += 1

    return {"items": items, "stats": stats}


def _upsert_items(items: List[Dict[str, Any]], collection: str, replace_by_path: bool, progress_cb: ProgressCb = None) -> Dict[str, Any]:
    col = get_collection(collection)
    if replace_by_path:
        paths = sorted({str((it.get("metadata") or {}).get("path", "")) for it in items if (it.get("metadata") or {}).get("path")})
        for path in paths:
            try:
                col.delete(where={"path": path})
            except Exception as exc:
                error_reporter.warn(
                    stage="upsert_items_replace_by_path_delete",
                    symbol=path,
                    message=f"failed to delete stale chunks collection={collection}",
                    exc=exc,
                )
    sub = max(1, settings.embed_batch_size * 2)
    written = 0
    for i in range(0, len(items), sub):
        chunk = items[i : i + sub]
        texts = [r["text"] for r in chunk]
        ids = [r["id"] for r in chunk]
        metas = [r["metadata"] for r in chunk]
        embs = embed_texts(texts)
        if hasattr(col, "upsert"):
            col.upsert(documents=texts, embeddings=embs, ids=ids, metadatas=metas)
        else:
            col.add(documents=texts, embeddings=embs, ids=ids, metadatas=metas)
        written += len(chunk)
        _progress(progress_cb, "upsert_batch", written=written, total=len(items), batch_size=len(chunk))
    return {"written": written}


def delete_paths(collection: str, rel_paths: List[str], progress_cb: ProgressCb = None) -> Dict[str, Any]:
    col = get_collection(collection)
    deleted = 0
    for idx, rel_path in enumerate(rel_paths, start=1):
        try:
            col.delete(where={"path": rel_path})
            deleted += 1
        except Exception as exc:
            error_reporter.warn(
                stage="delete_paths",
                symbol=rel_path,
                message=f"failed to delete path in collection={collection}",
                exc=exc,
            )
        _progress(progress_cb, "delete_path", index=idx, total=len(rel_paths), path=rel_path, deleted=deleted)
    return {"deleted_paths": deleted}


def ingest_paths(
    rel_paths: List[str],
    root: Path,
    collection: str = "code",
    replace_by_path: bool = True,
    max_file_bytes: int | None = None,
    progress_cb: ProgressCb = None,
) -> Dict[str, Any]:
    max_file_bytes = max_file_bytes or int(getattr(settings, "auto_ingest_max_file_bytes", 800_000))
    manifest = load_manifest(collection)
    prepared = _build_items_for_files(rel_paths, root, collection, max_file_bytes, progress_cb=progress_cb)
    items = prepared["items"]
    write_report = {"written": 0}
    if items:
        write_report = _upsert_items(items, collection, replace_by_path=replace_by_path, progress_cb=progress_cb)
    for rel_path in rel_paths:
        fp = root / rel_path
        if fp.exists() and fp.is_file():
            try:
                stat = fp.stat()
                manifest[rel_path] = {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
            except Exception as exc:
                error_reporter.warn(
                    stage="ingest_paths_manifest_stat",
                    symbol=rel_path,
                    message="failed to update manifest stat for path",
                    exc=exc,
                )
        else:
            manifest.pop(rel_path, None)
    save_manifest(collection, manifest)
    return {
        "mode": "incremental",
        "collection": collection,
        "root": str(root),
        "paths_requested": len(rel_paths),
        "prepared": prepared["stats"],
        "written": write_report["written"],
    }


def full_ingest(
    root: Path,
    collection: str = "code",
    exts: set[str] | None = None,
    includes: List[str] | None = None,
    limit_files: int | None = None,
    max_file_bytes: int | None = None,
    progress_cb: ProgressCb = None,
) -> Dict[str, Any]:
    exts = exts or set(_get_default_exts())
    max_file_bytes = max_file_bytes or int(getattr(settings, "auto_ingest_max_file_bytes", 800_000))
    state = snapshot_project(root, exts, includes=includes, limit=limit_files)
    rel_paths = sorted(state.keys())
    _progress(progress_cb, "reset_collection", collection=collection)
    reset_collection(collection)
    # Save empty manifest immediately after reset so resume works correctly:
    # if the process is killed mid-way, the next sync_project will see a non-empty
    # manifest and switch to incremental mode, processing only remaining files.
    manifest: Dict[str, Dict[str, int]] = {}
    save_manifest(collection, manifest)

    col = get_collection(collection)
    batch_size = max(8, settings.embed_batch_size)
    target_buffer_size = batch_size * 4  # Accumulate chunks up to this size
    total_chunks = 0
    written = 0
    stats = {"files_seen": 0, "files_loaded": 0, "files_skipped": 0, "chunks_prepared": 0}

    item_buffer: List[Dict[str, Any]] = []
    manifest_updates_buffer: Dict[str, Dict[str, int]] = {}
    files_processed_in_buffer = 0

    def _flush_buffer() -> None:
        nonlocal written, item_buffer, manifest_updates_buffer, files_processed_in_buffer
        if not item_buffer:
            return
        
        # Batch insert
        for i in range(0, len(item_buffer), batch_size):
            batch = item_buffer[i : i + batch_size]
            texts = [r["text"] for r in batch]
            ids = [r["id"] for r in batch]
            metas = [r["metadata"] for r in batch]
            embs = embed_texts(texts)
            if hasattr(col, "upsert"):
                col.upsert(documents=texts, embeddings=embs, ids=ids, metadatas=metas)
            else:
                col.add(documents=texts, embeddings=embs, ids=ids, metadatas=metas)
            written += len(batch)
            _progress(progress_cb, "upsert_batch", written=written, total=total_chunks, batch_size=len(batch))
        
        # Update manifest for the entire flushed group
        manifest.update(manifest_updates_buffer)
        save_manifest(collection, manifest)
        
        stats["files_loaded"] += files_processed_in_buffer
        item_buffer.clear()
        manifest_updates_buffer.clear()
        files_processed_in_buffer = 0

    for idx, rel_path in enumerate(rel_paths, start=1):
        stats["files_seen"] += 1
        fp = root / rel_path
        try:
            items = _prepare_single_file(rel_path, root, collection, max_file_bytes)
            if not items:
                stats["files_skipped"] += 1
                continue
            
            _progress(
                progress_cb,
                "prepare_file",
                index=idx,
                total=len(rel_paths),
                path=rel_path,
                chunks_in_file=len(items),
            )
            
            total_chunks += len(items)
            stats["chunks_prepared"] += len(items)
            
            # Add to buffer
            item_buffer.extend(items)
            files_processed_in_buffer += 1
            
            # Record manifest update for this file
            try:
                stat = fp.stat()
                manifest_updates_buffer[rel_path] = {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
            except Exception:
                pass

            # Flush if buffer is large enough
            if len(item_buffer) >= target_buffer_size:
                _flush_buffer()

        except StopIngestion:
            # Graceful stop: flush whatever is in the buffer first
            _flush_buffer()
            save_pause_state(
                collection,
                files_done=stats["files_loaded"],
                files_total=len(rel_paths),
                chunks_written=written,
            )
            error_reporter.warn(
                stage="full_ingest_stop",
                symbol=rel_path,
                message=f"ingest paused before this file, {stats['files_loaded']}/{len(rel_paths)} done",
            )
            return {
                "mode": "stopped",
                "collection": collection,
                "root": str(root),
                "files_indexed": stats["files_loaded"],
                "files_remaining": len(rel_paths) - idx + 1,
                "prepared": stats,
                "written": written,
            }
        except Exception as exc:
            error_reporter.warn(
                stage="full_ingest_file",
                symbol=rel_path,
                message="file skipped during full ingest",
                exc=exc,
            )
            stats["files_skipped"] += 1
            continue

    # Final flush for any remaining items
    _flush_buffer()

    # Full ingest completed successfully — remove any stale pause marker
    clear_pause_state(collection)
    return {
        "mode": "full",
        "collection": collection,
        "root": str(root),
        "files_indexed": stats["files_loaded"],
        "prepared": stats,
        "written": written,
    }


def sync_project(
    root: Path,
    collection: str = "code",
    exts: set[str] | None = None,
    includes: List[str] | None = None,
    limit_files: int | None = None,
    max_file_bytes: int | None = None,
    force_full: bool = False,
    progress_cb: ProgressCb = None,
) -> Dict[str, Any]:
    exts = exts or set(_get_default_exts())
    manifest_entries = includes if includes is not None else load_index_resources(root)
    if includes is None and not manifest_entries:
        manifest_entries = _parse_allowlist_patterns(getattr(settings, "auto_ingest_allowlist", ""))
    _progress(
        progress_cb,
        "resource_manifest",
        source="explicit" if includes is not None else INDEX_RESOURCES_FILENAME,
        path=str(root / INDEX_RESOURCES_FILENAME),
        resources_count=len(manifest_entries),
        resources=manifest_entries,
    )
    current = snapshot_project(root, exts, includes=manifest_entries, limit=limit_files)
    previous = load_manifest(collection)
    pause_state = load_pause_state(collection)

    current_paths = set(current.keys())
    previous_paths = set(previous.keys())
    new_paths = sorted(current_paths - previous_paths)
    deleted_paths = sorted(previous_paths - current_paths)
    changed_paths = sorted(
        path for path in (current_paths & previous_paths)
        if current[path] != previous[path]
    )
    changed_total = len(new_paths) + len(changed_paths) + len(deleted_paths)
    previous_total = max(1, len(previous_paths))
    changed_ratio = changed_total / previous_total

    collection_count = get_collection(collection).count()

    _progress(
        progress_cb,
        "scan_complete",
        collection=collection,
        total_files=len(current),
        new_paths=len(new_paths),
        changed_paths=len(changed_paths),
        deleted_paths=len(deleted_paths),
        changed_ratio=changed_ratio,
        collection_count=collection_count,
        paused=bool(pause_state),
    )

    if force_full or not previous or collection_count == 0:
        reason = "force_full" if force_full else ("missing_manifest" if not previous else "empty_collection")
        clear_pause_state(collection)
        report = full_ingest(
            root=root,
            collection=collection,
            exts=exts,
            includes=manifest_entries,
            limit_files=limit_files,
            max_file_bytes=max_file_bytes,
            progress_cb=progress_cb,
        )
        report["reason"] = reason
        report["changes"] = {
            "new": len(new_paths),
            "changed": len(changed_paths),
            "deleted": len(deleted_paths),
            "ratio": changed_ratio,
        }
        return report

    if changed_total == 0:
        # If paused with nothing left to do — clean up the marker
        if pause_state:
            clear_pause_state(collection)
        return {
            "mode": "noop",
            "reason": "up_to_date",
            "collection": collection,
            "root": str(root),
            "changes": {
                "new": 0,
                "changed": 0,
                "deleted": 0,
                "ratio": 0.0,
            },
        }

    # Resume from pause: bypass the ratio threshold so we don't restart from
    # scratch. The "new" paths here are exactly the files that weren't processed
    # before the stop — we just continue incrementally.
    if pause_state:
        delete_report = delete_paths(collection, deleted_paths, progress_cb=progress_cb) if deleted_paths else {"deleted_paths": 0}
        ingest_report = ingest_paths(
            rel_paths=sorted(new_paths + changed_paths),
            root=root,
            collection=collection,
            replace_by_path=True,
            max_file_bytes=max_file_bytes,
            progress_cb=progress_cb,
        )
        save_manifest(collection, current)
        clear_pause_state(collection)
        ingest_report["reason"] = "resume"
        ingest_report["resumed_from"] = {
            "files_done": pause_state.get("files_done", 0),
            "files_total": pause_state.get("files_total", 0),
            "chunks_written": pause_state.get("chunks_written", 0),
        }
        ingest_report["deleted_paths"] = delete_report["deleted_paths"]
        ingest_report["changes"] = {
            "new": len(new_paths),
            "changed": len(changed_paths),
            "deleted": len(deleted_paths),
            "ratio": changed_ratio,
        }
        return ingest_report

    # Always use incremental updates for any changes, regardless of their volume.
    # Automatic full reindex (threshold) removed to prevent RAG blocking on large projects.

    delete_report = delete_paths(collection, deleted_paths, progress_cb=progress_cb) if deleted_paths else {"deleted_paths": 0}
    ingest_report = ingest_paths(
        rel_paths=sorted(new_paths + changed_paths),
        root=root,
        collection=collection,
        replace_by_path=True,
        max_file_bytes=max_file_bytes,
        progress_cb=progress_cb,
    )
    save_manifest(collection, current)
    ingest_report["reason"] = "incremental"
    ingest_report["deleted_paths"] = delete_report["deleted_paths"]
    ingest_report["changes"] = {
        "new": len(new_paths),
        "changed": len(changed_paths),
        "deleted": len(deleted_paths),
        "ratio": changed_ratio,
    }
    return ingest_report
