import contextvars
import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

# Capture MCP_TRANSPORT from process env BEFORE loading .env so that
# auto-detection (stdio vs http) works when the IDE spawns us without setting it.
_PRE_DOTENV_TRANSPORT = os.environ.get("MCP_TRANSPORT")

# Load mcp_server/.env before rag_server.settings so gateway-specific
# vars (RAG_BACKEND, RAG_SERVER, MCP_TRANSPORT) override project-wide defaults.
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
    _gw_env = Path(__file__).resolve().parent / ".env"
    if _gw_env.exists():
        _load_dotenv(str(_gw_env), override=True)
except Exception:
    pass

from rag_server.settings import settings

_RAG_BACKEND = os.environ.get("RAG_BACKEND", "local").strip().lower()
_RAG_URL = os.environ.get("RAG_SERVER", settings.rag_server).rstrip("/")

# Project key resolved at startup — sent as X-Project-Key in every HTTP request to RAG server
_rag_project_key: str = ""
_rag_project_key_lock = threading.Lock()

def _set_rag_project_key(key: str) -> None:
    global _rag_project_key
    with _rag_project_key_lock:
        _rag_project_key = str(key or "").strip()

def _rag_http(method: str, url: str, **kwargs):
    """requests wrapper that injects X-Project-Key header into every RAG server call."""
    import requests as _requests  # noqa: PLC0415
    headers = dict(kwargs.pop("headers", {}) or {})
    with _rag_project_key_lock:
        key = _rag_project_key
    if key:
        headers["X-Project-Key"] = key
    return _requests.request(method, url, headers=headers, **kwargs)

# HTTP mode connection state — updated by background monitor thread
_rag_connected = False          # True when last healthz succeeded
_rag_connected_lock = threading.Lock()

def _set_rag_connected(state: bool) -> None:
    global _rag_connected
    with _rag_connected_lock:
        _rag_connected = state

def _is_rag_connected() -> bool:
    with _rag_connected_lock:
        return _rag_connected

if _RAG_BACKEND == "local":
    from rag_server.ingest_ops import StopIngestion
    from rag_server.ingest_ops import sync_project as local_sync_project
    from rag_server.ingest_ops import ingest_paths as local_ingest_paths
else:
    StopIngestion = Exception  # type: ignore[assignment,misc]
    local_sync_project = None  # type: ignore[assignment]
    local_ingest_paths = None  # type: ignore[assignment]


PROJECT_ROOT = Path(settings.project_root).resolve()

RAG_INSTRUCTIONS = """
You are MCP-RAG — a semantic search engine for code and documents.

## First-time setup

If the user has not configured MCP-RAG yet, ask them two questions:

**1. Mode — local or remote?**

- **Local (stdio)** — RAG runs on this machine, no server needed.
  Add to your agent's settings.json:
  ```json
  {
    "mcpServers": {
      "mcp-rag": {
        "command": "python",
        "args": ["/path/to/mcp_gateway.py"],
        "env": {"PROJECT_ROOT": "/path/to/your/project"}
      }
    }
  }
  ```

- **Remote (HTTP)** — RAG runs on a dedicated server (GPU, cloud, etc).
  First, get a project key from the server:
    POST http://<host>:<port>/project/register  →  {"project_key": "..."}
  Then add to settings.json:
  ```json
  {
    "mcpServers": {
      "mcp-rag": {
        "transport": "http",
        "url": "http://<host>:<port>/mcp",
        "headers": {"X-Project-Key": "<your-project-key>"}
      }
    }
  }
  ```

**2. After configuration** — call the `health` tool to verify the connection.

## Operating rules

1. INCREMENTAL ONLY: Always use incremental sync. Never force full re-index unless the collection is confirmed empty.
2. SCAN BEFORE INGEST: Always run `scan_project` first to show the user the scope.
3. HYBRID SEARCH: Pass `hybrid=true` for better results on specific symbol/variable names.
4. RERANKING: Pass `rerank_results=true` for higher precision (slower, uses cross-encoder).
5. BACKGROUND JOBS: Ingest runs in background. Poll `get_job_status` for progress.
"""

mcp = FastMCP("RAG Gateway", json_response=True, instructions=RAG_INSTRUCTIONS.strip())
logger = logging.getLogger("rag_server.mcp_gateway")

# In-process cache so the agent can compose search -> fetch_chunks without
# forcing changes to the existing HTTP RAG server contract.
_CHUNK_CACHE: Dict[str, Dict[str, Any]] = {}
_JOBS: Dict[str, Dict[str, Any]] = {}
_STOP_EVENTS: Dict[str, threading.Event] = {}
_RECENT_JOBS: "deque[str]" = deque(maxlen=200)
_PENDING_MUTATIONS: Dict[str, Dict[str, Any]] = {}
_ALLOWED_CONFIRM_ANSWERS = {"confirm", "yes", "do it", "ok"}
DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / "debug.log"


def _fmt_duration(seconds: float | None) -> str:
    """Format seconds into human-readable 1h 23m 45s."""
    if seconds is None:
        return "--"
    s = max(0, int(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class _IngestProgressTracker:
    """Tracks ingestion progress and computes rolling ETA for the LLM.

    Passed as ``progress_cb`` to sync_project / ingest_paths. Updates the
    corresponding _JOBS entry so get_job_status() can expose it.
    """

    _RATE_SAMPLES = 12  # rolling window for ETA

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id
        self._stop_event = _STOP_EVENTS.get(job_id)
        self._started_at = time.monotonic()
        self._phase = "init"
        self._phase_started_at = time.monotonic()

        self._files_total: int = 0
        self._files_done: int = 0
        self._chunks_total: int = 0
        self._chunks_written: int = 0

        # (monotonic_time, chunks_written) pairs for rolling rate
        self._rate_samples: List[tuple] = []

    def __call__(self, stage: str, payload: Dict[str, Any]) -> None:
        # Check stop signal at the start of each new file — clean checkpoint
        # because full_ingest saves manifest after completing the previous file.
        if stage == "prepare_file" and self._stop_event and self._stop_event.is_set():
            raise StopIngestion("stop requested by stop_job")
        now = time.monotonic()
        self._update_counters(stage, payload, now)
        snap = self._snapshot(stage, now)
        _update_job(self._job_id, message=f"ingest:{stage}", progress=snap)
        _debug_log(f"[ingest] job={self._job_id[:8]} {snap['summary']}")

    # ------------------------------------------------------------------ #

    def _update_counters(self, stage: str, payload: Dict[str, Any], now: float) -> None:
        if stage == "scan_complete":
            self._set_phase("scan_done", now)
            self._files_total = int(payload.get("total_files", 0) or 0)

        elif stage == "scan_plan_complete":
            self._set_phase("scan_done", now)
            self._files_total = int(payload.get("files_planned", 0) or 0)
            self._chunks_total = int(payload.get("chunks_total", 0) or 0)

        elif stage == "reset_collection":
            self._set_phase("reset", now)
            self._chunks_total = 0
            self._chunks_written = 0
            self._rate_samples.clear()

        elif stage == "prepare_file":
            if self._phase != "preparing":
                self._set_phase("preparing", now)
            self._files_done = int(payload.get("index", 0) or 0)
            self._files_total = max(self._files_total, int(payload.get("total", 0) or 0))

        elif stage == "upsert_batch":
            if self._phase != "embedding":
                self._set_phase("embedding", now)
                self._rate_samples.clear()
            self._chunks_written = int(payload.get("written", 0) or 0)
            self._chunks_total = max(self._chunks_total, int(payload.get("total", 0) or 0))
            self._rate_samples.append((now, self._chunks_written))
            if len(self._rate_samples) > self._RATE_SAMPLES:
                self._rate_samples.pop(0)

        elif stage == "delete_path":
            if self._phase != "deleting":
                self._set_phase("deleting", now)
            self._files_done = int(payload.get("index", 0) or 0)
            self._files_total = max(self._files_total, int(payload.get("total", 0) or 0))

    def _set_phase(self, phase: str, now: float) -> None:
        self._phase = phase
        self._phase_started_at = now

    def _snapshot(self, stage: str, now: float) -> Dict[str, Any]:
        elapsed_total = now - self._started_at
        phase_elapsed = now - self._phase_started_at

        rate: float | None = None
        eta_s: float | None = None
        percent: float | None = None

        if self._phase == "preparing" and self._files_total > 0 and phase_elapsed > 0.5:
            rate = self._files_done / phase_elapsed
            percent = self._files_done / self._files_total * 100.0
            if rate > 0:
                eta_s = (self._files_total - self._files_done) / rate

        elif self._phase == "embedding":
            if len(self._rate_samples) >= 2:
                t0, c0 = self._rate_samples[0]
                t1, c1 = self._rate_samples[-1]
                dt = t1 - t0
                if dt > 0.5:
                    rate = (c1 - c0) / dt
            if self._chunks_total > 0:
                percent = self._chunks_written / self._chunks_total * 100.0
                if rate and rate > 0:
                    eta_s = (self._chunks_total - self._chunks_written) / rate

        elif self._phase == "deleting" and self._files_total > 0 and phase_elapsed > 0.5:
            rate = self._files_done / phase_elapsed
            percent = self._files_done / self._files_total * 100.0
            if rate > 0:
                eta_s = (self._files_total - self._files_done) / rate

        summary = self._make_summary(percent, rate, eta_s, elapsed_total)
        return {
            "stage": stage,
            "phase": self._phase,
            "elapsed_s": round(elapsed_total, 1),
            "elapsed_human": _fmt_duration(elapsed_total),
            "files_total": self._files_total,
            "files_done": self._files_done,
            "chunks_total": self._chunks_total,
            "chunks_written": self._chunks_written,
            "percent": round(percent, 1) if percent is not None else None,
            "rate": round(rate, 2) if rate is not None else None,
            "eta_s": round(eta_s) if eta_s is not None else None,
            "eta_human": _fmt_duration(eta_s) if eta_s is not None else None,
            "summary": summary,
        }

    def _make_summary(self, percent, rate, eta_s, elapsed) -> str:
        phase = self._phase
        if phase == "preparing":
            label = f"preparing files {self._files_done}/{self._files_total}"
        elif phase == "embedding":
            label = f"vectorizing chunks {self._chunks_written}/{self._chunks_total}"
        elif phase == "deleting":
            label = f"deleting stale {self._files_done}/{self._files_total}"
        elif phase == "scan_done":
            label = f"scan complete: {self._files_total} files"
        elif phase == "reset":
            label = "resetting collection"
        else:
            label = phase

        parts = [label]
        if percent is not None:
            parts.append(f"({percent:.1f}%)")
        if rate is not None:
            unit = "chunks/s" if phase == "embedding" else "files/s"
            parts.append(f"@ {rate:.1f} {unit}")
        if eta_s is not None:
            parts.append(f"~{_fmt_duration(eta_s)} remaining")
        parts.append(f"[elapsed {_fmt_duration(elapsed)}]")
        return " | ".join(parts)


def _debug_log(message: str) -> None:
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


def _cache_chunk(match: Dict[str, Any], query: str, rank: int) -> Dict[str, Any]:
    payload = {
        "query": query,
        "rank": rank,
        "doc": match.get("doc", ""),
        "meta": match.get("meta") or {},
        "score": match.get("score"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    chunk_id = hashlib.sha1(raw).hexdigest()
    _CHUNK_CACHE[chunk_id] = payload
    return {
        "id": chunk_id,
        "score": payload["score"],
        "meta": payload["meta"],
        "preview": payload["doc"][:400],
    }


def _tokenize(text: str) -> set[str]:
    return {part for part in text.lower().replace("_", " ").split() if part}


def _new_job(action: str, args: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {
        "job_id": job_id,
        "action": action,
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "args": args,
        "progress": {},
        "message": "",
        "result": None,
        "error": None,
    }
    _STOP_EVENTS[job_id] = threading.Event()
    _RECENT_JOBS.appendleft(job_id)
    return job_id


def _update_job(job_id: str, *, status: str | None = None, message: str | None = None, progress: Dict[str, Any] | None = None, result: Any = None, error: Any = None) -> None:
    job = _JOBS[job_id]
    if status is not None:
        job["status"] = status
    if message is not None:
        job["message"] = message
    if progress is not None:
        job["progress"] = progress
    if result is not None:
        job["result"] = result
    if error is not None:
        job["error"] = error
    if status in {"completed", "error"}:
        job["finished_at"] = time.time()


def _job_view(job_id: str) -> Dict[str, Any]:
    return dict(_JOBS[job_id])


def _run_action(action: str, args: Dict[str, Any], fn: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
    job_id = _new_job(action, args)
    try:
        result = fn(job_id)
        _update_job(job_id, status="completed", result=result, message="completed")
        if isinstance(result, dict):
            return {"job_id": job_id, **result}
        return {"job_id": job_id, "result": result}
    except Exception as exc:
        logger.error(
            "[ERROR] symbol=- tf=- ts=- stage=mcp_gateway_run_action "
            "message=job_failed action=%s exc=%s:%s",
            action,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        _debug_log(
            f"[ERROR] symbol=- tf=- ts=- stage=mcp_gateway_run_action "
            f"message=job_failed action={action} exc={type(exc).__name__}:{exc}"
        )
        _update_job(
            job_id,
            status="error",
            message=str(exc),
            error={"message": str(exc), "traceback": traceback.format_exc(limit=5)},
        )
        raise


def _create_mutation_request(action: str, args: Dict[str, Any], summary: str) -> Dict[str, Any]:
    request_id = uuid.uuid4().hex
    payload = {
        "request_id": request_id,
        "status": "confirmation_required",
        "action": action,
        "args": args,
        "question": f"Confirm RAG mutation: {summary}",
        "allowed_answers": sorted(_ALLOWED_CONFIRM_ANSWERS),
        "created_at": time.time(),
    }
    _PENDING_MUTATIONS[request_id] = payload
    _debug_log(f"[rag-approval] requested action={action} request_id={request_id} args={json.dumps(args, ensure_ascii=False, sort_keys=True)}")
    return dict(payload)


def _require_mutation_confirmation(
    action: str,
    args: Dict[str, Any],
    summary: str,
    request_id: str | None,
    user_answer: str | None,
) -> Dict[str, Any] | None:
    _debug_log(
        f"[rag-approval] bypassed action={action} request_id={json.dumps(request_id, ensure_ascii=False)} "
        f"answer={json.dumps(user_answer, ensure_ascii=False)} summary={json.dumps(summary, ensure_ascii=False)}"
    )
    return None



@mcp.tool()
def health() -> Dict[str, Any]:
    """Check RAG health/status (local in-process or remote HTTP)."""
    def runner(job_id: str) -> Dict[str, Any]:
        if _RAG_BACKEND == "http":
            payload = _rag_http("GET", f"{_RAG_URL}/healthz", timeout=10).json()
            payload["mode"] = "http"
            payload["rag_server"] = _RAG_URL
        else:
            from rag_server.store import get_collection  # noqa: PLC0415
            from rag_server.embeddings import get_embedder_info  # noqa: PLC0415
            info = get_embedder_info()
            payload = {
                "status": "ok",
                "mode": "local",
                "embed_backend": settings.embed_backend,
                "embedder_type": info.get("type", "unknown"),
                "gpu_active": info.get("gpu", False),
                "dual_mode": info.get("dual_mode", False),
                "cpu_fraction": info.get("cpu_fraction"),
                "collection_docs": int(get_collection("code").count()),
            }
        _debug_log(
            f"[control] stage=mcp_health mode={payload.get('mode','?')} "
            f"embedder={payload.get('embedder_type','?')} gpu={payload.get('gpu_active',False)}"
        )
        return payload
    return _run_action("health", {}, runner)


@mcp.resource("rag://system/instructions")
def get_system_instructions() -> str:
    """Core architectural rules and optimization principles for this MCP RAG system."""
    return """# MCP RAG System Instructions

## Core Principles (Innate Intelligence)
1. **Incremental Only:** This system MUST always use incremental updates. Never return to full reindexing logic that blocks the UI/Agent for hours.
2. **Context Isolation:** Each project has its own database in `.mcp_rag/db` and its own settings in `.env` or `rag_index_resources.txt`.
3. **Scan Before Ingest:** Always use `scan_project` to preview the scope before running `ingest_project`.
4. **Hybrid Search Ready:** Prefer semantic search, but be aware that specific symbols might need lexical matching.

## Optimization Rules
- **Batching:** Embeddings are processed in batches (default: 8).
- **CPU Tuning:** ONNX threads are limited to prevent system overheating.
- **Filtering:** VCS (.git), caches (__pycache__), and build artifacts are ignored by default.

## Project Customization
- To include specific paths, create `rag_index_resources.txt` in the project root.
- To ignore files, use the standard `.gitignore`.
"""


@mcp.tool()
def search(
    query: str,
    top_k: int = 6,
    collection: str = "code",
    hybrid: bool = False,
    rerank_results: bool = False,
) -> Dict[str, Any]:
    """Search relevant chunks in the RAG server.

    Args:
        hybrid: Combine vector search with BM25 lexical search via RRF fusion.
                Better for specific symbol/variable names.
        rerank_results: Apply cross-encoder reranking after retrieval for higher precision.

    Before using, consult 'rag://system/instructions' for best practices.
    """
    def runner(job_id: str) -> Dict[str, Any]:
        _update_job(job_id, message="querying_rag", progress={"stage": "querying_rag"})
        if _RAG_BACKEND == "http":
            data = _rag_http("GET", f"{_RAG_URL}/query/",
                             params={"q": query, "k": top_k, "collection": collection, "hybrid": str(hybrid).lower()},
                             timeout=30).json()
        else:
            from rag_server.router_query import query as _query  # noqa: PLC0415
            data = _query(q=query, k=top_k, collection=collection, hybrid=hybrid)
        matches = data.get("matches") or []
        if rerank_results and matches and _RAG_BACKEND == "local":
            from rag_server.reranker import rerank as _rerank  # noqa: PLC0415
            matches = _rerank(query, matches)
        out = [_cache_chunk(match, query=query, rank=i) for i, match in enumerate(matches, start=1)]
        _debug_log(
            f"[control] stage=mcp_search collection={collection} "
            f"top_k={top_k} hybrid={hybrid} rerank={rerank_results} count={len(out)}"
        )
        return {
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "hybrid": hybrid,
            "reranked": rerank_results,
            "count": len(out),
            "matches": out,
        }
    return _run_action(
        "search",
        {"query": query, "top_k": top_k, "collection": collection,
         "hybrid": hybrid, "rerank_results": rerank_results},
        runner,
    )


@mcp.tool()
def fetch_chunks(ids: list) -> Dict[str, Any]:
    """Return full chunk text for ids previously returned by search()."""
    def runner(job_id: str) -> Dict[str, Any]:
        found = []
        missing = []
        for chunk_id in ids:
            item = _CHUNK_CACHE.get(chunk_id)
            if item is None:
                missing.append(chunk_id)
                continue
            found.append(
                {
                    "id": chunk_id,
                    "doc": item["doc"],
                    "meta": item["meta"],
                    "score": item["score"],
                    "rank": item["rank"],
                    "query": item["query"],
                }
            )
        return {"found": found, "missing": missing}
    return _run_action("fetch_chunks", {"ids": ids}, runner)


@mcp.tool()
def rerank(query: str, ids: list) -> Dict[str, Any]:
    """Rerank cached chunks by cross-encoder score (falls back to lexical overlap).

    Pass chunk IDs previously returned by search(). Cross-encoder gives significantly
    better precision than vector similarity for short queries.
    """
    def runner(job_id: str) -> Dict[str, Any]:
        chunks = []
        missing = []
        for chunk_id in ids:
            item = _CHUNK_CACHE.get(chunk_id)
            if item is None:
                missing.append(chunk_id)
            else:
                chunks.append({"_id": chunk_id, **item})

        if chunks:
            from rag_server.reranker import rerank as _rerank  # noqa: PLC0415
            chunks = _rerank(query, chunks)

        ranked = [
            {
                "id": c["_id"],
                "rerank_score": c.get("rerank_score"),
                "score": c.get("score"),
                "meta": c.get("meta"),
                "preview": c.get("doc", "")[:400],
            }
            for c in chunks
        ]
        return {"query": query, "ranked": ranked, "missing": missing}
    return _run_action("rerank", {"query": query, "ids": ids}, runner)



@mcp.tool()
def scan_project(root: str = "", limit_files: int = 5000) -> Dict[str, Any]:
    """Preview what files would be ingested from the project root — fast, no embedding.

    Call this before ingest_project to report scope to the user:
    - which root directory will be scanned
    - how many files were found and their breakdown by extension
    - which directories are automatically skipped (VCS, caches, venvs, etc.)
    - 20 sample paths
    - estimated MB

    If AUTO_INGEST_ALLOWLIST is set in .env only those sub-paths are scanned;
    otherwise the full project tree is scanned recursively.
    """
    target_root = (Path(root).resolve() if root else PROJECT_ROOT)

    def runner(job_id: str) -> Dict[str, Any]:
        if _RAG_BACKEND == "http":
            preview = _rag_http("POST", f"{_RAG_URL}/rag/scan",
                                json={"root": str(target_root), "limit_files": limit_files},
                                timeout=30).json()
        else:
            from rag_server.ingest_ops import scan_files_preview  # noqa: PLC0415
            preview = scan_files_preview(root=target_root, limit_files=limit_files)
        allowlist = preview.get("allowlist") or []
        scope = (
            f"full tree from {target_root}"
            if not allowlist
            else f"only: {', '.join(allowlist)} (AUTO_INGEST_ALLOWLIST)"
        )
        preview["scope_summary"] = (
            f"Root: {target_root} | "
            f"Scope: {scope} | "
            f"Files: {preview['files_found']} ({preview['total_mb']} MB)"
        )
        _debug_log(f"[control] stage=mcp_scan_project {preview['scope_summary']}")
        return preview

    return _run_action("scan_project", {"root": str(target_root)}, runner)


def _launch_ingest_thread(job_id: str, target_root: Path, collection: str, force_full: bool) -> None:
    """Spin up a daemon thread that runs sync_project and updates the job record.

    In local mode: runs sync_project in-process with progress callback.
    In http mode: delegates to remote rag_server /rag/sync and polls for completion.
    """
    ctx = contextvars.copy_context()

    def _run() -> None:
        ctx.run(_run_inner)

    def _run_inner() -> None:
        if _RAG_BACKEND == "http":
            _run_http()
        else:
            _run_local()

    def _run_http() -> None:
        try:
            r = _rag_http("POST", f"{_RAG_URL}/rag/sync", json={
                "root": str(target_root), "collection": collection, "force_full": force_full,
            }, timeout=30)
            r.raise_for_status()
            remote_job_id = r.json()["job_id"]
            _debug_log(f"[ingest] job={job_id[:8]} remote_job={remote_job_id} started on rag_server")
            # Poll remote job status and mirror progress locally
            while True:
                remote = _rag_http("GET", f"{_RAG_URL}/rag/jobs/{remote_job_id}", timeout=10).json()
                remote_status = remote.get("status", "running")
                _update_job(job_id, message=f"remote:{remote_status}",
                            progress=remote.get("progress") or {})
                if remote_status in {"completed", "error", "stopped"}:
                    break
                time.sleep(3)
            if remote_status == "completed":
                _update_job(job_id, status="completed", result=remote.get("result") or {},
                            message="completed")
                _debug_log(f"[ingest] job={job_id[:8]} remote completed")
            elif remote_status == "stopped":
                _update_job(job_id, status="stopped", message="stopped by request")
            else:
                err = remote.get("error") or {"message": remote.get("message", "unknown")}
                _update_job(job_id, status="error", message=err.get("message", ""),
                            error=err)
        except Exception as exc:
            _debug_log(f"[ERROR] ingest http job={job_id[:8]} failed exc={type(exc).__name__}:{exc}")
            _update_job(job_id, status="error", message=str(exc),
                        error={"message": str(exc), "traceback": traceback.format_exc(limit=5)})

    def _run_local() -> None:
        try:
            tracker = _IngestProgressTracker(job_id)
            result = local_sync_project(
                root=target_root,
                collection=collection,
                force_full=force_full,
                progress_cb=tracker,
            )
            _update_job(job_id, status="completed", result=result, message="completed")
            _debug_log(
                f"[ingest] job={job_id[:8]} completed "
                f"mode={result.get('mode')} written={result.get('written', 0)}"
            )
            try:
                from rag_server.hybrid_search import invalidate as _bm25_invalidate  # noqa: PLC0415
                _bm25_invalidate()
            except Exception:
                pass
        except StopIngestion:
            job = _JOBS.get(job_id, {})
            partial = job.get("progress", {})
            _update_job(
                job_id,
                status="stopped",
                message="stopped by request",
                result={
                    "mode": "stopped",
                    "chunks_written": partial.get("chunks_written", 0),
                    "files_done": partial.get("files_done", 0),
                },
            )
            _debug_log(f"[ingest] job={job_id[:8]} stopped gracefully")
        except Exception as exc:
            _debug_log(f"[ERROR] ingest job={job_id[:8]} failed exc={type(exc).__name__}:{exc}")
            _update_job(
                job_id,
                status="error",
                message=str(exc),
                error={"message": str(exc), "traceback": traceback.format_exc(limit=5)},
            )

    t = threading.Thread(target=_run, daemon=True, name=f"ingest-{job_id[:8]}")
    t.start()


@mcp.tool()
def ingest_project(collection: str = "code", root: str = "", force_full: bool = False) -> Dict[str, Any]:
    """Start project ingestion as a background job.
    
    ALWAYS call 'scan_project' first to confirm the file count and scope.
    Consult 'rag://system/instructions' for architectural constraints.
    """
    target_root = (Path(root).resolve() if root else PROJECT_ROOT)
    job_id = _new_job("ingest_project", {"collection": collection, "root": str(target_root), "force_full": force_full})
    _launch_ingest_thread(job_id, target_root, collection, force_full)
    return {
        "job_id": job_id,
        "status": "running",
        "collection": collection,
        "root": str(target_root),
        "force_full": force_full,
        "message": "Ingest started in background. Call get_job_status(job_id) every 30-60 seconds to track progress and ETA.",
        "poll_hint": "Full reindex of a large project may take 1-2 hours.",
    }


@mcp.tool()
def confirm_ingest_project(
    request_id: str,
    user_answer: str,
    collection: str = "code",
    root: str = "",
    force_full: bool = False,
) -> Dict[str, Any]:
    """Execute ingest_project only after raw user confirmation."""
    target_root = (Path(root).resolve() if root else PROJECT_ROOT)
    args = {"collection": collection, "root": str(target_root), "force_full": force_full}
    confirmation = _require_mutation_confirmation(
        "ingest_project",
        args,
        f"ingest_project collection={collection} root={target_root} force_full={force_full}",
        request_id,
        user_answer,
    )
    if confirmation is not None:
        return confirmation

    job_id = _new_job("ingest_project", args)
    _launch_ingest_thread(job_id, target_root, collection, force_full)
    return {
        "job_id": job_id,
        "status": "running",
        "collection": collection,
        "root": str(target_root),
        "force_full": force_full,
        "message": "Ingest started in background. Call get_job_status(job_id) to track progress.",
    }


def _launch_ingest_paths_thread(job_id: str, target_root: Path, collection: str, rel_paths: list) -> None:
    """Spin up a daemon thread that runs ingest_paths and updates the job record."""
    ctx = contextvars.copy_context()

    def _run() -> None:
        ctx.run(_run_inner)

    def _run_inner() -> None:
        try:
            tracker = _IngestProgressTracker(job_id)
            result = local_ingest_paths(
                rel_paths=rel_paths,
                root=target_root,
                collection=collection,
                progress_cb=tracker,
            )
            _update_job(job_id, status="completed", result=result, message="completed")
            _debug_log(f"[ingest] job={job_id[:8]} paths completed written={result.get('written', 0)}")
            try:
                from rag_server.hybrid_search import invalidate as _bm25_invalidate  # noqa: PLC0415
                _bm25_invalidate()
            except Exception:
                pass
        except StopIngestion:
            _update_job(job_id, status="stopped", message="stopped by request")
            _debug_log(f"[ingest] job={job_id[:8]} paths stopped gracefully")
        except Exception as exc:
            _debug_log(f"[ERROR] ingest_paths job={job_id[:8]} failed exc={type(exc).__name__}:{exc}")
            _update_job(
                job_id,
                status="error",
                message=str(exc),
                error={"message": str(exc), "traceback": traceback.format_exc(limit=5)},
            )

    t = threading.Thread(target=_run, daemon=True, name=f"ingest-paths-{job_id[:8]}")
    t.start()


@mcp.tool()
def ingest_paths(paths: list, collection: str = "code", root: str = "") -> Dict[str, Any]:
    """Incrementally ingest specific relative file paths as a background job.

    Returns a job_id immediately. Use get_job_status(job_id) to track progress.
    """
    target_root = (Path(root).resolve() if root else PROJECT_ROOT)
    job_id = _new_job("ingest_paths", {"paths": paths, "collection": collection, "root": str(target_root)})
    _launch_ingest_paths_thread(job_id, target_root, collection, paths)
    return {
        "job_id": job_id,
        "status": "running",
        "paths_count": len(paths),
        "collection": collection,
        "root": str(target_root),
        "message": "Ingest started in background. Call get_job_status(job_id) to track progress.",
    }


@mcp.tool()
def confirm_ingest_paths(
    request_id: str,
    user_answer: str,
    paths: list,
    collection: str = "code",
    root: str = "",
) -> Dict[str, Any]:
    """Execute ingest_paths only after raw user confirmation."""
    target_root = (Path(root).resolve() if root else PROJECT_ROOT)
    args = {"paths": paths, "collection": collection, "root": str(target_root)}
    confirmation = _require_mutation_confirmation(
        "ingest_paths",
        args,
        f"ingest_paths collection={collection} root={target_root} paths={len(paths)}",
        request_id,
        user_answer,
    )
    if confirmation is not None:
        return confirmation

    job_id = _new_job("ingest_paths", args)
    _launch_ingest_paths_thread(job_id, target_root, collection, paths)
    return {
        "job_id": job_id,
        "status": "running",
        "paths_count": len(paths),
        "collection": collection,
        "root": str(target_root),
        "message": "Ingest started in background. Call get_job_status(job_id) to track progress.",
    }


@mcp.tool()
def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get status and progress of a background job (e.g. ingestion).

    Returns phase, percent complete, rate (chunks/s), ETA, and a human_summary
    string ready to relay to the user.

    status values: 'running' | 'completed' | 'error'
    """
    if job_id not in _JOBS:
        raise ValueError(f"Unknown job id: {job_id}")
    job = _job_view(job_id)

    status = job.get("status", "unknown")
    action = job.get("action", "")
    progress = job.get("progress") or {}
    started_at = job.get("started_at") or time.time()

    lines: List[str] = [f"Job {job_id[:8]} | action={action} | status={status}"]

    if status == "running" and progress:
        summary = progress.get("summary")
        eta_human = progress.get("eta_human")
        elapsed_human = progress.get("elapsed_human", "")
        percent = progress.get("percent")
        if summary:
            lines.append(f"Progress: {summary}")
        if percent is not None:
            lines.append(f"Done: {percent:.1f}%")
        if eta_human and eta_human != "--":
            lines.append(f"Remaining: ~{eta_human}")
        if elapsed_human:
            lines.append(f"Elapsed: {elapsed_human}")
        lines.append("(keep polling every 30-60 seconds)")

    elif status == "completed":
        result = job.get("result") or {}
        finished_at = job.get("finished_at") or time.time()
        elapsed = finished_at - started_at
        mode = result.get("mode", "")
        written = result.get("written", 0)
        changes = result.get("changes") or {}
        lines.append(
            f"Completed in {_fmt_duration(elapsed)}: "
            f"mode={mode} written={written} chunks | "
            f"new={changes.get('new', 0)} changed={changes.get('changed', 0)} "
            f"deleted={changes.get('deleted', 0)}"
        )

    elif status == "error":
        error = job.get("error") or {}
        lines.append(f"Error: {error.get('message', job.get('message', ''))}")

    job["human_summary"] = "\n".join(lines)
    return job


@mcp.tool()
def list_recent_jobs(limit: int = 20) -> Dict[str, Any]:
    """List recent MCP actions and ingestion jobs."""
    job_ids = list(_RECENT_JOBS)[: max(1, limit)]
    return {"jobs": [_job_view(job_id) for job_id in job_ids]}


@mcp.tool()
def stop_job(job_id: str) -> Dict[str, Any]:
    """Gracefully stop a running ingest job (full or incremental).

    The stop signal is checked at the start of each file — so the current file
    finishes completely (its chunks are written and its manifest entry is saved)
    before the loop exits.  This means the index is always in a consistent state
    after stopping: the next sync_project run will continue from exactly where
    this one left off using incremental mode.

    Returns immediately with status='stop_requested'. Poll get_job_status(job_id)
    until status becomes 'stopped' (usually within seconds of the current file).
    """
    if job_id not in _JOBS:
        raise ValueError(f"Unknown job id: {job_id}")

    job = _JOBS[job_id]
    current_status = job.get("status", "unknown")

    if current_status != "running":
        return {
            "job_id": job_id,
            "status": current_status,
            "message": f"Job already finished with status '{current_status}' — no stop needed.",
        }

    event = _STOP_EVENTS.get(job_id)
    if event is None:
        return {
            "job_id": job_id,
            "status": "error",
            "message": "No stop_event for this job (legacy format). Re-run ingest.",
        }

    event.set()
    if _RAG_BACKEND == "http":
        try:
            _rag_http("POST", f"{_RAG_URL}/rag/jobs/{job_id}/stop", timeout=5)
        except Exception:
            pass
    _update_job(job_id, message="stop_requested")
    progress = job.get("progress") or {}
    _debug_log(f"[control] stage=stop_job job={job_id[:8]} chunks_written={progress.get('chunks_written', '?')}")

    return {
        "job_id": job_id,
        "status": "stop_requested",
        "message": (
            "Stop signal sent. Current file will finish and ingest will stop. "
            "Manifest already saved for all processed files — next run will continue from that point. "
            "Call get_job_status(job_id) to confirm status became 'stopped'."
        ),
        "chunks_written_so_far": progress.get("chunks_written", 0),
        "files_done_so_far": progress.get("files_done", 0),
    }


@mcp.tool()
def register_project(project_path: str = "") -> Dict[str, Any]:
    """Register a new project on the RAG server and get a project key.

    Call this once per project. Save the returned key in your agent's settings.json
    under env.PROJECT_KEY (stdio) or headers.X-Project-Key (HTTP).
    """
    def runner(job_id: str) -> Dict[str, Any]:
        from rag_server.project_manager import get_manager  # noqa: PLC0415
        entry = get_manager().register(hint=project_path)
        _debug_log(f"[control] stage=register_project key={entry['key'][:8]} name={entry['name']}")
        return {
            "project_key": entry["key"],
            "name": entry["name"],
            "chroma_path": entry["chroma_path"],
            "message": (
                "Project registered. Save your key — it cannot be recovered if lost.\n"
                "stdio:  add PROJECT_KEY=<key> to env in settings.json\n"
                "HTTP:   add X-Project-Key: <key> to headers in settings.json"
            ),
        }
    return _run_action("register_project", {"project_path": project_path}, runner)


@mcp.tool()
def project_status() -> Dict[str, Any]:
    """Show current project context (key, chroma_path, collection doc count)."""
    def runner(job_id: str) -> Dict[str, Any]:
        from rag_server.project_context import current_project  # noqa: PLC0415
        from rag_server.store import get_collection  # noqa: PLC0415
        proj = current_project()
        try:
            col = get_collection("code")
            doc_count = col.count()
        except Exception:
            doc_count = -1
        return {
            "project_key": proj["key"] or "(not set — using default paths)",
            "chroma_path": proj["chroma_path"] or settings.chroma_path,
            "project_root": proj["project_root"] or str(PROJECT_ROOT),
            "collection_docs": doc_count,
        }
    return _run_action("project_status", {}, runner)


@mcp.resource("rag://chunk/{chunk_id}")
def read_cached_chunk(chunk_id: str) -> str:
    """Read a cached chunk body by id after a search() call."""
    item = _CHUNK_CACHE.get(chunk_id)
    if item is None:
        raise ValueError(f"Unknown chunk id: {chunk_id}")
    return item["doc"]


@mcp.resource("project://file/{rel_path}")
def read_project_file(rel_path: str) -> str:
    """Read a source file from the configured project root."""
    candidate = (PROJECT_ROOT / rel_path).resolve()
    if PROJECT_ROOT not in candidate.parents and candidate != PROJECT_ROOT:
        raise ValueError("Path escapes project root")
    if not candidate.exists() or not candidate.is_file():
        raise ValueError(f"File not found: {rel_path}")
    return candidate.read_text(encoding="utf-8", errors="ignore")


_MCP_RAG_CONFIG = ".mcp-rag"


def _generate_sid() -> str:
    """Generate a cryptographically random, unique project session ID (UUID4 hex)."""
    return uuid.uuid4().hex


def _resolve_project_key() -> tuple:
    """Determine project key and root from CWD.

    1. Read CWD as project_root — fatal error + exit(1) if unreadable.
    2. Look for .mcp-rag in CWD with {"project_key": "..."}.
    3. If absent or key empty — generate UUID4 hex SID and write .mcp-rag.
       Fatal error + exit(1) if write fails.

    Returns (project_key: str, project_root: Path).
    """
    try:
        project_root = Path.cwd().resolve()
    except Exception as exc:
        msg = f"[FATAL] Cannot determine project root (CWD unreadable): {exc}"
        print(f"[mcp-rag] {msg}", file=sys.stderr)
        _debug_log(msg)
        sys.exit(1)

    if not project_root.exists() or not project_root.is_dir():
        msg = f"[FATAL] Project root does not exist or is not a directory: {project_root}"
        print(f"[mcp-rag] {msg}", file=sys.stderr)
        _debug_log(msg)
        sys.exit(1)

    mcp_rag_file = project_root / _MCP_RAG_CONFIG

    if mcp_rag_file.exists():
        try:
            data = json.loads(mcp_rag_file.read_text(encoding="utf-8"))
            key = str(data.get("project_key") or "").strip()
            if key:
                _debug_log(
                    f"[control] stage=project_key_loaded key={key[:8]}... root={project_root}"
                )
                return key, project_root
        except Exception as exc:
            _debug_log(f"[WARN] stage=project_key_file_read_error file={mcp_rag_file} exc={exc}")

    key = _generate_sid()
    try:
        mcp_rag_file.write_text(
            json.dumps(
                {"project_key": key, "project_root": str(project_root)},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        _debug_log(f"[control] stage=project_key_created key={key[:8]}... root={project_root}")
    except Exception as exc:
        msg = f"[FATAL] Cannot write project key to {mcp_rag_file}: {exc}"
        print(f"[mcp-rag] {msg}", file=sys.stderr)
        _debug_log(msg)
        sys.exit(1)

    return key, project_root


def _init_rag_components(project_key: str, project_root: Path) -> None:
    """Register project context and start background warm-up. Returns in < 50 ms.

    All heavy operations (ChromaDB open, embed_server start, HTTP health poll)
    run in a daemon thread so mcp.run() starts immediately after this returns.
    """
    _debug_log(f"[control] stage=rag_init_start key={project_key[:8]}... root={project_root}")

    # --- Local mode: register project with in-process manager (fast file I/O) ---
    if _RAG_BACKEND == "local":
        try:
            from rag_server.project_manager import get_manager  # noqa: PLC0415
            from rag_server.project_context import set_project  # noqa: PLC0415
            manager = get_manager()
            proj = manager.get(project_key)
            if proj is None:
                proj = manager.register(key=project_key, name=project_key, hint=str(project_root))
                _debug_log(
                    f"[control] stage=rag_init_project_registered key={project_key[:8]} "
                    f"chroma={proj['chroma_path']}"
                )
            set_project(
                chroma_path=proj["chroma_path"],
                project_root=proj.get("project_root") or str(project_root),
                key=project_key,
            )
            _debug_log(f"[control] stage=rag_init_project_ok key={project_key[:8]} chroma={proj['chroma_path']}")
        except Exception as exc:
            _debug_log(f"[ERROR] stage=rag_init_project_local exc={type(exc).__name__}:{exc}")

    # --- Background thread: heavy init that must not block mcp.run() ---
    if _RAG_BACKEND == "http":
        def _background_warmup() -> None:
            # 1. Register project with remote RAG server (retry until success)
            _poll_s = int(os.environ.get("RAG_CONNECT_POLL_S", "10"))
            _registered = False
            _was_connected = False

            while True:
                try:
                    if not _registered:
                        resp = _rag_http(
                            "POST", f"{_RAG_URL}/project/register",
                            json={"key": project_key, "project_path": str(project_root)},
                            timeout=5,
                        )
                        resp.raise_for_status()
                        _registered = True
                        _debug_log(f"[control] stage=rag_init_remote_project_ok key={project_key[:8]}")

                    _health_data = _rag_http("GET", f"{_RAG_URL}/healthz", timeout=5).json()
                    if not _was_connected:
                        _embed = _health_data.get("embed_server", {})
                        _debug_log(
                            f"[control] stage=rag_connected rag_server={_RAG_URL} "
                            f"status={_health_data.get('status')} "
                            f"embed_type={_embed.get('type', '?')} embed_gpu={_embed.get('gpu', False)}"
                        )
                        _was_connected = True
                    _set_rag_connected(True)
                except Exception as exc:
                    _set_rag_connected(False)
                    level = "ALERT" if _was_connected else "INFO"
                    _debug_log(
                        f"[{level}] stage=rag_unavailable url={_RAG_URL} "
                        f"exc={type(exc).__name__}:{exc} retry_in={_poll_s}s"
                    )
                    _was_connected = False
                time.sleep(_poll_s)
    else:
        def _background_warmup() -> None:
            # 1. Open ChromaDB (can take 1-3 s)
            try:
                from rag_server.store import _get_client, get_collection  # noqa: PLC0415
                _get_client()
                doc_count = get_collection("code").count()
                _debug_log(f"[control] stage=rag_init_chroma_ok collection=code docs={doc_count}")
            except Exception as exc:
                _debug_log(f"[ERROR] stage=rag_init_chroma_failed exc={type(exc).__name__}:{exc}")

            # 2. Start embed_server subprocess (30 s wait on cold start)
            try:
                from embed_server.lifecycle import ensure_running  # noqa: PLC0415
                ok = ensure_running(settings.embed_server_url)
                _debug_log(f"[control] stage=rag_init_embed_server ok={ok} url={settings.embed_server_url}")
            except Exception as exc:
                _debug_log(f"[ERROR] stage=rag_init_embed_server exc={type(exc).__name__}:{exc}")

    t = threading.Thread(target=_background_warmup, daemon=True, name="rag-warmup")
    t.start()

    _debug_log("[control] stage=rag_init_done")


def _make_project_key_middleware(app):
    """Pure ASGI middleware: reads X-Project-Key header, sets project context per-request."""
    async def middleware(scope, receive, send):
        if scope["type"] == "http":
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            key = headers.get(b"x-project-key", b"").decode("utf-8", errors="ignore").strip()
            if key:
                try:
                    from rag_server.project_manager import get_manager  # noqa: PLC0415
                    from rag_server.project_context import set_project  # noqa: PLC0415
                    proj = get_manager().get(key)
                    if proj:
                        set_project(
                            chroma_path=proj["chroma_path"],
                            project_root=proj["project_root"],
                            key=key,
                        )
                except Exception as exc:
                    _debug_log(f"[ERROR] stage=project_key_middleware exc={type(exc).__name__}:{exc}")
        await app(scope, receive, send)
    return middleware


def _register_project_rest_endpoint() -> None:
    """Add REST endpoint POST /project/register for getting a project key."""
    import json as _json  # noqa: PLC0415
    from starlette.requests import Request  # noqa: PLC0415
    from starlette.responses import JSONResponse  # noqa: PLC0415

    @mcp.custom_route("/project/register", methods=["POST"])
    async def project_register(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        hint = str(body.get("project_path", "")).strip()
        from rag_server.project_manager import get_manager  # noqa: PLC0415
        entry = get_manager().register(hint=hint)
        _debug_log(f"[control] stage=rest_register_project key={entry['key'][:8]}")
        return JSONResponse({
            "project_key": entry["key"],
            "name": entry["name"],
            "chroma_path": entry["chroma_path"],
        })

    @mcp.custom_route("/project/list", methods=["GET"])
    async def project_list(request: Request) -> JSONResponse:
        from rag_server.project_manager import get_manager  # noqa: PLC0415
        return JSONResponse({"projects": get_manager().list_all()})


def main() -> None:
    # Resolve project key and root from CWD — fatal error if CWD unreadable or .mcp-rag unwritable.
    project_key, project_root = _resolve_project_key()

    # Update module-level PROJECT_ROOT so all tool defaults point to the correct directory.
    global PROJECT_ROOT
    PROJECT_ROOT = project_root

    # Make project key available to all _rag_http calls immediately.
    _set_rag_project_key(project_key)

    # Use value captured before .env was loaded — only "explicit" if set by caller
    if _PRE_DOTENV_TRANSPORT:
        transport = _PRE_DOTENV_TRANSPORT
    else:
        # Auto-detect: IDE subprocess (stdin not a TTY) → stdio; standalone shell → http
        transport = "stdio" if not sys.stdin.isatty() else "http"

    _debug_log(
        f"[control] stage=mcp_gateway_boot transport={transport} "
        f"key={project_key[:8]}... root={project_root}"
    )

    _init_rag_components(project_key, project_root)

    try:
        if transport in ("http", "sse"):
            import uvicorn  # noqa: PLC0415
            import anyio  # noqa: PLC0415
            host = os.getenv("MCP_HOST", "127.0.0.1")
            port = int(os.getenv("MCP_PORT", "8000"))
            _register_project_rest_endpoint()
            starlette_app = mcp.streamable_http_app()
            app_with_middleware = _make_project_key_middleware(starlette_app)
            config = uvicorn.Config(
                app_with_middleware,
                host=host,
                port=port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            anyio.run(server.serve)
        else:
            mcp.run(transport="stdio")
    finally:
        _debug_log("[control] stage=mcp_gateway_shutdown_done")


if __name__ == "__main__":
    main()
