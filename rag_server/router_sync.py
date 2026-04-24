"""Sync/scan/jobs REST endpoints for remote rag_server mode.

When mcp_rag_server runs with RAG_BACKEND=http, it forwards ingest_project and
scan_project calls to these endpoints on the rag_server.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException, Body

from .settings import settings


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"

router = APIRouter(prefix="/rag", tags=["sync"])

_JOBS: Dict[str, Dict[str, Any]] = {}
_STOP_EVENTS: Dict[str, threading.Event] = {}


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
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    job = _JOBS[job_id]
    for k, v in kwargs.items():
        if v is not None:
            job[k] = v
    if job.get("status") in {"completed", "error", "stopped"}:
        job["finished_at"] = job.get("finished_at") or time.time()


@router.post("/scan")
def scan(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Preview files that would be ingested — no embedding."""
    from .ingest_ops import scan_files_preview  # noqa: PLC0415
    root_str = body.get("root") or settings.project_root
    limit = int(body.get("limit_files") or 5000)
    root = Path(root_str).resolve()
    result = scan_files_preview(root=root, limit_files=limit)
    result["root"] = str(root)
    return result


@router.post("/sync")
def sync(body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """Start incremental project sync as a background job. Returns job_id."""
    from .ingest_ops import sync_project, StopIngestion  # noqa: PLC0415

    root_str = body.get("root") or settings.project_root
    collection = str(body.get("collection") or "code")
    force_full = bool(body.get("force_full") or False)
    root = Path(root_str).resolve()

    job_id = _new_job("sync", {"root": str(root), "collection": collection, "force_full": force_full})
    stop_event = _STOP_EVENTS[job_id]

    def _run() -> None:
        try:
            _rate_samples: List[Tuple[float, int]] = []
            _started = time.monotonic()

            def _progress(stage: str, payload: Dict[str, Any]) -> None:
                if stage == "prepare_file" and stop_event.is_set():
                    raise StopIngestion("stop requested")

                now = time.monotonic()
                elapsed = now - _started
                extra: Dict[str, Any] = {
                    "elapsed_s": round(elapsed),
                    "elapsed_human": _fmt_duration(elapsed),
                }

                if stage == "upsert_batch":
                    written = payload.get("written", 0)
                    total = payload.get("total", 0)
                    _rate_samples.append((now, written))
                    if len(_rate_samples) > 10:
                        _rate_samples.pop(0)
                    rate = None
                    eta_s = None
                    if len(_rate_samples) >= 2 and total:
                        t0, c0 = _rate_samples[0]
                        t1, c1 = _rate_samples[-1]
                        dt = t1 - t0
                        if dt > 0:
                            rate = (c1 - c0) / dt
                            if rate > 0:
                                eta_s = (total - written) / rate
                    if total:
                        extra["percent"] = round(written / total * 100, 1)
                    extra["summary"] = f"upsert {written}/{total} chunks"
                    if rate is not None:
                        extra["rate"] = round(rate, 2)
                    if eta_s is not None:
                        extra["eta_s"] = round(eta_s)
                        extra["eta_human"] = _fmt_duration(eta_s)

                elif stage == "prepare_file":
                    idx = payload.get("index", 0)
                    total = payload.get("total", 0)
                    if total:
                        extra["percent"] = round(idx / total * 100, 1)
                    extra["summary"] = f"preparing {idx}/{total} files"

                elif stage == "scan_complete":
                    new_c = payload.get("new_paths", 0)
                    chg_c = payload.get("changed_paths", 0)
                    del_c = payload.get("deleted_paths", 0)
                    extra["summary"] = f"new={new_c} changed={chg_c} deleted={del_c}"

                _update_job(job_id, progress={"stage": stage, **payload, **extra})

            result = sync_project(
                root=root,
                collection=collection,
                force_full=force_full,
                progress_cb=_progress,
            )
            _update_job(job_id, status="completed", result=result, message="completed")
            try:
                from .hybrid_search import invalidate  # noqa: PLC0415
                invalidate()
            except Exception:
                pass
        except StopIngestion:
            _update_job(job_id, status="stopped", message="stopped by request")
        except Exception as exc:
            _update_job(
                job_id,
                status="error",
                message=str(exc),
                error={"message": str(exc), "traceback": traceback.format_exc(limit=5)},
            )

    threading.Thread(target=_run, daemon=True, name=f"sync-{job_id[:8]}").start()
    return {"job_id": job_id, "status": "running", "root": str(root), "collection": collection}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    """Poll background job status."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return dict(_JOBS[job_id])


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str) -> Dict[str, Any]:
    """Signal a running sync job to stop gracefully."""
    if job_id not in _JOBS:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    ev = _STOP_EVENTS.get(job_id)
    if ev:
        ev.set()
    _update_job(job_id, message="stop_requested")
    return {"job_id": job_id, "status": "stop_requested"}
