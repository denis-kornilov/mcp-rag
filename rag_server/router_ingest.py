from fastapi import APIRouter, HTTPException, Body
import uuid
import logging
from typing import Any, Dict, List
from .store import get_collection, reset_collection
from .embeddings import embed_texts
from .settings import settings
from .error_reporter import ErrorReporter

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger("rag_server.ingest")
error_reporter = ErrorReporter("rag_server.ingest")

@router.post("/texts")
def ingest_texts(texts: list[str] = Body(..., embed=True)):
    if not texts or not all(isinstance(t, str) for t in texts):
        raise HTTPException(status_code=400, detail='Body must be {"texts": ["..."]}')
    logger.info("ingest/texts: count=%d", len(texts))
    ids = [str(uuid.uuid4()) for _ in texts]
    embs = embed_texts(texts)
    col = get_collection()
    col.add(documents=texts, embeddings=embs, ids=ids)
    return {"ok": True, "count": len(texts)}


@router.post("/reset")
def ingest_reset():
    reset_collection()
    return {"ok": True, "reset": True}


@router.post("/items")
def ingest_items(payload: Dict[str, Any] = Body(...)):
    """Ingest items with metadata and optional collection.

    Payload: { items: [...], replace_by_path?: bool }
    Each item: { text: str, id?: str, metadata?: {path?: str, ...}, collection?: str }
    Groups by collection and inserts with embeddings (upsert if available).
    """
    items = payload.get("items") if isinstance(payload, dict) else None
    if items is None and isinstance(payload, list):
        items = payload
    if not isinstance(items, list) or not items:
        raise HTTPException(400, detail='Body must be {"items": [...]}')

    replace_by_path = bool(payload.get("replace_by_path", False)) if isinstance(payload, dict) else False

    # Group by collection name and collect paths for optional replacement
    by_col: Dict[str, List[Dict[str, Any]]] = {}
    paths_by_col: Dict[str, set] = {}
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("text"), str):
            raise HTTPException(400, detail="Each item must include text: str")
        col = it.get("collection") or "docs"
        by_col.setdefault(col, []).append(it)
        if replace_by_path:
            md = it.get("metadata") or {}
            p = md.get("path")
            if isinstance(p, str) and p:
                paths_by_col.setdefault(col, set()).add(p)

    total = 0
    out = {}
    logger.info("ingest/items: total_items=%d replace_by_path=%s groups=%d", len(items), replace_by_path, len(by_col))
    for col_name, rows in by_col.items():
        col = get_collection(col_name)
        # Optional: delete previous chunks for these paths to avoid stale segments
        if replace_by_path and col_name in paths_by_col:
            for p in paths_by_col[col_name]:
                try:
                    col.delete(where={"path": p})
                except Exception as exc:
                    error_reporter.warn(
                        stage="ingest_items_replace_by_path_delete",
                        symbol=p,
                        message=f"delete stale chunks failed collection={col_name}",
                        exc=exc,
                    )

        logger.info("ingest/items: collection=%s rows=%d", col_name, len(rows))
        # Process in sub-batches to provide progress logs and control memory
        sub = max(1, settings.embed_batch_size * 2)
        done = 0
        for i in range(0, len(rows), sub):
            chunk = rows[i : i + sub]
            texts = [r["text"] for r in chunk]
            ids = [r.get("id") or str(uuid.uuid4()) for r in chunk]
            metas = [r.get("metadata") or {} for r in chunk]
            try:
                embs = embed_texts(texts)
            except Exception as e:
                logger.exception("ingest/items: embedding failed at %d/%d in collection %s", i, len(rows), col_name)
                error_reporter.error(
                    stage="ingest_items_embed_batch",
                    symbol=col_name,
                    message=f"embedding failed at {i}/{len(rows)}",
                    exc=e,
                )
                raise HTTPException(status_code=503, detail=f"embedding failed: {e}")
            # Upsert to overwrite existing ids instead of duplicating
            if hasattr(col, "upsert"):
                col.upsert(documents=texts, embeddings=embs, ids=ids, metadatas=metas)
            else:
                col.add(documents=texts, embeddings=embs, ids=ids, metadatas=metas)
            done += len(chunk)
            logger.info("ingest/items: collection=%s progress %d/%d", col_name, done, len(rows))
        total += len(rows)
        out[col_name] = len(rows)

    return {"ok": True, "ingested": total, "by_collection": out, "replaced": {k: len(v) for k, v in paths_by_col.items()}}
