"""embed_server — CPU/GPU compute server for text embeddings.

Exposes a single endpoint:
  POST /embed   {"texts": ["...", ...], "priority": 0|10}
              →  {"embeddings": [[...], ...], "count": N, "elapsed_ms": N}

Requests are routed to one of two independent ORT sessions:
  priority == 0  → search_batcher  (SEARCH_THREADS cores,   1 by default)
  priority != 0  → ingest_batcher  (ONNX_NUM_THREADS cores, 4 by default)

Both sessions run in true OS-level parallelism (GIL released during ORT C++).

Start:
  uvicorn embed_server.main:app --host 0.0.0.0 --port 8001
Or via server.py:
  EMBED_SERVER_PORT=8001 python -m embed_server.server
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embed_server.batcher import get_ingest_batcher, get_search_batcher
from embed_server.model import embedder_info

logger = logging.getLogger("embed_server")

app = FastAPI(title="Embed Server", version="2.0")


class EmbedRequest(BaseModel):
    texts: List[str]
    priority: int = 10  # 0 = search session, anything else = ingest session


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    count: int
    elapsed_ms: float


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts must be non-empty")

    t0 = time.monotonic()
    batcher = get_search_batcher() if req.priority == 0 else get_ingest_batcher()
    fut = batcher.submit(req.texts)

    loop = asyncio.get_event_loop()
    try:
        vecs = await loop.run_in_executor(None, lambda: fut.result(timeout=120))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Embedding failed: {exc}") from exc

    elapsed_ms = (time.monotonic() - t0) * 1000
    return EmbedResponse(
        embeddings=vecs.tolist(),
        count=len(req.texts),
        elapsed_ms=round(elapsed_ms, 1),
    )


@app.get("/healthz")
def health() -> dict:
    try:
        info = embedder_info()
        return {"status": "ok", **info}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.on_event("startup")
async def _startup() -> None:
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, get_search_batcher)
    loop.run_in_executor(None, get_ingest_batcher)
    logger.info("[embed_server] startup — search + ingest batchers initialising in background")
