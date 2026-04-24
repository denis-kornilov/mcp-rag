"""Dynamic batching engine for embed_server.

Two independent batchers handle search and ingest traffic on separate ORT sessions:

  - search_batcher  → get_search_embedder()  (1 thread, SEARCH_THREADS)
  - ingest_batcher  → get_ingest_embedder()  (4 threads, ONNX_NUM_THREADS)

Each batcher has its own FIFO queue and worker thread. Because the sessions are
fully independent (GIL released during C++ ORT inference), search and ingest run
in true OS-level parallelism — no priority queue needed.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future
from typing import Callable, List, Tuple

import numpy as np

logger = logging.getLogger("embed_server.batcher")

MAX_BATCH_SIZE = int(os.environ.get("EMBED_MAX_BATCH_SIZE", "32"))
MAX_WAIT_MS    = float(os.environ.get("EMBED_MAX_WAIT_MS", "50"))


class DynamicBatcher:
    """Thread-safe FIFO batching queue for one embedding session."""

    def __init__(self, embedder_fn: Callable, name: str) -> None:
        self._embedder_fn = embedder_fn
        self._name = name
        # Each item: (texts: List[str], future: Future[np.ndarray])
        self._queue: List[Tuple[List[str], Future]] = []
        self._lock  = threading.Lock()
        self._event = threading.Event()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name=f"embed-{name}"
        )
        self._worker.start()
        logger.info(
            "[batcher:%s] started max_batch=%d max_wait_ms=%.0f",
            name, MAX_BATCH_SIZE, MAX_WAIT_MS,
        )

    def submit(self, texts: List[str]) -> Future:
        """Submit texts for embedding. Returns a Future[np.ndarray]."""
        fut: Future = Future()
        with self._lock:
            self._queue.append((texts, fut))
        self._event.set()
        return fut

    def _run(self) -> None:
        try:
            self._embedder_fn().encode(["warmup"], batch_size=1)
            logger.info("[batcher:%s] model warmed up", self._name)
        except Exception as exc:
            logger.error("[batcher:%s] warmup failed: %s", self._name, exc)

        while True:
            self._event.wait()
            self._event.clear()

            # Wait up to MAX_WAIT_MS to accumulate more items
            deadline = time.monotonic() + MAX_WAIT_MS / 1000.0
            while time.monotonic() < deadline:
                with self._lock:
                    total = sum(len(t) for t, _ in self._queue)
                if total >= MAX_BATCH_SIZE:
                    break
                time.sleep(0.005)

            with self._lock:
                batch_items: List[Tuple[List[str], Future]] = []
                collected = 0
                remaining: List[Tuple[List[str], Future]] = []
                for texts, fut in self._queue:
                    if collected + len(texts) <= MAX_BATCH_SIZE or not batch_items:
                        batch_items.append((texts, fut))
                        collected += len(texts)
                    else:
                        remaining.append((texts, fut))
                self._queue = remaining
                if remaining:
                    self._event.set()

            if not batch_items:
                continue

            all_texts: List[str] = []
            slices: List[Tuple[int, int]] = []
            for texts, _ in batch_items:
                start = len(all_texts)
                all_texts.extend(texts)
                slices.append((start, len(all_texts)))

            logger.debug(
                "[batcher:%s] processing batch size=%d requests=%d",
                self._name, len(all_texts), len(batch_items),
            )

            try:
                embedder = self._embedder_fn()
                vecs: np.ndarray = embedder.encode(all_texts, batch_size=len(all_texts))
                for (start, end), (_, fut) in zip(slices, batch_items):
                    fut.set_result(vecs[start:end])
            except Exception as exc:
                logger.error("[batcher:%s] inference failed: %s", self._name, exc, exc_info=True)
                for _, fut in batch_items:
                    if not fut.done():
                        fut.set_exception(exc)


# ─── Singletons ───────────────────────────────────────────────────────────────

_search_batcher: DynamicBatcher | None = None
_ingest_batcher: DynamicBatcher | None = None
_search_lock = threading.Lock()
_ingest_lock = threading.Lock()


def get_search_batcher() -> DynamicBatcher:
    global _search_batcher
    if _search_batcher is None:
        with _search_lock:
            if _search_batcher is None:
                from embed_server.model import get_search_embedder
                _search_batcher = DynamicBatcher(get_search_embedder, "search")
    return _search_batcher


def get_ingest_batcher() -> DynamicBatcher:
    global _ingest_batcher
    if _ingest_batcher is None:
        with _ingest_lock:
            if _ingest_batcher is None:
                from embed_server.model import get_ingest_embedder
                _ingest_batcher = DynamicBatcher(get_ingest_embedder, "ingest")
    return _ingest_batcher


def get_batcher() -> DynamicBatcher:
    """Backward-compatible alias — returns the ingest batcher."""
    return get_ingest_batcher()
