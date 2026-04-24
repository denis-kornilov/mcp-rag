"""Embedding client — distributes computation across embed_server workers via HTTP.

rag_server never loads a model. It splits text batches across one or more
embed_server instances (workers), calls them in parallel, and reassembles results
in original order.

Config (rag_server/.env):
  EMBED_SERVER_URL=http://127.0.0.1:8001              # single worker (default)
  EMBED_SERVER_URLS=http://h1:8001,http://h2:8001     # multiple workers (overrides above)
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

from .settings import settings

logger = logging.getLogger("rag_server.embeddings")


def _worker_urls() -> List[str]:
    raw = getattr(settings, "embed_server_urls", "").strip()
    if raw:
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    return [str(settings.embed_server_url).rstrip("/")]


def _call_worker(url: str, texts: List[str], priority: int = 10) -> List[List[float]]:
    try:
        resp = requests.post(f"{url}/embed", json={"texts": texts, "priority": priority}, timeout=120)
        resp.raise_for_status()
        return resp.json()["embeddings"]
    except requests.RequestException as exc:
        raise RuntimeError(f"embed_server unreachable at {url}: {exc}") from exc


def embed_texts(texts: List[str], priority: int = 10) -> List[List[float]]:
    if not texts:
        return []
    max_chars = int(settings.embed_max_chars)
    payload = [t[:max_chars] for t in texts]
    workers = _worker_urls()

    if len(workers) == 1:
        return _call_worker(workers[0], payload, priority)

    # Split payload across workers — each worker gets a contiguous slice
    n = len(workers)
    chunk_size = max(1, (len(payload) + n - 1) // n)
    slices = [payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)]

    results: List[List[List[float]] | None] = [None] * len(slices)
    with ThreadPoolExecutor(max_workers=len(slices)) as pool:
        futures = {
            pool.submit(_call_worker, workers[i % n], slices[i], priority): i
            for i in range(len(slices))
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()

    out: List[List[float]] = []
    for r in results:
        out.extend(r)  # type: ignore[arg-type]
    return out


def get_embedder_info() -> dict:
    workers = _worker_urls()
    infos = []
    for url in workers:
        try:
            resp = requests.get(f"{url}/healthz", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            infos.append({
                "url": url,
                "type": data.get("type", "remote"),
                "gpu": data.get("gpu", False),
                "backend": data.get("backend", "unknown"),
            })
        except Exception as exc:
            infos.append({"url": url, "type": "unreachable", "error": str(exc)})

    primary = infos[0] if infos else {}
    return {
        "type": primary.get("type", "remote"),
        "gpu": any(w.get("gpu", False) for w in infos),
        "dual_mode": len(workers) > 1,
        "cpu_fraction": None,
        "embed_server": workers[0] if len(workers) == 1 else workers,
        "backend": primary.get("backend", "unknown"),
        "workers": infos,
    }
