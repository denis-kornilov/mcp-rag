"""Hybrid search: in-memory BM25 + vector semantic, fused via Reciprocal Rank Fusion (RRF).

BM25 index is built lazily from all docs in the collection and cached in memory.
Cache invalidates when collection doc count changes (e.g. after ingest).
"""
from __future__ import annotations

import math
import re
import threading
from typing import Any, Dict, List, Tuple

from .error_reporter import ErrorReporter
from .store import get_collection

error_reporter = ErrorReporter("rag_server.hybrid_search")

_LOCK = threading.Lock()
_INDEX: "_BM25Index | None" = None
_CACHED_COUNT: int = -1


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_]\w*|[а-яёА-ЯЁ]+|\d+", text.lower())


class _BM25Index:
    K1 = 1.5
    B = 0.75

    def __init__(self, ids: List[str], docs: List[str]) -> None:
        self.ids = ids
        self._n = len(docs)
        tokenized = [_tokenize(d) for d in docs]
        df: Dict[str, int] = {}
        total_len = 0
        for toks in tokenized:
            total_len += len(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._avgdl = total_len / max(1, self._n)
        self._tokenized = tokenized
        self._df = df

    def top_k(self, query: str, k: int) -> List[Tuple[str, float]]:
        q_toks = set(_tokenize(query))
        scores = [0.0] * self._n
        for tok in q_toks:
            df = self._df.get(tok, 0)
            if not df:
                continue
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            for i, toks in enumerate(self._tokenized):
                tf = toks.count(tok)
                if not tf:
                    continue
                dl = len(toks)
                tf_norm = tf * (self.K1 + 1) / (
                    tf + self.K1 * (1 - self.B + self.B * dl / max(1, self._avgdl))
                )
                scores[i] += idf * tf_norm
        top = sorted(range(self._n), key=lambda i: -scores[i])[:k]
        return [(self.ids[i], scores[i]) for i in top if scores[i] > 0]


def invalidate() -> None:
    """Call after any ingest to force BM25 index rebuild on next query."""
    global _INDEX, _CACHED_COUNT
    with _LOCK:
        _INDEX = None
        _CACHED_COUNT = -1


def _get_index(collection: str) -> "_BM25Index":
    global _INDEX, _CACHED_COUNT
    col = get_collection(collection)
    count = col.count()
    with _LOCK:
        if _INDEX is None or count != _CACHED_COUNT:
            result = col.get(include=["documents"], limit=max(count, 1))
            ids: List[str] = result.get("ids") or []
            docs: List[str] = result.get("documents") or []
            _INDEX = _BM25Index(ids, docs)
            _CACHED_COUNT = count
        return _INDEX


def hybrid_query(
    q: str,
    k: int,
    collection: str,
    vector_matches: List[Dict[str, Any]],
    vector_ids: List[str],
) -> List[Dict[str, Any]]:
    """Fuse vector results + BM25 via Reciprocal Rank Fusion (k_rrf=60).

    Vector-only hits and BM25-only hits are both included in fusion.
    Falls back to vector-only results if BM25 fails.
    """
    K_RRF = 60
    rrf: Dict[str, float] = {}
    doc_by_id: Dict[str, Dict[str, Any]] = {}

    for rank, (match, doc_id) in enumerate(zip(vector_matches, vector_ids), start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (K_RRF + rank)
        doc_by_id[doc_id] = match

    try:
        index = _get_index(collection)
        for bm25_rank, (doc_id, _score) in enumerate(index.top_k(q, k * 3), start=1):
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (K_RRF + bm25_rank)
            if doc_id not in doc_by_id:
                col = get_collection(collection)
                fetched = col.get(ids=[doc_id], include=["documents", "metadatas"])
                fids: List[str] = fetched.get("ids") or []
                fdocs: List[str] = fetched.get("documents") or []
                fmetas: List[Any] = fetched.get("metadatas") or []
                if fids:
                    doc_by_id[doc_id] = {
                        "doc": fdocs[0] if fdocs else "",
                        "meta": fmetas[0] if fmetas else {},
                        "score": 0.0,
                    }
    except Exception as exc:
        error_reporter.warn(
            stage="hybrid_query_bm25",
            message="BM25 failed, falling back to vector-only",
            exc=exc,
        )
        return vector_matches[:k]

    merged = sorted(rrf.items(), key=lambda x: -x[1])[:k]
    result = []
    for doc_id, rrf_score in merged:
        if doc_id in doc_by_id:
            item = dict(doc_by_id[doc_id])
            item["rrf_score"] = round(rrf_score, 6)
            result.append(item)
    return result
