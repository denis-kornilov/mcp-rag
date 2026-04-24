"""Cross-encoder reranker using sentence-transformers CrossEncoder.

Loaded lazily on first call, cached as singleton.
Falls back to original order if model unavailable or inference fails.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

from .error_reporter import ErrorReporter
from .settings import settings

logger = logging.getLogger("rag_server.reranker")
error_reporter = ErrorReporter("rag_server.reranker")

_LOCK = threading.Lock()
_RERANKER = None
_INIT_ATTEMPTED = False

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_reranker():
    global _RERANKER, _INIT_ATTEMPTED
    if _INIT_ATTEMPTED:
        return _RERANKER
    with _LOCK:
        if _INIT_ATTEMPTED:
            return _RERANKER
        _INIT_ATTEMPTED = True
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            model_id = str(getattr(settings, "reranker_model", DEFAULT_MODEL))
            logger.info("[reranker] Loading CrossEncoder %s...", model_id)
            _RERANKER = CrossEncoder(model_id, max_length=512)
            logger.info("[reranker] CrossEncoder ready: %s", model_id)
        except Exception as exc:
            error_reporter.warn(
                stage="reranker_init",
                message="CrossEncoder unavailable, rerank will use lexical fallback",
                exc=exc,
            )
            _RERANKER = None
    return _RERANKER


def rerank(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank chunks by cross-encoder score. Falls back to lexical overlap on error."""
    if not chunks:
        return chunks

    reranker = _get_reranker()
    if reranker is not None:
        try:
            pairs = [(query, c["doc"]) for c in chunks]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(chunks, scores), key=lambda x: -float(x[1]))
            return [dict(c, rerank_score=round(float(s), 4)) for c, s in ranked]
        except Exception as exc:
            error_reporter.warn(
                stage="reranker_predict",
                message="CrossEncoder predict failed, using lexical fallback",
                exc=exc,
            )

    # Lexical overlap fallback
    q_tokens = {t for t in query.lower().split() if t}
    def _overlap(c: Dict[str, Any]) -> int:
        return len(q_tokens & {t for t in c.get("doc", "").lower().split() if t})
    return sorted(chunks, key=lambda c: -_overlap(c))
