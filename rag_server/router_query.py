from fastapi import APIRouter, Query
from typing import Any, Dict, List

from .embeddings import embed_texts
from .store import get_collection

router = APIRouter(prefix="/query", tags=["query"])


@router.get("/")
def query(
    q: str = Query(..., min_length=1),
    k: int = 4,
    collection: str = "docs",
    hybrid: bool = False,
) -> Dict[str, Any]:
    col = get_collection(collection)
    q_emb = embed_texts([q], priority=0)
    res = col.query(query_embeddings=q_emb, n_results=k)
    docs: List[List[str]] = res.get("documents") or [[]]
    dists: List[List[float]] = res.get("distances") or [[]]
    metas: List[List[Any]] = res.get("metadatas") or [[]]
    ids: List[List[str]] = res.get("ids") or [[]]

    items: List[Dict[str, Any]] = []
    for doc, dist, meta in zip(docs[0], dists[0], metas[0]):
        score = max(0.0, 1.0 - float(dist))
        items.append({"doc": doc, "score": score, "meta": meta})

    vec_ids: List[str] = ids[0] if ids else []

    if hybrid:
        from .hybrid_search import hybrid_query  # noqa: PLC0415
        items = hybrid_query(q, k, collection, items, vec_ids)

    return {"matches": items, "vec_ids": vec_ids}
