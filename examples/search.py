"""
Example: semantic search via rag_server HTTP API.

Requires:
    - rag_server running: bash rag_server/start_rag.sh
    - embed_server running: bash embed_server/start_embed.sh
    - project registered and indexed

Usage:
    python examples/search.py "how does the batcher work"
    python examples/search.py "project key isolation" --hybrid --rerank --k 8
"""

import argparse
import json
import sys

import requests

RAG_URL = "http://127.0.0.1:8000"
PROJECT_KEY = ""  # set your project key or read from .mcp-rag


def load_project_key(path: str = ".mcp-rag") -> str:
    try:
        with open(path) as f:
            data = json.load(f)
            return data.get("project_key", "")
    except FileNotFoundError:
        return ""


def search(query: str, k: int = 5, hybrid: bool = False, rerank: bool = False) -> None:
    key = PROJECT_KEY or load_project_key()
    if not key:
        print("ERROR: project key not found. Run mcp_server/gateway.py from your project dir first.")
        sys.exit(1)

    headers = {"X-Project-Key": key}
    params = {"q": query, "k": k, "collection": "code", "hybrid": str(hybrid).lower()}

    resp = requests.get(f"{RAG_URL}/query/", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    matches = resp.json()["matches"]

    if rerank and matches:
        resp2 = requests.post(
            f"{RAG_URL}/query/rerank",
            json={"query": query, "chunks": matches},
            headers=headers,
            timeout=30,
        )
        if resp2.ok:
            matches = resp2.json()["chunks"]

    print(f"\nQuery: {query!r}   hits={len(matches)}  hybrid={hybrid}  rerank={rerank}\n")
    for i, m in enumerate(matches, 1):
        score = m.get("rerank_score", m.get("rrf_score", m.get("score", 0)))
        path = (m.get("meta") or {}).get("path", "?")
        line = (m.get("meta") or {}).get("start_line", "")
        loc = f"{path}:{line}" if line else path
        print(f"  [{i}] score={score:.4f}  {loc}")
        print(f"      {m['doc'][:200].strip()!r}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Search query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hybrid", action="store_true", help="Enable BM25+vector fusion")
    ap.add_argument("--rerank", action="store_true", help="Cross-encoder reranking")
    ap.add_argument("--url", default=RAG_URL)
    args = ap.parse_args()
    RAG_URL = args.url
    search(args.query, k=args.k, hybrid=args.hybrid, rerank=args.rerank)
