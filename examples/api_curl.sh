#!/bin/bash
# mcp-rag HTTP API — curl examples
#
# Prerequisites:
#   export PROJECT_KEY=<your-key>   # from .mcp-rag file in your project dir
#   bash rag_server/start_rag.sh
#   bash embed_server/start_embed.sh

RAG="http://127.0.0.1:8000"
KEY="${PROJECT_KEY:-your-project-key-here}"
H='-H "X-Project-Key: '"$KEY"'"'

# ── Health check (no key required) ───────────────────────────────────────────
echo "=== Health ==="
curl -s "$RAG/healthz" | python3 -m json.tool

# ── Register project ──────────────────────────────────────────────────────────
echo -e "\n=== Register project ==="
curl -s -X POST "$RAG/project/register" \
  -H "Content-Type: application/json" \
  -d '{"key": "'"$KEY"'", "name": "my-project", "project_path": "/path/to/my-project"}' \
  | python3 -m json.tool

# ── Scan project (preview files without indexing) ────────────────────────────
echo -e "\n=== Scan project ==="
curl -s -X POST "$RAG/rag/scan" \
  -H "Content-Type: application/json" \
  -H "X-Project-Key: $KEY" \
  -d '{"root": "/path/to/my-project", "limit_files": 1000}' \
  | python3 -m json.tool

# ── Start incremental ingest job ─────────────────────────────────────────────
echo -e "\n=== Start ingest ==="
JOB=$(curl -s -X POST "$RAG/rag/sync" \
  -H "Content-Type: application/json" \
  -H "X-Project-Key: $KEY" \
  -d '{"root": "/path/to/my-project", "collection": "code"}')
echo "$JOB" | python3 -m json.tool
JOB_ID=$(echo "$JOB" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# ── Poll job status ───────────────────────────────────────────────────────────
echo -e "\n=== Job status ($JOB_ID) ==="
curl -s "$RAG/rag/jobs/$JOB_ID" \
  -H "X-Project-Key: $KEY" \
  | python3 -m json.tool

# ── Vector search ─────────────────────────────────────────────────────────────
echo -e "\n=== Search (vector) ==="
curl -s "$RAG/query/?q=how+does+batching+work&k=5&collection=code" \
  -H "X-Project-Key: $KEY" \
  | python3 -m json.tool

# ── Hybrid search (BM25 + vector RRF) ────────────────────────────────────────
echo -e "\n=== Search (hybrid BM25+vector) ==="
curl -s "$RAG/query/?q=DynamicBatcher&k=5&collection=code&hybrid=true" \
  -H "X-Project-Key: $KEY" \
  | python3 -m json.tool

# ── embed_server direct (no key needed) ──────────────────────────────────────
echo -e "\n=== embed_server health ==="
curl -s "http://127.0.0.1:8001/healthz" | python3 -m json.tool

echo -e "\n=== Embed texts ==="
curl -s -X POST "http://127.0.0.1:8001/embed" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Semantic search"], "priority": 0}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'count={d[\"count\"]} elapsed={d[\"elapsed_ms\"]}ms dims={len(d[\"embeddings\"][0])}')"
