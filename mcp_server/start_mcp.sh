#!/bin/bash
# MCP server — runs independently.
# IDE connects to it via HTTP.
# Acts as a RAG client itself (connects to rag_server).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

TRANSPORT="$(grep '^MCP_TRANSPORT=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'http')"
HOST="$(grep '^MCP_HOST=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo '127.0.0.1')"
PORT="$(grep '^MCP_PORT=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo '8002')"
RAG="$(grep '^RAG_SERVER=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'http://127.0.0.1:8000')"

PYTHON="${PYTHON:-$(conda run -n mcp-gateway which python 2>/dev/null || which python3)}"

echo "[mcp] MCP server  : $TRANSPORT://$HOST:$PORT"
echo "[mcp] RAG client  : $RAG"

export PYTHONPATH="$ROOT_DIR"
cd "$SCRIPT_DIR"
exec "$PYTHON" gateway.py
