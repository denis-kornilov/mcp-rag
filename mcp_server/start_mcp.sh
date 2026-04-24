#!/bin/bash
# MCP server — runs independently.
# IDE connects to it via HTTP.
# Acts as a RAG client itself (connects to rag_server).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

export MCP_TRANSPORT="$(grep '^MCP_TRANSPORT=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'http')"
export MCP_HOST="$(grep '^MCP_HOST=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo '127.0.0.1')"
export MCP_PORT="$(grep '^MCP_PORT=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo '8002')"
export RAG_SERVER="$(grep '^RAG_SERVER=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'http://127.0.0.1:8000')"
export RAG_BACKEND="$(grep '^RAG_BACKEND=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'http')"
export RAG_CONNECT_POLL_S="$(grep '^RAG_CONNECT_POLL_S=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo '10')"

PYTHON="${PYTHON:-$(conda run -n core2 which python 2>/dev/null || which python3)}"

echo "[mcp] MCP server  : $MCP_TRANSPORT://$MCP_HOST:$MCP_PORT"
echo "[mcp] RAG client  : $RAG_SERVER"

export PYTHONPATH="$ROOT_DIR"
cd "$SCRIPT_DIR"
exec "$PYTHON" gateway.py
