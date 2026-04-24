#!/bin/bash
# rag_server — start (remote HTTP mode)
# Checks and corrects .env before starting if necessary.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$SCRIPT_DIR/.env"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; C='\033[0;36m'; W='\033[0m'
ok()   { echo -e "${G}  ✓ $*${W}"; }
warn() { echo -e "${Y}  ! $*${W}"; }
info() { echo -e "${C}  » $*${W}"; }
err()  { echo -e "${R}  ✗ $*${W}"; exit 1; }

echo ""
echo -e "${B}╔══════════════════════════════════════════╗${W}"
echo -e "${B}║          rag_server — start             ║${W}"
echo -e "${B}╚══════════════════════════════════════════╝${W}"
echo ""

# ── 1. Check .env ──────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    err ".env not found: $ENV_FILE — copy from .env.example or run install.sh"
fi

_env() { grep "^$1=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- | xargs; }

RAG_HOST="$(_env RAG_HOST)"
RAG_PORT="$(_env RAG_PORT)"
ENV_NAME="$(_env ENV_NAME)"
EMBED_URL="$(_env EMBED_SERVER_URL)"
DATA_ROOT="$(_env SERVER_DATA_ROOT)"
FS_WATCHER="$(_env FS_WATCHER_ENABLED)"

echo -e "${B}[1/3] Checking .env configuration...${W}"

# RAG_HOST
RAG_HOST="${RAG_HOST:-0.0.0.0}"
if [ -z "$(_env RAG_HOST)" ]; then
    warn "RAG_HOST not set — using 0.0.0.0"
    echo "RAG_HOST=0.0.0.0" >> "$ENV_FILE"
fi
ok "RAG_HOST=$RAG_HOST"

# RAG_PORT
RAG_PORT="${RAG_PORT:-8000}"
if [ -z "$(_env RAG_PORT)" ]; then
    warn "RAG_PORT not set — using 8000"
    echo "RAG_PORT=8000" >> "$ENV_FILE"
fi
# Check conflict with embed_server
if [ "$RAG_PORT" = "8001" ]; then
    err "RAG_PORT=8001 conflicts with embed_server (port 8001). Set RAG_PORT=8000 in $ENV_FILE"
fi
ok "RAG_PORT=$RAG_PORT"

# ENV_NAME
ENV_NAME="${ENV_NAME:-mcp-rag}"
ok "ENV_NAME=$ENV_NAME"

# EMBED_SERVER_URL
if [ -z "$EMBED_URL" ]; then
    warn "EMBED_SERVER_URL not set in $ENV_FILE"
    warn "Default: http://127.0.0.1:8001"
    echo "EMBED_SERVER_URL=http://127.0.0.1:8001" >> "$ENV_FILE"
    EMBED_URL="http://127.0.0.1:8001"
fi
ok "EMBED_SERVER_URL=$EMBED_URL"

# FS_WATCHER — warning if enabled
if [ "$FS_WATCHER" = "true" ]; then
    warn "FS_WATCHER_ENABLED=true — watchdog is active."
    warn "Consumes inotify watches on large projects. Recommended: FS_WATCHER_ENABLED=false"
fi

# SERVER_DATA_ROOT — create if missing
DATA_ROOT="${DATA_ROOT:-./mcp_rag_projects}"
if [[ "$DATA_ROOT" != /* ]]; then
    DATA_ROOT="$ROOT_DIR/${DATA_ROOT#./}"
fi
if [ ! -d "$DATA_ROOT" ]; then
    mkdir -p "$DATA_ROOT"
    ok "Created SERVER_DATA_ROOT: $DATA_ROOT"
else
    ok "SERVER_DATA_ROOT: $DATA_ROOT"
fi

# ── 2. Check embed_server availability ─────────────────────────────────────
echo ""
echo -e "${B}[2/3] Checking embed_server availability...${W}"
EMBED_HEALTH="${EMBED_URL%/}/healthz"
if curl -sf --max-time 3 "$EMBED_HEALTH" > /dev/null 2>&1; then
    EMBED_INFO="$(curl -sf --max-time 3 "$EMBED_HEALTH" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('type','?'), 'threads_ingest='+str(d.get('ingest_threads','?')), 'threads_search='+str(d.get('search_threads','?')))" 2>/dev/null || echo "ok")"
    ok "embed_server available: $EMBED_INFO"
else
    warn "embed_server unavailable at $EMBED_URL"
    warn "Make sure embed_server is running: bash embed_server/start_embed.sh"
    warn "rag_server starting, but embedding requests will fail."
fi

# ── 3. Start ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${B}[3/3] Starting rag_server...${W}"

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
PYTHON="${PYTHON:-$("$CONDA_DIR/bin/conda" run -n "$ENV_NAME" which python 2>/dev/null || which python3)}"

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    err "Python not found in env '$ENV_NAME'. Run: bash $SCRIPT_DIR/install.sh"
fi

export PYTHONPATH="$ROOT_DIR"

echo ""
info "workdir : $ROOT_DIR"
info "listen  : $RAG_HOST:$RAG_PORT"
info "python  : $PYTHON"
info "data    : $DATA_ROOT"
echo ""

cd "$ROOT_DIR"
exec "$PYTHON" server.py --host "$RAG_HOST" --port "$RAG_PORT"
