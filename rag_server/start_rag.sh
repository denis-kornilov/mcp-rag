#!/bin/bash
# rag_server — запуск (remote HTTP mode)
# Перед стартом проверяет и при необходимости корректирует .env.
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
echo -e "${B}║          rag_server — запуск            ║${W}"
echo -e "${B}╚══════════════════════════════════════════╝${W}"
echo ""

# ── 1. Проверка .env ──────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    err ".env не найден: $ENV_FILE — скопируйте из .env.example или запустите install.sh"
fi

_env() { grep "^$1=" "$ENV_FILE" 2>/dev/null | cut -d= -f2- | xargs; }

RAG_HOST="$(_env RAG_HOST)"
RAG_PORT="$(_env RAG_PORT)"
ENV_NAME="$(_env ENV_NAME)"
EMBED_URL="$(_env EMBED_SERVER_URL)"
DATA_ROOT="$(_env SERVER_DATA_ROOT)"
FS_WATCHER="$(_env FS_WATCHER_ENABLED)"

echo -e "${B}[1/3] Проверка конфигурации .env...${W}"

# RAG_HOST
RAG_HOST="${RAG_HOST:-0.0.0.0}"
if [ -z "$(_env RAG_HOST)" ]; then
    warn "RAG_HOST не задан — используется 0.0.0.0"
    echo "RAG_HOST=0.0.0.0" >> "$ENV_FILE"
fi
ok "RAG_HOST=$RAG_HOST"

# RAG_PORT
RAG_PORT="${RAG_PORT:-8000}"
if [ -z "$(_env RAG_PORT)" ]; then
    warn "RAG_PORT не задан — используется 8000"
    echo "RAG_PORT=8000" >> "$ENV_FILE"
fi
# Проверка конфликта с embed_server
if [ "$RAG_PORT" = "8001" ]; then
    err "RAG_PORT=8001 конфликтует с embed_server (порт 8001). Установите RAG_PORT=8000 в $ENV_FILE"
fi
ok "RAG_PORT=$RAG_PORT"

# ENV_NAME
ENV_NAME="${ENV_NAME:-mcp-rag}"
ok "ENV_NAME=$ENV_NAME"

# EMBED_SERVER_URL
if [ -z "$EMBED_URL" ]; then
    warn "EMBED_SERVER_URL не задан в $ENV_FILE"
    warn "По умолчанию: http://127.0.0.1:8001"
    echo "EMBED_SERVER_URL=http://127.0.0.1:8001" >> "$ENV_FILE"
    EMBED_URL="http://127.0.0.1:8001"
fi
ok "EMBED_SERVER_URL=$EMBED_URL"

# FS_WATCHER — предупреждение если включён
if [ "$FS_WATCHER" = "true" ]; then
    warn "FS_WATCHER_ENABLED=true — watchdog активен."
    warn "На больших проектах расходует inotify watches. Рекомендуется: FS_WATCHER_ENABLED=false"
fi

# SERVER_DATA_ROOT — создать если нет
DATA_ROOT="${DATA_ROOT:-./mcp_rag_projects}"
if [[ "$DATA_ROOT" != /* ]]; then
    DATA_ROOT="$ROOT_DIR/${DATA_ROOT#./}"
fi
if [ ! -d "$DATA_ROOT" ]; then
    mkdir -p "$DATA_ROOT"
    ok "Создан SERVER_DATA_ROOT: $DATA_ROOT"
else
    ok "SERVER_DATA_ROOT: $DATA_ROOT"
fi

# ── 2. Проверка доступности embed_server ─────────────────────────────────────
echo ""
echo -e "${B}[2/3] Проверка embed_server...${W}"
EMBED_HEALTH="${EMBED_URL%/}/healthz"
if curl -sf --max-time 3 "$EMBED_HEALTH" > /dev/null 2>&1; then
    EMBED_INFO="$(curl -sf --max-time 3 "$EMBED_HEALTH" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('type','?'), 'threads_ingest='+str(d.get('ingest_threads','?')), 'threads_search='+str(d.get('search_threads','?')))" 2>/dev/null || echo "ok")"
    ok "embed_server доступен: $EMBED_INFO"
else
    warn "embed_server недоступен по $EMBED_URL"
    warn "Убедитесь что embed_server запущен: bash embed_server/start_embed.sh"
    warn "rag_server стартует, но запросы эмбеддинга будут завершаться с ошибкой."
fi

# ── 3. Запуск ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${B}[3/3] Запуск rag_server...${W}"

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
PYTHON="${PYTHON:-$("$CONDA_DIR/bin/conda" run -n "$ENV_NAME" which python 2>/dev/null || which python3)}"

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    err "Python не найден в env '$ENV_NAME'. Запустите: bash $SCRIPT_DIR/install.sh"
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
