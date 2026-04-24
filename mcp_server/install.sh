#!/bin/bash
# mcp_server (gateway) — интерактивный установщик
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="mcp-gateway"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; C='\033[0;36m'; W='\033[0m'
ok()   { echo -e "${G}  ✓ $*${W}"; }
warn() { echo -e "${Y}  ! $*${W}"; }
info() { echo -e "${C}  » $*${W}"; }
err()  { echo -e "${R}  ✗ $*${W}"; }

echo ""
echo -e "${B}╔══════════════════════════════════════════╗${W}"
echo -e "${B}║      mcp_server — установщик            ║${W}"
echo -e "${B}╚══════════════════════════════════════════╝${W}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 1. ПРОВЕРКА ТОГО ЧТО УЖЕ УСТАНОВЛЕНО
# ══════════════════════════════════════════════════════════════════════════════
echo -e "${B}[1/3] Проверяем установленное ПО...${W}"

if [ -d "$CONDA_DIR" ]; then
    ok "Conda: $("$CONDA_DIR/bin/conda" --version 2>/dev/null || echo '?')"
else
    warn "Conda не найдена — будет установлена"
fi

eval "$("$CONDA_DIR/bin/conda" shell.bash hook 2>/dev/null)" || true
if conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
    ok "Conda env '$ENV_NAME': $($CONDA_DIR/envs/$ENV_NAME/bin/python --version 2>/dev/null || echo '?')"
    ENV_EXISTS=true
    PY="$CONDA_DIR/envs/$ENV_NAME/bin/python"
    for pkg in mcp fastapi requests pydantic chromadb onnxruntime; do
        if "$PY" -c "import importlib; importlib.import_module('$pkg'.split('[')[0])" 2>/dev/null; then
            PKG_VER="$("$PY" -c "import importlib.metadata; print(importlib.metadata.version('$pkg'))" 2>/dev/null || echo '?')"
            ok "$pkg: $PKG_VER"
        else
            warn "$pkg: не установлен"
        fi
    done
else
    warn "Conda env '$ENV_NAME' не найдена"
    ENV_EXISTS=false
fi

# PYTHONPATH check
if echo "${PYTHONPATH:-}" | grep -q "$ROOT_DIR"; then
    ok "PYTHONPATH содержит $ROOT_DIR"
else
    warn "PYTHONPATH не содержит $ROOT_DIR (нужно для импорта rag_server.settings)"
fi

# Проверяем ~/.claude.json
if [ -f "$HOME/.claude.json" ]; then
    if grep -q "gateway.py" "$HOME/.claude.json" 2>/dev/null; then
        GW_PATH="$(python3 -c "import json; d=json.load(open('$HOME/.claude.json')); \
            [print(s.get('args',[''])[0]) for s in d.get('mcpServers',{}).values() \
            if 'gateway' in str(s.get('args',''))]" 2>/dev/null || echo '?')"
        ok "~/.claude.json: gateway.py → $GW_PATH"
        if [ "$GW_PATH" != "$SCRIPT_DIR/gateway.py" ]; then
            warn "Путь в .claude.json не совпадает с текущим: $SCRIPT_DIR/gateway.py"
        fi
    else
        warn "~/.claude.json не содержит ссылки на gateway.py"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# 2. КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[2/3] Конфигурация подключения...${W}"

# Читаем текущий RAG_SERVER из .env
CURRENT_RAG="$(grep '^RAG_SERVER=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo 'http://127.0.0.1:8000')"
echo ""
info "Текущий RAG_SERVER: $CURRENT_RAG"
read -rp "  RAG_SERVER (Enter — оставить '$CURRENT_RAG'): " NEW_RAG
NEW_RAG="${NEW_RAG:-$CURRENT_RAG}"
if [ "$NEW_RAG" != "$CURRENT_RAG" ]; then
    sed -i "s|^RAG_SERVER=.*|RAG_SERVER=$NEW_RAG|" "$SCRIPT_DIR/.env"
    ok "RAG_SERVER=$NEW_RAG"
else
    ok "RAG_SERVER=$CURRENT_RAG (без изменений)"
fi

# MCP транспорт
echo ""
echo "  Транспорт MCP (как IDE подключается к MCP серверу):"
echo "    1) http  — MCP сервер запускается независимо (start_mcp.sh), IDE подключается по HTTP  [по умолчанию]"
echo "    2) stdio — устаревший режим, не рекомендуется"
echo ""
read -rp "  Выбор [1/2, default=1]: " MCP_TRANSPORT_CHOICE
MCP_TRANSPORT_CHOICE="${MCP_TRANSPORT_CHOICE:-1}"
case "$MCP_TRANSPORT_CHOICE" in
    2)
        sed -i 's|^MCP_TRANSPORT=.*|# MCP_TRANSPORT=http|' "$SCRIPT_DIR/.env"
        SELECTED_TRANSPORT="stdio"
        ok "Транспорт: stdio"
        ;;
    *)
        SELECTED_TRANSPORT="http"
        ok "Транспорт: http (порт 8002)"
        ;;
esac

# ══════════════════════════════════════════════════════════════════════════════
# 3. УСТАНОВКА
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[3/3] Установка пакетов...${W}"
echo ""
read -rp "  Продолжить? [Y/n]: " CONFIRM
CONFIRM="${CONFIRM:-Y}"
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Отменено."
    exit 0
fi

# Miniconda
if [ ! -d "$CONDA_DIR" ]; then
    info "Устанавливаем Miniconda..."
    INST="/tmp/miniconda.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$INST"
    bash "$INST" -b -p "$CONDA_DIR" && rm "$INST"
    ok "Miniconda установлена"
fi
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
grep -q "miniconda3/bin/conda" ~/.bashrc 2>/dev/null \
    || echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc

# Conda env
if [ "$ENV_EXISTS" = false ]; then
    info "Создаём conda env '$ENV_NAME'..."
    conda env create -f "$SCRIPT_DIR/environment.yml"
    ok "Env создан"
fi
conda activate "$ENV_NAME"
pip install -q --upgrade pip

info "Устанавливаем пакеты..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"
ok "Все пакеты установлены"

# Инструкция по claude.json
PY_BIN="$CONDA_DIR/envs/$ENV_NAME/bin/python3"
MCP_HOST_VAL="$(grep '^MCP_HOST=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo '127.0.0.1')"
MCP_PORT_VAL="$(grep '^MCP_PORT=' "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 || echo '8002')"
echo ""
echo -e "${G}╔══════════════════════════════════════════╗${W}"
echo -e "${G}║           Установка завершена            ║${W}"
echo -e "${G}╚══════════════════════════════════════════╝${W}"
echo ""
echo "  Env       : $CONDA_DIR/envs/$ENV_NAME"
echo "  Транспорт : $SELECTED_TRANSPORT"
echo ""
if [ "$SELECTED_TRANSPORT" = "http" ]; then
    echo "  Запуск gateway: bash $SCRIPT_DIR/start_mcp.sh"
    echo ""
    echo "  ~/.claude.json → mcpServers:"
    echo '  "rag-ethalon": {'
    echo "    \"type\": \"http\","
    echo "    \"url\": \"http://$MCP_HOST_VAL:$MCP_PORT_VAL/mcp\""
    echo '  }'
else
    echo "  ~/.claude.json → mcpServers:"
    echo '  "rag-ethalon": {'
    echo '    "type": "stdio",'
    echo "    \"command\": \"$PY_BIN\","
    echo "    \"args\": [\"$SCRIPT_DIR/gateway.py\"],"
    echo "    \"env\": {\"PYTHONPATH\": \"$ROOT_DIR\"}"
    echo '  }'
fi
echo ""
