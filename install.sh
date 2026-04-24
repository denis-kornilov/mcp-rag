#!/bin/bash
# mcp-rag — корневой установщик всего стека
#
# Запускает установщики каждой подсистемы последовательно:
#   1. embed_server — модель BGE-M3 + ONNX Runtime (выбор CPU/GPU интерактивно)
#   2. rag_server   — FastAPI + ChromaDB (CPU only)
#   3. mcp_server   — MCP gateway + клиент для rag_server
#
# Использование:
#   bash install.sh            # установить все три подсистемы
#   bash install.sh embed      # только embed_server
#   bash install.sh rag        # только rag_server
#   bash install.sh mcp        # только mcp_server
#   bash install.sh gpu        # только GPU-драйвер (CUDA/ROCm)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; C='\033[0;36m'; W='\033[0m'
ok()   { echo -e "${G}  ✓ $*${W}"; }
warn() { echo -e "${Y}  ! $*${W}"; }
info() { echo -e "${C}  » $*${W}"; }
err()  { echo -e "${R}  ✗ $*${W}"; exit 1; }
sep()  { echo -e "${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${W}"; }

echo ""
echo -e "${B}╔══════════════════════════════════════════════════╗${W}"
echo -e "${B}║          mcp-rag — установщик стека             ║${W}"
echo -e "${B}╚══════════════════════════════════════════════════╝${W}"
echo ""
echo "  Стек:"
echo "    embed_server → conda env: mcp-embed   (порт 8001)"
echo "    rag_server   → conda env: mcp-rag     (порт 8000)"
echo "    mcp_server   → conda env: mcp-gateway (порт 8002)"
echo ""

TARGET="${1:-all}"

run_installer() {
    local name="$1"
    local script="$2"
    sep
    echo -e "${B}▶ $name${W}"
    sep
    if [ ! -f "$script" ]; then
        err "Установщик не найден: $script"
    fi
    bash "$script"
    ok "$name — установка завершена"
    echo ""
}

case "$TARGET" in
    embed)
        run_installer "embed_server" "$SCRIPT_DIR/embed_server/install.sh"
        ;;
    rag)
        run_installer "rag_server" "$SCRIPT_DIR/rag_server/install.sh"
        ;;
    mcp)
        run_installer "mcp_server" "$SCRIPT_DIR/mcp_server/install.sh"
        ;;
    gpu)
        run_installer "GPU drivers" "$SCRIPT_DIR/install_adds.sh"
        ;;
    all)
        echo -e "  Будут установлены все три подсистемы."
        echo ""
        read -rp "  Продолжить? [Y/n]: " CONFIRM
        CONFIRM="${CONFIRM:-Y}"
        if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
            echo "Отменено."
            exit 0
        fi
        echo ""
        run_installer "embed_server" "$SCRIPT_DIR/embed_server/install.sh"
        run_installer "rag_server"   "$SCRIPT_DIR/rag_server/install.sh"
        run_installer "mcp_server"   "$SCRIPT_DIR/mcp_server/install.sh"
        ;;
    *)
        echo "Использование: bash install.sh [all|embed|rag|mcp|gpu]"
        exit 1
        ;;
esac

sep
echo ""
echo -e "${G}╔══════════════════════════════════════════════════╗${W}"
echo -e "${G}║              Установка завершена                 ║${W}"
echo -e "${G}╚══════════════════════════════════════════════════╝${W}"
echo ""
echo "  Запуск серверов:"
echo "    bash embed_server/start_embed.sh   # терминал 1"
echo "    bash rag_server/start_rag.sh       # терминал 2"
echo "    bash mcp_server/start_mcp.sh       # терминал 3 (HTTP-режим)"
echo ""
echo "  Или добавить mcp_server в IDE (stdio-режим):"
echo "    ~/.claude.json → mcpServers → command: python mcp_server/gateway.py"
echo ""
