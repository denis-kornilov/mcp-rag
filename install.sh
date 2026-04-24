#!/bin/bash
# mcp-rag — root installer for the entire stack
#
# Runs installers for each subsystem sequentially:
#   1. embed_server — BGE-M3 model + ONNX Runtime (CPU/GPU choice interactively)
#   2. rag_server   — FastAPI + ChromaDB (CPU only)
#   3. mcp_server   — MCP gateway + client for rag_server
#
# Usage:
#   bash install.sh            # install all three subsystems
#   bash install.sh embed      # only embed_server
#   bash install.sh rag        # only rag_server
#   bash install.sh mcp        # only mcp_server
#   bash install.sh gpu        # only GPU driver (CUDA/ROCm)
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
echo -e "${B}║            mcp-rag — stack installer            ║${W}"
echo -e "${B}╚══════════════════════════════════════════════════╝${W}"
echo ""
echo "  Stack:"
echo "    embed_server → conda env: mcp-embed   (port 8001)"
echo "    rag_server   → conda env: mcp-rag     (port 8000)"
echo "    mcp_server   → conda env: mcp-gateway (port 8002)"
echo ""

TARGET="${1:-all}"

run_installer() {
    local name="$1"
    local script="$2"
    sep
    echo -e "${B}▶ $name${W}"
    sep
    if [ ! -f "$script" ]; then
        err "Installer not found: $script"
    fi
    bash "$script"
    ok "$name — installation complete"
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
        echo -e "  All three subsystems will be installed."
        echo ""
        read -rp "  Continue? [Y/n]: " CONFIRM
        CONFIRM="${CONFIRM:-Y}"
        if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        echo ""
        run_installer "embed_server" "$SCRIPT_DIR/embed_server/install.sh"
        run_installer "rag_server"   "$SCRIPT_DIR/rag_server/install.sh"
        run_installer "mcp_server"   "$SCRIPT_DIR/mcp_server/install.sh"
        ;;
    *)
        echo "Usage: bash install.sh [all|embed|rag|mcp|gpu]"
        exit 1
        ;;
esac

sep
echo ""
echo -e "${G}╔══════════════════════════════════════════════════╗${W}"
echo -e "${G}║              Installation complete               ║${W}"
echo -e "${G}╚══════════════════════════════════════════════════╝${W}"
echo ""
echo "  Start servers:"
echo "    bash embed_server/start_embed.sh   # terminal 1"
echo "    bash rag_server/start_rag.sh       # terminal 2"
echo "    bash mcp_server/start_mcp.sh       # terminal 3 (HTTP mode)"
echo ""
echo "  Or add mcp_server to IDE (stdio mode):"
echo "    ~/.claude.json → mcpServers → command: python mcp_server/gateway.py"
echo ""
