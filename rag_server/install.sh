#!/bin/bash
# rag_server — interactive installer
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="mcp-rag"

R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; C='\033[0;36m'; W='\033[0m'
ok()   { echo -e "${G}  ✓ $*${W}"; }
warn() { echo -e "${Y}  ! $*${W}"; }
info() { echo -e "${C}  » $*${W}"; }
err()  { echo -e "${R}  ✗ $*${W}"; }

echo ""
echo -e "${B}╔══════════════════════════════════════════╗${W}"
echo -e "${B}║        rag_server — installer           ║${W}"
echo -e "${B}╚══════════════════════════════════════════╝${W}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHECK INSTALLED SOFTWARE
# ══════════════════════════════════════════════════════════════════════════════
echo -e "${B}[1/3] Checking installed software...${W}"

if [ -d "$CONDA_DIR" ]; then
    ok "Conda: $("$CONDA_DIR/bin/conda" --version 2>/dev/null || echo '?')"
else
    warn "Conda not found — will be installed"
fi

eval "$("$CONDA_DIR/bin/conda" shell.bash hook 2>/dev/null)" || true
if conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
    ok "Conda env '$ENV_NAME': $($CONDA_DIR/envs/$ENV_NAME/bin/python --version 2>/dev/null || echo '?')"
    ENV_EXISTS=true
    PY="$CONDA_DIR/envs/$ENV_NAME/bin/python"
    for pkg in fastapi uvicorn chromadb watchdog pydantic-settings pathspec; do
        if "$PY" -c "import importlib; importlib.import_module('$pkg'.split('[')[0])" 2>/dev/null; then
            PKG_VER="$("$PY" -c "import importlib.metadata; print(importlib.metadata.version('$pkg'))" 2>/dev/null || echo '?')"
            ok "$pkg: $PKG_VER"
        else
            warn "$pkg: not installed"
        fi
    done
else
    warn "Conda env '$ENV_NAME' not found"
    ENV_EXISTS=false
fi

# ══════════════════════════════════════════════════════════════════════════════
# 2. SYSTEM RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[2/3] System resources...${W}"

CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'unknown')"
CPU_CORES="$(nproc 2>/dev/null || echo '?')"
RAM_GB="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo '?')"
ok "CPU: $CPU_MODEL ($CPU_CORES cores)"
ok "RAM: ${RAM_GB} GB"

# Disk for ChromaDB
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
if [ -d "$DATA_DIR" ]; then
    DISK_FREE="$(df -h "$DATA_DIR" 2>/dev/null | tail -1 | awk '{print $4}' || echo '?')"
    ok "Disk (data/): $DISK_FREE free"
else
    info "Folder data/ will be created on first start"
fi

echo ""
info "rag_server does not use GPU directly."
info "ONNX Runtime CPU is used for reranker."
info "Embeddings computation — on separate embed_server workers."

# ══════════════════════════════════════════════════════════════════════════════
# 3. INSTALLATION
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[3/3] Confirm installation:${W}"
echo ""
echo "  Will be installed in env '$ENV_NAME':"
echo "    fastapi, uvicorn, pydantic, pydantic-settings, chromadb,"
echo "    watchdog, pathspec and dependencies"
echo ""
read -rp "  Continue? [Y/n]: " CONFIRM
CONFIRM="${CONFIRM:-Y}"
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Miniconda
if [ ! -d "$CONDA_DIR" ]; then
    info "Installing Miniconda..."
    INST="/tmp/miniconda.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$INST"
    bash "$INST" -b -p "$CONDA_DIR" && rm "$INST"
    ok "Miniconda installed"
fi
eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
grep -q "miniconda3/bin/conda" ~/.bashrc 2>/dev/null \
    || echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc

# System deps
if command -v apt-get &>/dev/null; then
    info "System packages..."
    sudo apt-get update -q
    sudo apt-get install -y -q build-essential libgomp1 curl wget
fi

# Conda env
if [ "$ENV_EXISTS" = false ]; then
    info "Creating conda env '$ENV_NAME'..."
    conda env create -f "$SCRIPT_DIR/environment.yml"
    ok "Env created"
fi
conda activate "$ENV_NAME"
pip install -q --upgrade pip

info "Installing packages..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"
ok "All packages installed"

echo ""
echo -e "${G}╔══════════════════════════════════════════╗${W}"
echo -e "${G}║         Installation complete            ║${W}"
echo -e "${G}╚══════════════════════════════════════════╝${W}"
echo ""
echo "  Env    : $CONDA_DIR/envs/$ENV_NAME"
echo "  Start  : bash $SCRIPT_DIR/start_rag.sh"
echo ""
echo "  Don't forget to configure in $SCRIPT_DIR/.env:"
echo "    EMBED_SERVER_URL=http://<embed_host>:8001"
echo "    PROJECT_ROOT=<path to project>"
echo ""
