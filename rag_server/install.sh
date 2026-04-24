#!/bin/bash
# rag_server — интерактивный установщик
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
echo -e "${B}║        rag_server — установщик          ║${W}"
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
    for pkg in fastapi uvicorn chromadb watchdog pydantic-settings pathspec; do
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

# ══════════════════════════════════════════════════════════════════════════════
# 2. СИСТЕМНЫЕ РЕСУРСЫ
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[2/3] Системные ресурсы...${W}"

CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'unknown')"
CPU_CORES="$(nproc 2>/dev/null || echo '?')"
RAM_GB="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo '?')"
ok "CPU: $CPU_MODEL ($CPU_CORES ядер)"
ok "RAM: ${RAM_GB} GB"

# Диск для ChromaDB
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"
if [ -d "$DATA_DIR" ]; then
    DISK_FREE="$(df -h "$DATA_DIR" 2>/dev/null | tail -1 | awk '{print $4}' || echo '?')"
    ok "Диск (data/): $DISK_FREE свободно"
else
    info "Папка data/ будет создана при первом запуске"
fi

echo ""
info "rag_server не использует GPU напрямую."
info "Для reranker используется ONNX Runtime CPU."
info "Вычисления embeddings — на отдельных embed_server воркерах."

# ══════════════════════════════════════════════════════════════════════════════
# 3. УСТАНОВКА
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[3/3] Подтвердите установку:${W}"
echo ""
echo "  Будет установлено в env '$ENV_NAME':"
echo "    fastapi, uvicorn, pydantic, pydantic-settings, chromadb,"
echo "    watchdog, pathspec и зависимости"
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

# System deps
if command -v apt-get &>/dev/null; then
    info "Системные пакеты..."
    sudo apt-get update -q
    sudo apt-get install -y -q build-essential libgomp1 curl wget
fi

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

echo ""
echo -e "${G}╔══════════════════════════════════════════╗${W}"
echo -e "${G}║           Установка завершена            ║${W}"
echo -e "${G}╚══════════════════════════════════════════╝${W}"
echo ""
echo "  Env    : $CONDA_DIR/envs/$ENV_NAME"
echo "  Запуск : bash $SCRIPT_DIR/start_rag.sh"
echo ""
echo "  Не забудьте указать в $SCRIPT_DIR/.env:"
echo "    EMBED_SERVER_URL=http://<embed_host>:8001"
echo "    PROJECT_ROOT=<путь к проекту>"
echo ""
