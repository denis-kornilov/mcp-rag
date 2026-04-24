#!/bin/bash
# embed_server — интерактивный установщик
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="mcp-embed"

# ── цвета ─────────────────────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; C='\033[0;36m'; W='\033[0m'
ok()   { echo -e "${G}  ✓ $*${W}"; }
warn() { echo -e "${Y}  ! $*${W}"; }
info() { echo -e "${C}  » $*${W}"; }
err()  { echo -e "${R}  ✗ $*${W}"; }

echo ""
echo -e "${B}╔══════════════════════════════════════════╗${W}"
echo -e "${B}║       embed_server — установщик         ║${W}"
echo -e "${B}╚══════════════════════════════════════════╝${W}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 1. ПРОВЕРКА ТОГО ЧТО УЖЕ УСТАНОВЛЕНО
# ══════════════════════════════════════════════════════════════════════════════
echo -e "${B}[1/4] Проверяем установленное ПО...${W}"

# Conda
if [ -d "$CONDA_DIR" ]; then
    CONDA_VER="$("$CONDA_DIR/bin/conda" --version 2>/dev/null || echo '?')"
    ok "Conda: $CONDA_VER ($CONDA_DIR)"
else
    warn "Conda не найдена — будет установлена"
fi

# Conda env
eval "$("$CONDA_DIR/bin/conda" shell.bash hook 2>/dev/null)" || true
if conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
    PYTHON_VER="$($CONDA_DIR/envs/$ENV_NAME/bin/python --version 2>/dev/null || echo '?')"
    ok "Conda env '$ENV_NAME': $PYTHON_VER"
    ENV_EXISTS=true
else
    warn "Conda env '$ENV_NAME' не найдена"
    ENV_EXISTS=false
fi

# Установленные пакеты (если env есть)
if [ "$ENV_EXISTS" = true ]; then
    PY="$CONDA_DIR/envs/$ENV_NAME/bin/python"
    for pkg in onnxruntime onnxruntime-gpu onnxruntime-rocm optimum transformers fastapi uvicorn; do
        if "$PY" -c "import importlib; importlib.import_module('$pkg'.replace('-','_').split('[')[0])" 2>/dev/null; then
            PKG_VER="$("$PY" -c "import importlib.metadata; print(importlib.metadata.version('$pkg'))" 2>/dev/null || echo '?')"
            ok "$pkg: $PKG_VER"
        else
            warn "$pkg: не установлен"
        fi
    done
fi

# ══════════════════════════════════════════════════════════════════════════════
# 2. ОПРЕДЕЛЕНИЕ ЖЕЛЕЗА И ДРАЙВЕРОВ
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[2/4] Определяем железо...${W}"

# CPU
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'unknown')"
CPU_CORES="$(nproc 2>/dev/null || echo '?')"
RAM_GB="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo '?')"
ok "CPU: $CPU_MODEL ($CPU_CORES ядер, RAM: ${RAM_GB}GB)"

# NVIDIA — проверяем lspci (bare metal) и nvidia-smi (WSL2/VM/container)
NVIDIA_HW=false
NVIDIA_DRIVER=false
if lspci 2>/dev/null | grep -qi nvidia; then
    NVIDIA_GPU="$(lspci 2>/dev/null | grep -i nvidia | head -1 | sed 's/.*: //')"
    NVIDIA_HW=true
    info "NVIDIA GPU (lspci): $NVIDIA_GPU"
fi
# WSL2 / контейнеры — lspci не видит GPU, но nvidia-smi работает
if [ "$NVIDIA_HW" = false ] && command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
    NVIDIA_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'NVIDIA GPU')"
    NVIDIA_HW=true
    info "NVIDIA GPU (nvidia-smi, WSL2/VM): $NVIDIA_GPU"
fi
if [ "$NVIDIA_HW" = true ]; then
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        NVIDIA_DRIVER=true
        CUDA_VER="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' || echo '?')"
        NVDR_VER="$(nvidia-smi | grep -oP 'Driver Version: \K[0-9.]+' || echo '?')"
        VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '?')"
        ok "  Драйвер: $NVDR_VER | CUDA: $CUDA_VER | VRAM: ${VRAM}MB"
    else
        warn "  Драйвер NVIDIA не установлен или не работает"
    fi
fi

# AMD
AMD_HW=false
AMD_DRIVER=false
if lspci 2>/dev/null | grep -qi 'amd\|radeon'; then
    AMD_GPU="$(lspci 2>/dev/null | grep -i 'amd\|radeon' | grep -i 'vga\|display\|3d' | head -1 | sed 's/.*: //')"
    if [ -n "$AMD_GPU" ]; then
        AMD_HW=true
        info "AMD GPU: $AMD_GPU"
        if { command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; } || [ -d /opt/rocm ]; then
            AMD_DRIVER=true
            ROCM_VER="$(cat /opt/rocm/.info/version 2>/dev/null || rocm-smi --version 2>/dev/null | head -1 || echo '?')"
            ok "  ROCm: $ROCM_VER"
        else
            warn "  ROCm не установлен"
        fi
    fi
fi

if [ "$NVIDIA_HW" = false ] && [ "$AMD_HW" = false ]; then
    info "GPU не обнаружен — доступен только CPU режим"
fi

# ══════════════════════════════════════════════════════════════════════════════
# 3. ВЫБОР РЕЖИМА ВЫЧИСЛЕНИЙ
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[3/4] Выберите режим вычислений:${W}"
echo ""

# Собираем доступные варианты
OPTIONS=()
LABELS=()
BACKENDS=()
ONNX_PKGS=()

# CPU всегда доступен
OPTIONS+=("1")
LABELS+=("CPU — ONNX Runtime INT8  (всегда работает, не требует GPU)")
BACKENDS+=("onnx-cpu")
ONNX_PKGS+=("onnxruntime")

IDX=2

if [ "$NVIDIA_HW" = true ]; then
    if [ "$NVIDIA_DRIVER" = true ]; then
        OPTIONS+=("$IDX")
        LABELS+=("NVIDIA GPU — ONNX CUDA  (драйвер установлен, CUDA $CUDA_VER)")
        BACKENDS+=("onnx-cuda")
        ONNX_PKGS+=("onnxruntime-gpu")
    else
        OPTIONS+=("$IDX")
        LABELS+=("NVIDIA GPU — ONNX CUDA  ${Y}[ДРАЙВЕР НЕ УСТАНОВЛЕН — будет предложена установка]${W}")
        BACKENDS+=("onnx-cuda")
        ONNX_PKGS+=("onnxruntime-gpu")
    fi
    IDX=$((IDX+1))
fi

if [ "$AMD_HW" = true ]; then
    if [ "$AMD_DRIVER" = true ]; then
        OPTIONS+=("$IDX")
        LABELS+=("AMD GPU — ONNX ROCm  (ROCm $ROCM_VER установлен)")
        BACKENDS+=("onnx-rocm")
        ONNX_PKGS+=("onnxruntime-rocm")
    else
        OPTIONS+=("$IDX")
        LABELS+=("AMD GPU — ONNX ROCm  ${Y}[ROCm НЕ УСТАНОВЛЕН — будет предложена установка]${W}")
        BACKENDS+=("onnx-rocm")
        ONNX_PKGS+=("onnxruntime-rocm")
    fi
    IDX=$((IDX+1))
fi

# Показываем меню
for i in "${!OPTIONS[@]}"; do
    echo -e "  ${G}${OPTIONS[$i]}${W}) ${LABELS[$i]}"
done
echo ""

while true; do
    read -rp "  Ваш выбор [${OPTIONS[0]}]: " CHOICE
    CHOICE="${CHOICE:-${OPTIONS[0]}}"
    for i in "${!OPTIONS[@]}"; do
        if [ "$CHOICE" = "${OPTIONS[$i]}" ]; then
            SELECTED_IDX=$i
            break 2
        fi
    done
    err "Неверный выбор. Введите одно из: ${OPTIONS[*]}"
done

EMBED_BACKEND="${BACKENDS[$SELECTED_IDX]}"
ONNXRUNTIME_PKG="${ONNX_PKGS[$SELECTED_IDX]}"
echo ""
ok "Выбрано: $EMBED_BACKEND ($ONNXRUNTIME_PKG)"

# Предложить установку драйвера если нужно
if [ "$EMBED_BACKEND" = "onnx-cuda" ] && [ "$NVIDIA_DRIVER" = false ]; then
    echo ""
    warn "Драйвер NVIDIA не установлен."
    read -rp "  Установить через ubuntu-drivers? [y/N]: " INST_DRV
    if [[ "$INST_DRV" =~ ^[Yy]$ ]]; then
        sudo apt-get install -y -q ubuntu-drivers-common && sudo ubuntu-drivers autoinstall \
            && ok "Драйвер установлен. Потребуется перезагрузка." \
            || warn "Не удалось установить драйвер — продолжаем с CPU"
    else
        warn "Пропускаем. onnxruntime-gpu может не работать без драйвера."
    fi
fi

if [ "$EMBED_BACKEND" = "onnx-rocm" ] && [ "$AMD_DRIVER" = false ]; then
    echo ""
    warn "ROCm не установлен."
    read -rp "  Установить ROCm? [y/N]: " INST_ROCM
    if [[ "$INST_ROCM" =~ ^[Yy]$ ]]; then
        bash "$(dirname "$SCRIPT_DIR")/install_adds.sh" \
            && ok "ROCm установлен. Потребуется перезагрузка." \
            || warn "Не удалось установить ROCm — продолжаем с CPU"
    else
        warn "Пропускаем. onnxruntime-rocm может не работать без ROCm."
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# 4. УСТАНОВКА
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[4/4] Устанавливаем...${W}"

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

# Base deps
info "Устанавливаем базовые пакеты..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"
ok "Базовые пакеты установлены"

# onnxruntime (конфликтуют — сначала удаляем все варианты)
info "Устанавливаем $ONNXRUNTIME_PKG..."
pip uninstall -q -y onnxruntime onnxruntime-gpu onnxruntime-rocm 2>/dev/null || true

INSTALL_OK=false

if [ "$EMBED_BACKEND" = "onnx-cuda" ]; then
    # Определяем версию CUDA для правильного пакета
    CUDA_MAJOR="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' | head -1 || echo '')"
    info "CUDA major version: ${CUDA_MAJOR:-неизвестно}"

    # Пробуем conda-forge (тянет cudatoolkit сам — самый надёжный способ)
    info "Пробуем conda install onnxruntime-gpu (рекомендуется — тянет CUDA сам)..."
    if conda install -y -q -c conda-forge onnxruntime-gpu 2>/dev/null; then
        INSTALL_OK=true
        ok "onnxruntime-gpu установлен через conda (CUDA bundled)"
    else
        # Fallback: pip с нужным индексом под версию CUDA
        if [ "$CUDA_MAJOR" = "12" ]; then
            PIP_INDEX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
            info "Пробуем pip install onnxruntime-gpu (CUDA 12)..."
            if pip install -q onnxruntime-gpu --extra-index-url "$PIP_INDEX"; then
                INSTALL_OK=true
                ok "onnxruntime-gpu (CUDA 12) установлен через pip"
            fi
        fi
        if [ "$INSTALL_OK" = false ]; then
            info "Пробуем pip install onnxruntime-gpu (стандартный, CUDA 11/12)..."
            if pip install -q onnxruntime-gpu; then
                INSTALL_OK=true
                ok "onnxruntime-gpu установлен через pip"
            fi
        fi
    fi

    # Проверяем что CUDAExecutionProvider реально доступен
    if [ "$INSTALL_OK" = true ]; then
        ORT_PROVIDERS="$(python -c "import onnxruntime as ort; print(ort.get_available_providers())" 2>/dev/null || echo '')"
        info "ORT providers: $ORT_PROVIDERS"
        if echo "$ORT_PROVIDERS" | grep -q "CUDAExecutionProvider"; then
            ok "CUDAExecutionProvider доступен — GPU будет использоваться"
        else
            warn "CUDAExecutionProvider НЕ найден несмотря на установку onnxruntime-gpu."
            warn "Вероятные причины:"
            warn "  1. CUDA Toolkit не установлен системно (apt install cuda-toolkit-12-x)"
            warn "  2. Версия CUDA в системе не совпадает с onnxruntime-gpu"
            warn "  3. WSL2 — нужен WSL2 CUDA driver (не обычный linux driver)"
            warn "Откатываемся на CPU. После установки CUDA запусти install.sh повторно."
            pip uninstall -q -y onnxruntime-gpu 2>/dev/null || true
            pip install -q onnxruntime
            EMBED_BACKEND="onnx-cpu"
        fi
    fi

elif [ "$EMBED_BACKEND" = "onnx-rocm" ]; then
    if pip install -q "$ONNXRUNTIME_PKG"; then
        INSTALL_OK=true
        ok "$ONNXRUNTIME_PKG установлен"
        ORT_PROVIDERS="$(python -c "import onnxruntime as ort; print(ort.get_available_providers())" 2>/dev/null || echo '')"
        info "ORT providers: $ORT_PROVIDERS"
        if echo "$ORT_PROVIDERS" | grep -q "ROCMExecutionProvider"; then
            ok "ROCMExecutionProvider доступен — GPU будет использоваться"
        else
            warn "ROCMExecutionProvider НЕ найден. Проверьте установку ROCm."
            pip uninstall -q -y onnxruntime-rocm 2>/dev/null || true
            pip install -q onnxruntime
            EMBED_BACKEND="onnx-cpu"
        fi
    fi

else
    # CPU
    pip install -q onnxruntime && ok "onnxruntime (CPU) установлен"
fi

if [ "$INSTALL_OK" = false ] && [ "$EMBED_BACKEND" != "onnx-cpu" ]; then
    warn "Установка GPU пакета не удалась, откат на CPU"
    pip install -q onnxruntime
    EMBED_BACKEND="onnx-cpu"
fi

# Обновить .env
sed -i "s|^EMBED_BACKEND=.*|EMBED_BACKEND=$EMBED_BACKEND|" "$SCRIPT_DIR/.env"
ok "EMBED_BACKEND=$EMBED_BACKEND записан в .env"

echo ""
echo -e "${G}╔══════════════════════════════════════════╗${W}"
echo -e "${G}║           Установка завершена            ║${W}"
echo -e "${G}╚══════════════════════════════════════════╝${W}"
echo ""
echo "  Режим   : $EMBED_BACKEND"
echo "  Env     : $CONDA_DIR/envs/$ENV_NAME"
echo "  Запуск  : bash $SCRIPT_DIR/start_embed.sh"
echo ""
