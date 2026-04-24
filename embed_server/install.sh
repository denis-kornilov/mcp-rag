#!/bin/bash
# embed_server — interactive installer
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="mcp-embed"

# ── colors ────────────────────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; C='\033[0;36m'; W='\033[0m'
ok()   { echo -e "${G}  ✓ $*${W}"; }
warn() { echo -e "${Y}  ! $*${W}"; }
info() { echo -e "${C}  » $*${W}"; }
err()  { echo -e "${R}  ✗ $*${W}"; }

echo ""
echo -e "${B}╔══════════════════════════════════════════╗${W}"
echo -e "${B}║         embed_server — installer         ║${W}"
echo -e "${B}╚══════════════════════════════════════════╝${W}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHECK INSTALLED SOFTWARE
# ══════════════════════════════════════════════════════════════════════════════
echo -e "${B}[1/4] Checking installed software...${W}"

# Conda
if [ -d "$CONDA_DIR" ]; then
    CONDA_VER="$("$CONDA_DIR/bin/conda" --version 2>/dev/null || echo '?')"
    ok "Conda: $CONDA_VER ($CONDA_DIR)"
else
    warn "Conda not found — will be installed"
fi

# Conda env
eval "$("$CONDA_DIR/bin/conda" shell.bash hook 2>/dev/null)" || true
if conda env list 2>/dev/null | grep -q "^$ENV_NAME "; then
    PYTHON_VER="$($CONDA_DIR/envs/$ENV_NAME/bin/python --version 2>/dev/null || echo '?')"
    ok "Conda env '$ENV_NAME': $PYTHON_VER"
    ENV_EXISTS=true
else
    warn "Conda env '$ENV_NAME' not found"
    ENV_EXISTS=false
fi

# Installed packages (if env exists)
if [ "$ENV_EXISTS" = true ]; then
    PY="$CONDA_DIR/envs/$ENV_NAME/bin/python"
    for pkg in onnxruntime onnxruntime-gpu onnxruntime-rocm optimum transformers fastapi uvicorn; do
        if "$PY" -c "import importlib; importlib.import_module('$pkg'.replace('-','_').split('[')[0])" 2>/dev/null; then
            PKG_VER="$("$PY" -c "import importlib.metadata; print(importlib.metadata.version('$pkg'))" 2>/dev/null || echo '?')"
            ok "$pkg: $PKG_VER"
        else
            warn "$pkg: not installed"
        fi
    done
fi

# ══════════════════════════════════════════════════════════════════════════════
# 2. DETECT HARDWARE AND DRIVERS
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[2/4] Detecting hardware...${W}"

# CPU
CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'unknown')"
CPU_CORES="$(nproc 2>/dev/null || echo '?')"
RAM_GB="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo '?')"
ok "CPU: $CPU_MODEL ($CPU_CORES cores, RAM: ${RAM_GB}GB)"

# NVIDIA — check lspci (bare metal) and nvidia-smi (WSL2/VM/container)
NVIDIA_HW=false
NVIDIA_DRIVER=false
if lspci 2>/dev/null | grep -qi nvidia; then
    NVIDIA_GPU="$(lspci 2>/dev/null | grep -i nvidia | head -1 | sed 's/.*: //')"
    NVIDIA_HW=true
    info "NVIDIA GPU (lspci): $NVIDIA_GPU"
fi
# WSL2 / containers — lspci doesn't see GPU, but nvidia-smi works
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
        ok "  Driver: $NVDR_VER | CUDA: $CUDA_VER | VRAM: ${VRAM}MB"
    else
        warn "  NVIDIA Driver is not installed or not working"
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
            warn "  ROCm is not installed"
        fi
    fi
fi

if [ "$NVIDIA_HW" = false ] && [ "$AMD_HW" = false ]; then
    info "GPU not detected — only CPU mode is available"
fi

# ══════════════════════════════════════════════════════════════════════════════
# 3. SELECT COMPUTE BACKEND
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[3/4] Select compute backend:${W}"
echo ""

# Build available options
OPTIONS=()
LABELS=()
BACKENDS=()
ONNX_PKGS=()

# CPU is always available
OPTIONS+=("1")
LABELS+=("CPU — ONNX Runtime INT8  (always works, no GPU required)")
BACKENDS+=("onnx-cpu")
ONNX_PKGS+=("onnxruntime")

IDX=2

if [ "$NVIDIA_HW" = true ]; then
    if [ "$NVIDIA_DRIVER" = true ]; then
        OPTIONS+=("$IDX")
        LABELS+=("NVIDIA GPU — ONNX CUDA  (driver installed, CUDA $CUDA_VER)")
        BACKENDS+=("onnx-cuda")
        ONNX_PKGS+=("onnxruntime-gpu")
    else
        OPTIONS+=("$IDX")
        LABELS+=("NVIDIA GPU — ONNX CUDA  ${Y}[DRIVER NOT INSTALLED — installation will be offered]${W}")
        BACKENDS+=("onnx-cuda")
        ONNX_PKGS+=("onnxruntime-gpu")
    fi
    IDX=$((IDX+1))
fi

if [ "$AMD_HW" = true ]; then
    if [ "$AMD_DRIVER" = true ]; then
        OPTIONS+=("$IDX")
        LABELS+=("AMD GPU — ONNX ROCm  (ROCm $ROCM_VER installed)")
        BACKENDS+=("onnx-rocm")
        ONNX_PKGS+=("onnxruntime-rocm")
    else
        OPTIONS+=("$IDX")
        LABELS+=("AMD GPU — ONNX ROCm  ${Y}[ROCm NOT INSTALLED — installation will be offered]${W}")
        BACKENDS+=("onnx-rocm")
        ONNX_PKGS+=("onnxruntime-rocm")
    fi
    IDX=$((IDX+1))
fi

# Show menu
for i in "${!OPTIONS[@]}"; do
    echo -e "  ${G}${OPTIONS[$i]}${W}) ${LABELS[$i]}"
done
echo ""

while true; do
    read -rp "  Your choice [${OPTIONS[0]}]: " CHOICE
    CHOICE="${CHOICE:-${OPTIONS[0]}}"
    for i in "${!OPTIONS[@]}"; do
        if [ "$CHOICE" = "${OPTIONS[$i]}" ]; then
            SELECTED_IDX=$i
            break 2
        fi
    done
    err "Invalid choice. Enter one of: ${OPTIONS[*]}"
done

EMBED_BACKEND="${BACKENDS[$SELECTED_IDX]}"
ONNXRUNTIME_PKG="${ONNX_PKGS[$SELECTED_IDX]}"
echo ""
ok "Selected: $EMBED_BACKEND ($ONNXRUNTIME_PKG)"

# Offer driver installation if needed
if [ "$EMBED_BACKEND" = "onnx-cuda" ] && [ "$NVIDIA_DRIVER" = false ]; then
    echo ""
    warn "NVIDIA Driver is not installed."
    read -rp "  Install via ubuntu-drivers? [y/N]: " INST_DRV
    if [[ "$INST_DRV" =~ ^[Yy]$ ]]; then
        sudo apt-get install -y -q ubuntu-drivers-common && sudo ubuntu-drivers autoinstall \
            && ok "Driver installed. Reboot is required." \
            || warn "Failed to install driver — continuing with CPU"
    else
        warn "Skipping. onnxruntime-gpu might not work without driver."
    fi
fi

if [ "$EMBED_BACKEND" = "onnx-rocm" ] && [ "$AMD_DRIVER" = false ]; then
    echo ""
    warn "ROCm is not installed."
    read -rp "  Install ROCm? [y/N]: " INST_ROCM
    if [[ "$INST_ROCM" =~ ^[Yy]$ ]]; then
        bash "$(dirname "$SCRIPT_DIR")/install_adds.sh" \
            && ok "ROCm installed. Reboot is required." \
            || warn "Failed to install ROCm — continuing with CPU"
    else
        warn "Skipping. onnxruntime-rocm might not work without ROCm."
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# 4. INSTALLATION
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${B}[4/4] Installing...${W}"

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

# Conda env
if [ "$ENV_EXISTS" = false ]; then
    info "Creating conda env '$ENV_NAME'..."
    conda env create -f "$SCRIPT_DIR/environment.yml"
    ok "Env created"
fi
conda activate "$ENV_NAME"
pip install -q --upgrade pip

# Base deps
info "Installing base packages..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"
ok "Base packages installed"

# onnxruntime (conflict with each other — uninstall all variants first)
info "Installing $ONNXRUNTIME_PKG..."
pip uninstall -q -y onnxruntime onnxruntime-gpu onnxruntime-rocm 2>/dev/null || true

INSTALL_OK=false

if [ "$EMBED_BACKEND" = "onnx-cuda" ]; then
    # Determine CUDA version for correct package
    CUDA_MAJOR="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' | head -1 || echo '')"
    info "CUDA major version: ${CUDA_MAJOR:-unknown}"

    # Try conda-forge (pulls cudatoolkit automatically — most reliable way)
    info "Trying conda install onnxruntime-gpu (recommended — pulls CUDA automatically)..."
    if conda install -y -q -c conda-forge onnxruntime-gpu 2>/dev/null; then
        INSTALL_OK=true
        ok "onnxruntime-gpu installed via conda (CUDA bundled)"
    else
        # Fallback: pip with correct index for CUDA version
        if [ "$CUDA_MAJOR" = "12" ]; then
            PIP_INDEX="https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"
            info "Trying pip install onnxruntime-gpu (CUDA 12)..."
            if pip install -q onnxruntime-gpu --extra-index-url "$PIP_INDEX"; then
                INSTALL_OK=true
                ok "onnxruntime-gpu (CUDA 12) installed via pip"
            fi
        fi
        if [ "$INSTALL_OK" = false ]; then
            info "Trying pip install onnxruntime-gpu (standard, CUDA 11/12)..."
            if pip install -q onnxruntime-gpu; then
                INSTALL_OK=true
                ok "onnxruntime-gpu installed via pip"
            fi
        fi
    fi

    # Verify CUDAExecutionProvider is actually available
    if [ "$INSTALL_OK" = true ]; then
        ORT_PROVIDERS="$(python -c "import onnxruntime as ort; print(ort.get_available_providers())" 2>/dev/null || echo '')"
        info "ORT providers: $ORT_PROVIDERS"
        if echo "$ORT_PROVIDERS" | grep -q "CUDAExecutionProvider"; then
            ok "CUDAExecutionProvider is available — GPU will be used"
        else
            warn "CUDAExecutionProvider NOT found despite installing onnxruntime-gpu."
            warn "Possible reasons:"
            warn "  1. System-wide CUDA Toolkit is not installed (apt install cuda-toolkit-12-x)"
            warn "  2. System CUDA version does not match onnxruntime-gpu"
            warn "  3. WSL2 — WSL2 CUDA driver is required (not standard linux driver)"
            warn "Falling back to CPU. After installing CUDA, run install.sh again."
            pip uninstall -q -y onnxruntime-gpu 2>/dev/null || true
            pip install -q onnxruntime
            EMBED_BACKEND="onnx-cpu"
        fi
    fi

elif [ "$EMBED_BACKEND" = "onnx-rocm" ]; then
    if pip install -q "$ONNXRUNTIME_PKG"; then
        INSTALL_OK=true
        ok "$ONNXRUNTIME_PKG installed"
        ORT_PROVIDERS="$(python -c "import onnxruntime as ort; print(ort.get_available_providers())" 2>/dev/null || echo '')"
        info "ORT providers: $ORT_PROVIDERS"
        if echo "$ORT_PROVIDERS" | grep -q "ROCMExecutionProvider"; then
            ok "ROCMExecutionProvider is available — GPU will be used"
        else
            warn "ROCMExecutionProvider NOT found. Check ROCm installation."
            pip uninstall -q -y onnxruntime-rocm 2>/dev/null || true
            pip install -q onnxruntime
            EMBED_BACKEND="onnx-cpu"
        fi
    fi

else
    # CPU
    pip install -q onnxruntime && ok "onnxruntime (CPU) installed"
fi

if [ "$INSTALL_OK" = false ] && [ "$EMBED_BACKEND" != "onnx-cpu" ]; then
    warn "GPU package installation failed, falling back to CPU"
    pip install -q onnxruntime
    EMBED_BACKEND="onnx-cpu"
fi

# Update .env
sed -i "s|^EMBED_BACKEND=.*|EMBED_BACKEND=$EMBED_BACKEND|" "$SCRIPT_DIR/.env"
ok "EMBED_BACKEND=$EMBED_BACKEND written to .env"

echo ""
echo -e "${G}╔══════════════════════════════════════════╗${W}"
echo -e "${G}║         Installation completed           ║${W}"
echo -e "${G}╚══════════════════════════════════════════╝${W}"
echo ""
echo "  Mode    : $EMBED_BACKEND"
echo "  Env     : $CONDA_DIR/envs/$ENV_NAME"
echo "  Run     : bash $SCRIPT_DIR/start_embed.sh"
echo ""
