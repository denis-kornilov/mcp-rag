#!/bin/bash
# embed_server — start (remote HTTP mode)
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
echo -e "${B}║         embed_server — start            ║${W}"
echo -e "${B}╚══════════════════════════════════════════╝${W}"
echo ""

# ── 1. Check .env ──────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    err ".env not found: $ENV_FILE — run install.sh"
fi

_env() { { grep "^$1=" "$ENV_FILE" 2>/dev/null || true; } | cut -d= -f2- | xargs; }

EMBED_HOST="$(_env EMBED_HOST)"
EMBED_PORT="$(_env EMBED_PORT)"
ENV_NAME="$(_env ENV_NAME)"
EMBED_BACKEND="$(_env EMBED_BACKEND)"
ONNX_THREADS="$(_env ONNX_NUM_THREADS)"
SEARCH_THREADS="$(_env SEARCH_THREADS)"
INTER_THREADS="$(_env ONNX_INTER_THREADS)"
EXEC_MODE="$(_env ONNX_EXECUTION_MODE)"
CUDA_DEVICE_ID="$(_env CUDA_DEVICE_ID)"
ROCM_DEVICE_ID="$(_env ROCM_DEVICE_ID)"

echo -e "${B}[1/3] Checking .env configuration...${W}"

# EMBED_HOST
EMBED_HOST="${EMBED_HOST:-0.0.0.0}"
ok "EMBED_HOST=$EMBED_HOST"

# EMBED_PORT
EMBED_PORT="${EMBED_PORT:-8001}"
ok "EMBED_PORT=$EMBED_PORT"

# ENV_NAME
ENV_NAME="${ENV_NAME:-mcp-embed}"
ok "ENV_NAME=$ENV_NAME"

# EMBED_BACKEND
EMBED_BACKEND="${EMBED_BACKEND:-onnx-cpu}"
VALID_BACKENDS="onnx-cpu onnx-cuda onnx-rocm onnx-auto"
BACKEND_OK=false
for b in $VALID_BACKENDS; do
    [ "$EMBED_BACKEND" = "$b" ] && BACKEND_OK=true && break
done
if [ "$BACKEND_OK" = false ]; then
    warn "Unknown EMBED_BACKEND=$EMBED_BACKEND — valid options: $VALID_BACKENDS"
    warn "Will use onnx-cpu"
    sed -i "s|^EMBED_BACKEND=.*|EMBED_BACKEND=onnx-cpu|" "$ENV_FILE"
    EMBED_BACKEND="onnx-cpu"
fi
ok "EMBED_BACKEND=$EMBED_BACKEND"

# Threads
ONNX_THREADS="${ONNX_THREADS:-4}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
INTER_THREADS="${INTER_THREADS:-1}"
EXEC_MODE="${EXEC_MODE:-sequential}"
ok "Threads: ingest=$ONNX_THREADS search=$SEARCH_THREADS inter=$INTER_THREADS mode=$EXEC_MODE"

# GPU device IDs
CUDA_DEVICE_ID="${CUDA_DEVICE_ID:-0}"
ROCM_DEVICE_ID="${ROCM_DEVICE_ID:-0}"

HF_HOME_ABS="$SCRIPT_DIR/embed_data/hf_home"
ok "HF_HOME: $HF_HOME_ABS"

# ── 2. Check ONNX model and onnxruntime ─────────────────────────────────────
echo ""
echo -e "${B}[2/3] Checking model and onnxruntime...${W}"

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
PYTHON="${PYTHON:-$("$CONDA_DIR/bin/conda" run -n "$ENV_NAME" which python 2>/dev/null || which python3)}"

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    err "Python not found in env '$ENV_NAME'. Run: bash $SCRIPT_DIR/install.sh"
fi
ok "Python: $PYTHON"

# Check onnxruntime
ORT_OK="$("$PYTHON" -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo '')"
if [ -z "$ORT_OK" ]; then
    err "onnxruntime not installed in env '$ENV_NAME'. Run: bash $SCRIPT_DIR/install.sh"
fi
ok "onnxruntime: $ORT_OK"

# Check ORT provider for selected backend
if [ "$EMBED_BACKEND" = "onnx-cuda" ]; then
    CUDA_OK="$("$PYTHON" -c "import onnxruntime as o; print('ok' if 'CUDAExecutionProvider' in o.get_available_providers() else '')" 2>/dev/null || echo '')"
    if [ -z "$CUDA_OK" ]; then
        warn "CUDAExecutionProvider unavailable — EMBED_BACKEND=$EMBED_BACKEND will not work."
        warn "Switch to CPU? Change EMBED_BACKEND=onnx-cpu in $ENV_FILE"
    else
        ok "CUDAExecutionProvider available — device_id=$CUDA_DEVICE_ID"
        # Show GPU info if nvidia-smi is available
        if command -v nvidia-smi &>/dev/null; then
            GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader --id="$CUDA_DEVICE_ID" 2>/dev/null | head -1)"
            GPU_MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader --id="$CUDA_DEVICE_ID" 2>/dev/null | head -1)"
            [ -n "$GPU_NAME" ] && info "GPU[$CUDA_DEVICE_ID]: $GPU_NAME  VRAM: $GPU_MEM"
        fi
    fi
elif [ "$EMBED_BACKEND" = "onnx-rocm" ]; then
    ROCM_OK="$("$PYTHON" -c "import onnxruntime as o; print('ok' if 'ROCMExecutionProvider' in o.get_available_providers() else '')" 2>/dev/null || echo '')"
    if [ -z "$ROCM_OK" ]; then
        warn "ROCMExecutionProvider unavailable — EMBED_BACKEND=$EMBED_BACKEND will not work."
        warn "Switch to CPU? Change EMBED_BACKEND=onnx-cpu in $ENV_FILE"
    else
        ok "ROCMExecutionProvider available — device_id=$ROCM_DEVICE_ID"
        # Show GPU info if rocm-smi is available
        if command -v rocm-smi &>/dev/null; then
            ROCM_INFO="$(rocm-smi --showproductname 2>/dev/null | grep -i "gpu\[${ROCM_DEVICE_ID}\]" | head -1)"
            [ -n "$ROCM_INFO" ] && info "$ROCM_INFO"
        fi
    fi
elif [ "$EMBED_BACKEND" = "onnx-auto" ]; then
    AUTO_PROV="$("$PYTHON" -c "
import onnxruntime as o
av = o.get_available_providers()
if 'CUDAExecutionProvider' in av: print('cuda')
elif 'ROCMExecutionProvider' in av: print('rocm')
else: print('cpu')
" 2>/dev/null || echo 'cpu')"
    case "$AUTO_PROV" in
        cuda) ok "onnx-auto → CUDAExecutionProvider (device_id=$CUDA_DEVICE_ID)" ;;
        rocm) ok "onnx-auto → ROCMExecutionProvider (device_id=$ROCM_DEVICE_ID)" ;;
        *)    ok "onnx-auto → CPUExecutionProvider (GPU not found)" ;;
    esac
fi

# Check ONNX model file
ONNX_DIR="$SCRIPT_DIR/embed_data/onnx_exports/bge-m3"
if [ -f "$ONNX_DIR/model_quantized.onnx" ]; then
    ok "Model (INT8): $ONNX_DIR/model_quantized.onnx"
elif [ -f "$ONNX_DIR/model.onnx" ]; then
    ok "Model (FP32): $ONNX_DIR/model.onnx"
else
    warn "ONNX model not found in $ONNX_DIR"
    warn "On first request, model will be exported automatically (~3–8 min)."
fi

# ── 3. Start ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${B}[3/3] Starting embed_server...${W}"

export HF_HOME="$HF_HOME_ABS"
export PYTHONPATH="$ROOT_DIR"

echo ""
info "workdir  : $SCRIPT_DIR"
info "listen   : $EMBED_HOST:$EMBED_PORT"
info "backend  : $EMBED_BACKEND  (ingest=$ONNX_THREADS + search=$SEARCH_THREADS intra / inter=$INTER_THREADS mode=$EXEC_MODE)"
info "HF_HOME  : $HF_HOME_ABS"
info "python   : $PYTHON"
echo ""

cd "$SCRIPT_DIR"
exec "$PYTHON" server.py --host "$EMBED_HOST" --port "$EMBED_PORT"
