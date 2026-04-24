#!/bin/bash
# GPU driver / runtime installer for MCP-RAG embed_server
#
# Detects GPU vendor and installs appropriate runtime:
#   NVIDIA → CUDA toolkit  (via apt or nvidia installer)
#   AMD    → ROCm          (via official AMD apt repo)
#   None   → prints notice, skips GPU install, continues
#
# Does NOT abort on GPU install failure — ONNX CPU fallback always works.
set -uo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[gpu-install] $*"; }
warn() { echo "[gpu-install] WARNING: $*" >&2; }

require_apt() {
    if ! command -v apt-get &>/dev/null; then
        warn "apt-get not found — GPU runtime install skipped (non-Debian system)"
        exit 0
    fi
}

# ── GPU detection ─────────────────────────────────────────────────────────────
VENDOR=""

if lspci 2>/dev/null | grep -qi "nvidia"; then
    VENDOR="nvidia"
elif lspci 2>/dev/null | grep -qi "amd\|radeon\|advanced micro"; then
    VENDOR="amd"
elif [ -d /proc/driver/nvidia ]; then
    VENDOR="nvidia"
elif [ -d /opt/rocm ]; then
    VENDOR="amd"
fi

log "Detected vendor: ${VENDOR:-none}"

if [ -z "$VENDOR" ]; then
    log "No supported GPU found. ONNX CPU mode will be used."
    log "If you have a GPU, ensure drivers are installed and lspci is available."
    exit 0
fi

# ── NVIDIA / CUDA ─────────────────────────────────────────────────────────────
install_cuda() {
    log "--- NVIDIA CUDA setup ---"

    # If nvidia-smi already works, drivers are installed
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        log "NVIDIA driver already installed:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
        log "CUDA runtime present — skipping driver install."
        log "If CUDA toolkit is missing, install manually:"
        log "  https://developer.nvidia.com/cuda-downloads"
        return 0
    fi

    log "nvidia-smi not found. Attempting to install NVIDIA driver via apt..."
    require_apt

    if ! sudo apt-get install -y -q ubuntu-drivers-common 2>/dev/null; then
        warn "ubuntu-drivers-common not available"
    fi

    # Try recommended driver
    if command -v ubuntu-drivers &>/dev/null; then
        sudo ubuntu-drivers autoinstall || warn "ubuntu-drivers autoinstall failed"
    else
        # Fallback: install latest nvidia driver from apt
        DRIVER_PKG="$(apt-cache search "^nvidia-driver-[0-9]" 2>/dev/null \
            | awk '{print $1}' | sort -V | tail -1)"
        if [ -n "$DRIVER_PKG" ]; then
            sudo apt-get install -y -q "$DRIVER_PKG" \
                && log "Installed $DRIVER_PKG" \
                || warn "$DRIVER_PKG install failed"
        else
            warn "No nvidia-driver package found in apt. Install manually from:"
            warn "  https://developer.nvidia.com/cuda-downloads"
        fi
    fi

    log "Reboot required after NVIDIA driver install."
}

# ── AMD / ROCm ────────────────────────────────────────────────────────────────
install_rocm() {
    log "--- AMD ROCm setup ---"
    require_apt

    # Check if ROCm already installed
    if command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
        ROCM_VER="$(cat /opt/rocm/.info/version 2>/dev/null || echo 'unknown')"
        log "ROCm already installed: $ROCM_VER"
        rocm-smi --showproductname 2>/dev/null || true
        return 0
    fi

    log "Installing ROCm..."

    # Dependencies
    sudo apt-get update -q
    sudo apt-get install -y -q \
        "linux-headers-$(uname -r)" \
        "linux-modules-extra-$(uname -r)" \
        wget gpg curl \
        || warn "Some system deps failed to install"

    # AMD apt repo
    sudo mkdir -p /etc/apt/keyrings
    if wget -q https://repo.radeon.com/rocm/rocm.gpg.key -O - \
            | gpg --dearmor \
            | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null; then
        log "ROCm GPG key added"
    else
        warn "Failed to add ROCm GPG key — network issue? ROCm install skipped."
        return 1
    fi

    # Detect Ubuntu codename
    CODENAME="$(lsb_release -cs 2>/dev/null || echo 'noble')"
    ROCM_REPO_VER="${ROCM_VERSION:-6.3.4}"

    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
https://repo.radeon.com/rocm/apt/${ROCM_REPO_VER} ${CODENAME} main" \
        | sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null

    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
        | sudo tee /etc/apt/preferences.d/rocm-pin-600 > /dev/null

    sudo apt-get update -q

    if sudo apt-get install -y rocm; then
        log "ROCm installed"
    else
        warn "ROCm install failed. CPU/ONNX mode will still work."
        return 1
    fi

    # User groups
    sudo usermod -a -G render,video "$USER" \
        && log "Added $USER to render,video groups" \
        || warn "Failed to add user to render/video groups"

    # GFX version override for unsupported APUs (e.g. Raven Ridge gfx902)
    GPU_GFX="$(rocminfo 2>/dev/null | grep -oP 'gfx\K[0-9]+' | head -1 || echo '')"
    if [ -n "$GPU_GFX" ]; then
        SUPPORTED_GFX=("900" "902" "906" "908" "90a" "1010" "1012" "1030" "1100" "1101" "1102")
        NEED_OVERRIDE=true
        for g in "${SUPPORTED_GFX[@]}"; do
            if [ "$GPU_GFX" = "$g" ]; then NEED_OVERRIDE=false; break; fi
        done
        if $NEED_OVERRIDE; then
            log "GFX version gfx$GPU_GFX may not be officially supported."
            log "Adding HSA_OVERRIDE_GFX_VERSION to ~/.bashrc"
            GFX_OVERRIDE="${GPU_GFX:0:1}.${GPU_GFX:1:1}.0"
            grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc \
                || echo "export HSA_OVERRIDE_GFX_VERSION=$GFX_OVERRIDE" >> ~/.bashrc
        fi
    fi

    log "Reboot required after ROCm install."
}

# ── Run ───────────────────────────────────────────────────────────────────────
case "$VENDOR" in
    nvidia) install_cuda  || warn "CUDA setup incomplete — CPU/ONNX fallback available" ;;
    amd)    install_rocm  || warn "ROCm setup incomplete — CPU/ONNX fallback available" ;;
esac

log ""
log "Done. GPU vendor: $VENDOR"
log "Next: run rag_server/install.sh to install Python dependencies."
log "ONNX CPU mode works without any GPU drivers."
