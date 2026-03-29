#!/bin/bash
# Install script for OpenSora + OpenVLA-OFT on H100 (sm_90, CUDA 12.4)
#
# Usage:
#   bash install_opensora_h100.sh [--use-mirror] [--no-root]
#
# Prerequisites:
#   - NVIDIA H100 / A100 GPU (sm_80 / sm_90)
#   - CUDA 12.4+ driver installed
#   - Git available

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR=".venv"
PYTHON_VERSION="3.11.14"
USE_MIRRORS=0
NO_ROOT=0
GITHUB_PREFIX=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --use-mirror) USE_MIRRORS=1; shift ;;
        --no-root)    NO_ROOT=1; shift ;;
        *)            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ "$USE_MIRRORS" -eq 1 ]; then
    export UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download
    export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
    export HF_ENDPOINT=https://hf-mirror.com
    GITHUB_PREFIX="https://ghfast.top/"
    git config --global url."${GITHUB_PREFIX}github.com/".insteadOf "https://github.com/"
fi

TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
TORCHAUDIO_VERSION="2.6.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

echo "============================================"
echo "  OpenSora + OpenVLA-OFT Installer (H100)"
echo "  PyTorch ${TORCH_VERSION} + CUDA 12.4"
echo "============================================"

# --- uv ---
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv 2>/dev/null || (wget -qO- https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH")
fi

# --- venv ---
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Reusing existing venv at $VENV_DIR"
    source "$VENV_DIR/bin/activate"
else
    uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
    source "$VENV_DIR/bin/activate"
fi
# uv sync installs project deps (including torch) from pyproject.toml.
# We MUST override torch afterwards with the correct version.
UV_TORCH_BACKEND=auto uv sync --extra embodied --active --no-install-project

# --- System deps ---
if [ "$NO_ROOT" -eq 0 ]; then
    bash "$SCRIPT_DIR/requirements/embodied/sys_deps.sh"
fi
{
    echo "export NVIDIA_DRIVER_CAPABILITIES=all"
    echo "export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json"
    echo "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json"
} >> "$VENV_DIR/bin/activate"

# --- Common embodied deps ---
uv pip install -r "$SCRIPT_DIR/requirements/embodied/envs/common.txt"

# Pin numpy < 2.0 so tensorflow 2.15 (needed by prismatic/dlimp at import time) works.
# torch 2.6 is compatible with numpy >= 1.22, so 1.26.4 satisfies both.
pip install numpy==1.26.4

# --- Step 1: PyTorch (pin the correct version, override uv sync) ---
echo "[1/8] Installing PyTorch ${TORCH_VERSION} (cu124)..."
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
    --index-url ${TORCH_INDEX_URL} --force-reinstall

# Verify
INSTALLED_TORCH_FULL=$(python -c "import torch; print(torch.__version__)")
echo "Installed torch: $INSTALLED_TORCH_FULL"

# --- Step 2: LIBERO + ManiSkill ---
echo "[2/8] Installing LIBERO + ManiSkill..."
LIBERO_DIR="$VENV_DIR/libero"
if [ ! -d "$LIBERO_DIR" ]; then
    git clone ${GITHUB_PREFIX}https://github.com/RLinf/LIBERO.git "$LIBERO_DIR"
fi
uv pip install -e "$LIBERO_DIR"
echo "export PYTHONPATH=$(realpath "$LIBERO_DIR"):\$PYTHONPATH" >> "$VENV_DIR/bin/activate"
uv pip install git+${GITHUB_PREFIX}https://github.com/haosulab/ManiSkill.git@v3.0.0b22
bash "$SCRIPT_DIR/requirements/embodied/download_assets.sh" --assets maniskill

# --- Step 3: OpenSora world model ---
echo "[3/8] Installing OpenSora world model..."
OPENSORA_DIR="$VENV_DIR/opensora"
if [ ! -d "$OPENSORA_DIR" ]; then
    git clone ${GITHUB_PREFIX}https://github.com/RLinf/opensora.git "$OPENSORA_DIR"
fi

# Install opensora's pip dependencies FIRST (before the editable install)
uv pip install -r "$SCRIPT_DIR/requirements/embodied/models/opensora.txt"
uv pip install git+${GITHUB_PREFIX}https://github.com/fangqi-Zhu/TensorNVMe.git --no-build-isolation
echo "export LD_LIBRARY_PATH=~/.tensornvme/lib:\$LD_LIBRARY_PATH" >> "$VENV_DIR/bin/activate"

# Editable install of opensora LAST so deps don't overwrite it.
# Use pip (not uv pip) for more reliable editable installs.
pip install -e "$OPENSORA_DIR" --no-deps

# Belt-and-suspenders: also add to PYTHONPATH so Ray workers always find it
OPENSORA_REAL="$(realpath "$OPENSORA_DIR")"
echo "export PYTHONPATH=${OPENSORA_REAL}:\$PYTHONPATH" >> "$VENV_DIR/bin/activate"
source "$VENV_DIR/bin/activate"

# Verify opensora is importable
python -c "from opensora.registry import MODELS; print('opensora import: OK')" || \
    { echo "ERROR: opensora package not importable after install"; exit 1; }

# --- Step 4: Apex ---
echo "[4/8] Installing Apex..."
local_torch_mm=$(python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}')")
local_py_tag="cp$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")"
apex_wheel="apex-0.1+torch${local_torch_mm}-${local_py_tag}-${local_py_tag}-linux_x86_64.whl"
apex_url="${GITHUB_PREFIX}https://github.com/RLinf/apex/releases/download/25.09/${apex_wheel}"
uv pip uninstall apex || true
uv pip install "$apex_url" 2>/dev/null || \
    (echo "Apex wheel not found, building from source..."; \
     APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install git+${GITHUB_PREFIX}https://github.com/RLinf/apex.git --no-build-isolation)

# --- Step 5: Flash Attention ---
echo "[5/8] Installing Flash Attention..."
flash_ver="2.7.4.post1"
cuda_major=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
flash_wheel="flash_attn-${flash_ver}+cu${cuda_major}torch${local_torch_mm}cxx11abiFALSE-${local_py_tag}-${local_py_tag}-linux_x86_64.whl"
flash_url="${GITHUB_PREFIX}https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_ver}/${flash_wheel}"
uv pip uninstall flash-attn || true
uv pip install "$flash_url" 2>/dev/null || \
    (echo "Flash-attn wheel not found, building from source..."; \
     uv pip install flash-attn==${flash_ver} --no-build-isolation)

# --- Step 6: OpenVLA-OFT (--no-deps to avoid torch downgrade) ---
echo "[6/8] Installing OpenVLA-OFT (without touching PyTorch)..."
pip install git+${GITHUB_PREFIX}https://github.com/moojink/openvla-oft.git --no-deps --force-reinstall
pip install git+${GITHUB_PREFIX}https://github.com/moojink/dlimp_openvla.git --no-deps

# Manually install prismatic/dlimp runtime dependencies (excluding torch/torchvision/torchaudio)
pip install draccus "timm>=0.9.10,<1.0.0" sentencepiece json-numpy jsonlines matplotlib \
    "tensorflow==2.15.0" "tensorflow_datasets==4.9.3" "tensorflow_graphics" \
    fastapi uvicorn

# --- Step 7: Re-pin torch to undo any accidental changes ---
echo "[7/8] Verifying PyTorch version..."
INSTALLED_TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
if [ "$INSTALLED_TORCH" != "$TORCH_VERSION" ]; then
    echo "WARNING: PyTorch was changed to $INSTALLED_TORCH, re-installing ${TORCH_VERSION}..."
    pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
        --index-url ${TORCH_INDEX_URL}
fi
# Re-pin numpy (later installs may have upgraded it to 2.x, breaking tensorflow)
pip install numpy==1.26.4

# --- Step 8: Restart Ray so workers inherit this environment ---
echo "[8/8] Restarting Ray..."
ray stop 2>/dev/null || true
ray start --head

# --- Verify ---
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"
python -c "
import torch
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.version.cuda}')
print(f'CUDA avail:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:           {torch.cuda.get_device_name(0)}')
assert hasattr(torch, 'Event'), 'torch.Event missing (need torch >= 2.4)'
from opensora.registry import MODELS
print(f'opensora:      OK')
import prismatic
print(f'prismatic:     OK')
print()
print('All checks passed!')
"

if [ "$USE_MIRRORS" -eq 1 ]; then
    unset UV_PYTHON_INSTALL_MIRROR UV_DEFAULT_INDEX HF_ENDPOINT
    git config --global --unset url."${GITHUB_PREFIX}github.com/".insteadOf 2>/dev/null || true
fi

echo ""
echo "Done! Activate the environment with:  source ${VENV_DIR}/bin/activate"
