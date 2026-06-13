#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Setup MapTR environment
#
# Assumed directory structure:
#
# hdmap-driving-workspace/
# ├── checkpoints/
# │   └── maptr/
# │       └── official/
# │
# └── repos/
#     └── hdmap-model-bench/
#         ├── scripts/
#         │   └── setup_maptr_env.sh
#         └── models/
#             └── maptr/
#                 └── external/
#                     └── MapTR/
#
# ------------------------------------------------------------
# Required system packages
# ------------------------------------------------------------
#
# MapTR uses an old OpenMMLab stack:
#
#   - PyTorch 1.9.1 + CUDA 11.1
#   - mmcv-full 1.4.0
#   - mmdet3d 0.17.2
#
# On Ubuntu 22.04, the default compiler is usually GCC 11.
# However, GCC 11 is too new for building the CUDA extensions
# used by mmdet3d 0.17.2 / MapTR.
#
# Therefore, gcc-9 and g++-9 must be installed before running
# this setup script.
#
# Install them with:
#
#   sudo apt update
#   sudo apt install -y gcc-9 g++-9
#
# This script will use:
#
#   CC=/usr/bin/gcc-9
#   CXX=/usr/bin/g++-9
#   CUDAHOSTCXX=/usr/bin/g++-9
#
# ------------------------------------------------------------
# Usage
# ------------------------------------------------------------
#
#   cd ~/hdmap-driving-workspace/repos/hdmap-model-bench
#   bash scripts/setup_maptr_env.sh
#
# Recreate environment:
#
#   RESET_MAPTR_ENV=1 bash scripts/setup_maptr_env.sh
#
# Optional:
#
#   MAPTR_ENV_NAME=maptr bash scripts/setup_maptr_env.sh
#   MAPTR_PYTHON_VERSION=3.8 bash scripts/setup_maptr_env.sh
#
# ============================================================

ENV_NAME="${MAPTR_ENV_NAME:-maptr}"
PYTHON_VERSION="${MAPTR_PYTHON_VERSION:-3.8}"
RESET_ENV="${RESET_MAPTR_ENV:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"

MAPTR_ROOT="${REPO_ROOT}/models/maptr/external/MapTR"
MAPTR_MMD3D_ROOT="${MAPTR_ROOT}/mmdetection3d"
MAPTR_GKT_ROOT="${MAPTR_ROOT}/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn"

MAPTR_CKPT_ROOT="${WORKSPACE_ROOT}/checkpoints/maptr"
MAPTR_OFFICIAL_CKPT_DIR="${MAPTR_CKPT_ROOT}/official"

echo "============================================================"
echo "[INFO] Setup MapTR environment"
echo "============================================================"
echo "[INFO] Repository root       : ${REPO_ROOT}"
echo "[INFO] Workspace root        : ${WORKSPACE_ROOT}"
echo "[INFO] MapTR root            : ${MAPTR_ROOT}"
echo "[INFO] mmdetection3d root    : ${MAPTR_MMD3D_ROOT}"
echo "[INFO] GKT root              : ${MAPTR_GKT_ROOT}"
echo "[INFO] Checkpoint root       : ${MAPTR_CKPT_ROOT}"
echo "[INFO] Official checkpoint   : ${MAPTR_OFFICIAL_CKPT_DIR}"
echo "[INFO] Conda env name        : ${ENV_NAME}"
echo "[INFO] Python version        : ${PYTHON_VERSION}"
echo "[INFO] Reset env             : ${RESET_ENV}"
echo "============================================================"

# ------------------------------------------------------------
# Check required commands
# ------------------------------------------------------------

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not found."
    exit 1
fi

if ! command -v wget >/dev/null 2>&1; then
    echo "[ERROR] wget command not found."
    echo "[ERROR] Please install wget:"
    echo "  sudo apt install -y wget"
    exit 1
fi

# ------------------------------------------------------------
# Check MapTR submodule
# ------------------------------------------------------------

if [ ! -d "${MAPTR_ROOT}" ]; then
    echo "[ERROR] MapTR repository not found:"
    echo "  ${MAPTR_ROOT}"
    echo ""
    echo "[ERROR] Please initialize submodules first:"
    echo "  cd ${REPO_ROOT}"
    echo "  git submodule update --init --recursive"
    exit 1
fi

if [ ! -d "${MAPTR_MMD3D_ROOT}" ]; then
    echo "[ERROR] mmdetection3d directory not found:"
    echo "  ${MAPTR_MMD3D_ROOT}"
    exit 1
fi

if [ ! -d "${MAPTR_GKT_ROOT}" ]; then
    echo "[ERROR] geometric_kernel_attn directory not found:"
    echo "  ${MAPTR_GKT_ROOT}"
    echo ""
    echo "[ERROR] Please check MapTR branch. Expected branch is maptrv2."
    exit 1
fi

# ------------------------------------------------------------
# Check gcc-9 / g++-9
# ------------------------------------------------------------

if ! command -v gcc-9 >/dev/null 2>&1 || ! command -v g++-9 >/dev/null 2>&1; then
    echo "[ERROR] gcc-9 / g++-9 not found."
    echo ""
    echo "Please install them first:"
    echo "  sudo apt update"
    echo "  sudo apt install -y gcc-9 g++-9"
    echo ""
    exit 1
fi

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9

echo "[INFO] Using host compiler for CUDA build:"
echo "[INFO] CC=${CC}"
echo "[INFO] CXX=${CXX}"
echo "[INFO] CUDAHOSTCXX=${CUDAHOSTCXX}"

# ------------------------------------------------------------
# Check nvcc
# ------------------------------------------------------------

if command -v nvcc >/dev/null 2>&1; then
    echo "[INFO] nvcc path:"
    which nvcc
    nvcc --version
else
    echo "[ERROR] nvcc not found."
    echo "[ERROR] CUDA toolkit is required to build MapTR CUDA extensions."
    exit 1
fi

# ------------------------------------------------------------
# Enable conda in non-interactive shell
# ------------------------------------------------------------

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ------------------------------------------------------------
# Remove env if requested
# ------------------------------------------------------------

if [ "${RESET_ENV}" = "1" ]; then
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "[INFO] Removing existing conda environment: ${ENV_NAME}"
        conda deactivate || true
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "[INFO] Conda environment '${ENV_NAME}' does not exist. Skip removal."
    fi
fi

# ------------------------------------------------------------
# Create and activate conda environment
# ------------------------------------------------------------

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Conda environment '${ENV_NAME}' already exists."
else
    echo "[INFO] Creating conda environment '${ENV_NAME}'..."
    conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo "[INFO] Activating conda environment '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

echo "[INFO] Python path:"
which python
python --version

# ------------------------------------------------------------
# Keep build tools old enough for legacy OpenMMLab stack
# ------------------------------------------------------------

echo "[INFO] Installing base build tools..."
python -m pip install --upgrade "pip<24"
python -m pip install "setuptools<60" "wheel" "ninja"

# ------------------------------------------------------------
# Install PyTorch
# ------------------------------------------------------------

echo "[INFO] Installing PyTorch 1.9.1 + CUDA 11.1..."
pip install \
    torch==1.9.1+cu111 \
    torchvision==0.10.1+cu111 \
    torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ------------------------------------------------------------
# Install pinned legacy dependencies before mmdet3d
# ------------------------------------------------------------

echo "[INFO] Installing pinned dependencies for mmdet3d 0.17.x..."

conda install -c conda-forge \
    numpy=1.19.5 \
    numba=0.48.0 \
    networkx=2.2 \
    plyfile \
    scikit-image=0.19.3 \
    tensorboard=2.6.0 \
    protobuf=3.20.3 \
    matplotlib \
    tqdm \
    yapf \
    terminaltables \
    shapely \
    scipy \
    pandas \
    pyquaternion \
    pyyaml \
    opencv \
    -y

pip install trimesh==2.35.39

# ------------------------------------------------------------
# Install OpenMMLab dependencies
# ------------------------------------------------------------

echo "[INFO] Installing mmcv-full==1.4.0..."
pip install mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

echo "[INFO] Installing mmdet==2.14.0..."
pip install mmdet==2.14.0

echo "[INFO] Installing mmsegmentation==0.14.1..."
pip install mmsegmentation==0.14.1

echo "[INFO] Installing timm..."
pip install timm

# ------------------------------------------------------------
# Clean old build files
# ------------------------------------------------------------

echo "[INFO] Cleaning previous build files..."

cd "${MAPTR_MMD3D_ROOT}"
rm -rf build
rm -rf mmdet3d.egg-info
find . -name "*.so" -delete

cd "${MAPTR_GKT_ROOT}"
rm -rf build
find . -name "*.so" -delete

# ------------------------------------------------------------
# Install mmdetection3d without dependency resolution
# ------------------------------------------------------------

echo "[INFO] Installing MapTR bundled mmdetection3d without dependency resolution..."

cd "${MAPTR_MMD3D_ROOT}"

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9

pip install -v -e . --no-deps

# ------------------------------------------------------------
# Build and install geometric kernel attention
# ------------------------------------------------------------

echo "[INFO] Building and installing geometric kernel attention..."

cd "${MAPTR_GKT_ROOT}"

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDAHOSTCXX=/usr/bin/g++-9

python setup.py build install

# ------------------------------------------------------------
# Install MapTR requirements carefully
# ------------------------------------------------------------

echo "[INFO] Installing MapTR requirements carefully..."

cd "${MAPTR_ROOT}"

if [ -f "requirement.txt" ]; then
    echo "[INFO] requirement.txt found."
    echo "[INFO] Installing requirements with --no-deps to avoid upgrading pinned legacy packages..."
    pip install -r requirement.txt --no-deps || {
        echo "[WARN] Failed to install requirement.txt with --no-deps."
        echo "[WARN] Continue because core dependencies were installed manually."
    }
    echo "[INFO] Installing av2 runtime dependency required for import upath..."
    pip install "universal-pathlib<0.3"
else
    echo "[WARN] requirement.txt not found. Skipping."
fi

# ------------------------------------------------------------
# Re-pin legacy dependencies after requirement install
# ------------------------------------------------------------

echo "[INFO] Re-pinning legacy dependencies..."

conda install -c conda-forge \
    numpy=1.19.5 \
    numba=0.48.0 \
    networkx=2.2 \
    plyfile \
    scikit-image=0.19.3 \
    tensorboard=2.6.0 \
    protobuf=3.20.3 \
    -y

pip install trimesh==2.35.39

# ------------------------------------------------------------
# Prepare checkpoint directories
# ------------------------------------------------------------

echo "[INFO] Preparing checkpoint directories..."

mkdir -p "${MAPTR_OFFICIAL_CKPT_DIR}"

mkdir -p "${REPO_ROOT}/checkpoints"

if [ -L "${REPO_ROOT}/checkpoints/maptr" ]; then
    echo "[INFO] Repository checkpoint symlink already exists:"
    echo "  ${REPO_ROOT}/checkpoints/maptr"
elif [ -e "${REPO_ROOT}/checkpoints/maptr" ]; then
    echo "[WARN] ${REPO_ROOT}/checkpoints/maptr already exists but is not a symlink."
    echo "[WARN] Please check manually."
else
    ln -s ../../../checkpoints/maptr "${REPO_ROOT}/checkpoints/maptr"
    echo "[INFO] Created symlink:"
    echo "  ${REPO_ROOT}/checkpoints/maptr -> ../../../checkpoints/maptr"
fi

# ------------------------------------------------------------
# Download official pretrained backbone checkpoints
# ------------------------------------------------------------

echo "[INFO] Downloading official pretrained backbone checkpoints..."

cd "${MAPTR_OFFICIAL_CKPT_DIR}"

if [ ! -f "resnet50-19c8e357.pth" ]; then
    wget https://download.pytorch.org/models/resnet50-19c8e357.pth
else
    echo "[INFO] resnet50-19c8e357.pth already exists."
fi

if [ ! -f "resnet18-f37072fd.pth" ]; then
    wget https://download.pytorch.org/models/resnet18-f37072fd.pth
else
    echo "[INFO] resnet18-f37072fd.pth already exists."
fi

# ------------------------------------------------------------
# Create MapTR/ckpts symlink to workspace official checkpoints
# ------------------------------------------------------------

echo "[INFO] Preparing MapTR/ckpts symlink..."

cd "${MAPTR_ROOT}"

if [ -L "ckpts" ]; then
    echo "[INFO] ${MAPTR_ROOT}/ckpts already exists as a symlink."
elif [ -d "ckpts" ]; then
    echo "[WARN] ${MAPTR_ROOT}/ckpts already exists as a directory."
    echo "[WARN] Keeping existing directory."
    echo "[WARN] If needed, move its contents to:"
    echo "  ${MAPTR_OFFICIAL_CKPT_DIR}"
    echo "[WARN] Then replace it with symlink manually."
else
    ln -s "${MAPTR_OFFICIAL_CKPT_DIR}" ckpts
    echo "[INFO] Created symlink:"
    echo "  ${MAPTR_ROOT}/ckpts -> ${MAPTR_OFFICIAL_CKPT_DIR}"
fi

# ------------------------------------------------------------
# Verify installation
# ------------------------------------------------------------

echo "============================================================"
echo "[INFO] Verifying installation"
echo "============================================================"

python - <<'PY'
import sys
import os

print("python:", sys.version)
print("python executable:", sys.executable)

import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

import numpy
print("numpy:", numpy.__version__)

import numba
print("numba:", numba.__version__)

import networkx
print("networkx:", networkx.__version__)

import skimage
print("scikit-image:", skimage.__version__)

import tensorboard
print("tensorboard:", tensorboard.__version__)

import trimesh
print("trimesh:", trimesh.__version__)

import mmcv
print("mmcv:", mmcv.__version__)

import mmdet
print("mmdet:", mmdet.__version__)

import mmseg
print("mmseg:", mmseg.__version__)

import mmdet3d
print("mmdet3d: imported")

try:
    from mmdet3d.ops import Voxelization
    print("mmdet3d.ops.Voxelization: imported")
except Exception as exc:
    print("[WARN] Failed to import mmdet3d.ops.Voxelization:", repr(exc))

try:
    import timm
    print("timm:", timm.__version__)
except Exception as exc:
    print("[WARN] Failed to import timm:", repr(exc))

print("verification completed")
PY

echo "============================================================"
echo "[INFO] MapTR environment setup completed."
echo "============================================================"
echo ""
echo "Activate environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Important paths:"
echo "  MapTR root:"
echo "    ${MAPTR_ROOT}"
echo ""
echo "  Official checkpoints:"
echo "    ${MAPTR_OFFICIAL_CKPT_DIR}"
echo ""
echo "  hdmap-model-bench checkpoint symlink:"
echo "    ${REPO_ROOT}/checkpoints/maptr"
echo ""
echo "  MapTR ckpts:"
echo "    ${MAPTR_ROOT}/ckpts"
echo ""