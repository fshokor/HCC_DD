#!/bin/bash
# =============================================================================
# HCC Drug Discovery — Conda Environment Setup Script
# =============================================================================
#
# Usage:
#   chmod +x env/setup_env.sh
#   ./env/setup_env.sh              # CPU-only (recommended — no GPU needed)
#   ./env/setup_env.sh --gpu        # CUDA 12.1 GPU support
#   ./env/setup_env.sh --skip-r     # skip R package installation
#
# After setup:
#   conda activate hcc_drug_discovery
#   jupyter lab
# =============================================================================

set -e

ENV_NAME="hcc_drug_discovery"
GPU_MODE=false
SKIP_R=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Parse arguments ───────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --gpu)    GPU_MODE=true ;;
        --skip-r) SKIP_R=true ;;
        --help)
            echo "Usage: ./env/setup_env.sh [--gpu] [--skip-r]"
            echo "  --gpu     CUDA 12.1 PyTorch (requires NVIDIA GPU)"
            echo "  --skip-r  Skip R package installation"
            exit 0 ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║      HCC Drug Discovery — Environment Setup         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Repo root   : $REPO_ROOT"
echo "  Environment : $ENV_NAME"
echo "  Mode        : $([ "$GPU_MODE" = true ] && echo 'GPU (CUDA 12.1)' || echo 'CPU-only')"
echo ""

# ── Step 1: Check conda ───────────────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "✗ conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "✓ conda: $(conda --version)"

# ── Step 2: Handle existing environment ──────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    read -p "  Environment '$ENV_NAME' already exists. Remove and recreate? [y/N] " yn
    case $yn in
        [Yy]*) conda env remove -n "$ENV_NAME" -y && echo "  ✓ Removed" ;;
        *)     echo "  Aborted."; exit 0 ;;
    esac
fi

# ── Step 3: Select environment file ──────────────────────────────────────────
if [ "$GPU_MODE" = true ]; then
    ENV_FILE="$SCRIPT_DIR/environment_gpu.yml"
else
    ENV_FILE="$SCRIPT_DIR/environment.yml"
fi

echo ""
echo "── Creating conda environment from $ENV_FILE ──"
conda env create -f "$ENV_FILE" -n "$ENV_NAME"
echo "✓ Conda environment created"

# ── Step 4: Install PyTorch Geometric (pip, version-matched) ─────────────────
echo ""
echo "── Installing PyTorch Geometric ──"

TORCH_VER=$(conda run -n "$ENV_NAME" python -c \
    "import torch; print(torch.__version__.split('+')[0])")
echo "  Detected PyTorch: $TORCH_VER"

if [ "$GPU_MODE" = true ]; then
    CUDA_TAG="cu121"
else
    CUDA_TAG="cpu"
fi

PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
echo "  PyG wheel URL: $PYG_URL"

conda run -n "$ENV_NAME" pip install \
    torch-geometric \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    --find-links "$PYG_URL" \
    --quiet

echo "✓ PyTorch Geometric installed"

# ── Step 5: Install pip-only packages ─────────────────────────────────────────
echo ""
echo "── Installing pip-only packages ──"

conda run -n "$ENV_NAME" pip install \
    celltypist>=1.6 \
    anndata2ri>=1.3 \
    --quiet

echo "✓ celltypist, anndata2ri installed"

# ── Step 6: R packages (optional) ─────────────────────────────────────────────
if [ "$SKIP_R" = false ]; then
    echo ""
    echo "── Installing R packages ──"

    if ! command -v Rscript &> /dev/null; then
        echo "  ✗ Rscript not found — skipping R package installation."
        echo "    Install R 4.3+ from https://cran.r-project.org/"
        echo "    Then run: Rscript env/r_packages.R"
    else
        R_VERSION=$(Rscript --version 2>&1 | head -1)
        echo "  R: $R_VERSION"
        Rscript "$SCRIPT_DIR/r_packages.R"
        echo "✓ R packages installed"
    fi
else
    echo ""
    echo "── Skipping R packages (--skip-r flag set) ──"
    echo "  Run manually when ready: Rscript env/r_packages.R"
fi

# ── Step 7: Register Jupyter kernel ───────────────────────────────────────────
echo ""
echo "── Registering Jupyter kernel ──"

conda run -n "$ENV_NAME" python -m ipykernel install \
    --user \
    --name "$ENV_NAME" \
    --display-name "Python ($ENV_NAME)"

echo "✓ Kernel registered as: Python ($ENV_NAME)"

# ── Step 8: Verify installation ───────────────────────────────────────────────
echo ""
echo "── Verifying installation ──"

conda run -n "$ENV_NAME" python -c "
import sys
print(f'  Python      : {sys.version.split()[0]}')

import torch
print(f'  PyTorch     : {torch.__version__}')
print(f'  CUDA avail  : {torch.cuda.is_available()}')

import torch_geometric
print(f'  PyG         : {torch_geometric.__version__}')

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
print(f'  GCN/GAT/SAGE: OK')

import scanpy
print(f'  scanpy      : {scanpy.__version__}')

import celltypist
print(f'  celltypist  : {celltypist.__version__}')

import anndata2ri
print(f'  anndata2ri  : {anndata2ri.__version__}')

import rpy2
print(f'  rpy2        : {rpy2.__version__}')

import lifelines
print(f'  lifelines   : {lifelines.__version__}')

import jupyterlab
print(f'  jupyterlab  : {jupyterlab.__version__}')

print()
print('  ✓ All packages verified')
"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║               Setup complete!                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Activate the environment:"
echo "       conda activate $ENV_NAME"
echo ""
echo "  2. Download the dataset (first time only, ~204 MB):"
echo "       python scripts/data_download.py"
echo ""
echo "  3. Open JupyterLab:"
echo "       jupyter lab"
echo ""
echo "  4. Select kernel: 'Python ($ENV_NAME)'"
echo "     Run notebooks in order starting with 01_preprocessing.ipynb"
echo ""
