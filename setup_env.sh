#!/bin/bash
# Setup conda environment for CEBRA pipeline on Savio
# Usage: bash setup_env.sh

set -e

ENV_DIR="$HOME/envs/cebra"

echo "=== CEBRA Pipeline Environment Setup ==="

# Load modules
module load anaconda3/2024.02-1-11.4

# Remove existing env if requested
if [ "$1" == "--clean" ] && [ -d "$ENV_DIR" ]; then
    echo "Removing existing environment..."
    conda env remove -p "$ENV_DIR" -y
fi

# Create environment
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating conda environment at $ENV_DIR..."
    conda create -p "$ENV_DIR" python=3.10 -y
else
    echo "Environment already exists at $ENV_DIR"
fi

# Activate
source activate "$ENV_DIR"

# Install PyTorch with CUDA
echo "Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install CEBRA
echo "Installing CEBRA..."
pip install cebra

# Install CatBoost with GPU support
echo "Installing CatBoost..."
pip install catboost

# Install remaining dependencies
echo "Installing other dependencies..."
pip install numpy pandas scikit-learn matplotlib scipy tqdm jupyter ipykernel

# Install ffmpeg for animations
conda install -c conda-forge ffmpeg -y

# Register Jupyter kernel
python -m ipykernel install --user --name cebra --display-name "CEBRA Pipeline"

echo ""
echo "=== Setup Complete ==="
echo "Environment: $ENV_DIR"
echo "Jupyter kernel: 'CEBRA Pipeline'"
echo ""
echo "To activate: source activate $ENV_DIR"
echo "To use in OOD Jupyter: select 'CEBRA Pipeline' kernel"
