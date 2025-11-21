#!/bin/bash
# Setup script for Rivanna HPC cluster
# 
# This script creates the complete 'image-analyzer' conda environment
# with ALL required dependencies for the Image Analyzer project.
# 
# IMPORTANT: This script is designed for miniforge-only environments.
# Rivanna uses miniforge, NOT standard Anaconda.
# 
# It will:
# 1. Load miniforge module
# 2. Create conda environment 'image-analyzer' with Python 3.11
# 3. Install PyTorch with CUDA 11.8 support
# 4. Install all core, ML, and web dependencies
# 5. Verify all packages are correctly installed
#
# Run this script ONCE before submitting any SLURM jobs.
# All SLURM scripts use this environment.

echo "========================================"
echo "Setting up Image Analyzer on Rivanna"
echo "Creating 'image-analyzer' conda environment"
echo "Using miniforge (NOT Anaconda)"
echo "========================================"
echo ""

# Load modules - miniforge only
module purge
echo "Loading miniforge module..."
if ! module load miniforge/24.3.0-py3.11 2>/dev/null; then
    echo "ERROR: miniforge/24.3.0-py3.11 module not found!"
    echo "Available miniforge modules:"
    module avail miniforge 2>&1 | grep -i miniforge
    exit 1
fi

# Verify miniforge is loaded
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda command not found after loading miniforge module!"
    echo "Please check that miniforge module is correctly installed."
    exit 1
fi

echo "✓ Miniforge loaded successfully"
echo "Conda location: $(which conda)"
echo ""

# Initialize conda for bash shell
# This is required for miniforge to work properly
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

# Verify conda is working
if ! conda --version &> /dev/null; then
    echo "ERROR: conda initialization failed!"
    exit 1
fi

echo "✓ Conda initialized: $(conda --version)"
echo ""

# Create conda environment
# Using conda (from miniforge) - NOT anaconda
echo "Creating conda environment 'image-analyzer'..."
if conda env list | grep -q "image-analyzer"; then
    echo "⚠ Warning: Environment 'image-analyzer' already exists."
    echo "Removing existing environment and recreating..."
    conda env remove -n image-analyzer -y
    conda create -n image-analyzer python=3.11 -y
else
    conda create -n image-analyzer python=3.11 -y
fi

# Activate environment
echo "Activating environment..."
conda activate image-analyzer

# Verify activation
if [[ "$CONDA_DEFAULT_ENV" != "image-analyzer" ]]; then
    echo "ERROR: Failed to activate image-analyzer environment!"
    exit 1
fi

echo "✓ Environment activated: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo "Installing core dependencies..."
pip install opencv-python numpy pillow scikit-learn scikit-image scipy pandas tqdm matplotlib seaborn

# Install web/dashboard dependencies
echo "Installing web/dashboard dependencies..."
pip install dash dash-bootstrap-components flask

# Install ML dependencies
echo "Installing ML dependencies..."
pip install tensorboard albumentations timm

# Note: Since we're using a conda environment (miniforge), we should NOT need
# --break-system-packages flag. If you encounter "externally-managed-environment" 
# errors, it means the environment wasn't activated properly.

# Verify installation
echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"
echo "Testing core packages..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'✓ Torchvision: {torchvision.__version__}')"
python -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')"
python -c "import numpy as np; print(f'✓ NumPy: {np.__version__}')"
python -c "import pandas as pd; print(f'✓ Pandas: {pd.__version__}')"
python -c "import PIL; print(f'✓ Pillow: {PIL.__version__}')"
python -c "import tqdm; print(f'✓ tqdm: {tqdm.__version__}')"
python -c "import sklearn; print(f'✓ scikit-learn: {sklearn.__version__}')"
python -c "import skimage; print(f'✓ scikit-image: {skimage.__version__}')"
python -c "import scipy; print(f'✓ SciPy: {scipy.__version__}')"
echo ""
echo "Testing ML packages..."
python -c "import albumentations; print(f'✓ Albumentations: {albumentations.__version__}')"
python -c "import timm; print(f'✓ timm: {timm.__version__}')"
python -c "import tensorboard; print(f'✓ TensorBoard: {tensorboard.__version__}')"
echo ""
echo "Testing web/dashboard packages..."
python -c "import dash; print(f'✓ Dash: {dash.__version__}')"
python -c "import dash_bootstrap_components; print(f'✓ dash-bootstrap-components: installed')"
python -c "import flask; print(f'✓ Flask: {flask.__version__}')"
echo ""
echo "Testing visualization packages..."
python -c "import matplotlib; print(f'✓ Matplotlib: {matplotlib.__version__}')"
python -c "import seaborn; print(f'✓ Seaborn: {seaborn.__version__}')"
echo ""
echo "All packages verified successfully!"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "IMPORTANT: To use conda in this shell session, run:"
echo "  module load miniforge/24.3.0-py3.11"
echo "  eval \"\$(conda shell.bash hook)\""
echo "  conda activate image-analyzer"
echo ""
echo "Or simply run:"
echo "  source ~/.bashrc  # if conda init was added to bashrc"
echo ""
echo "To submit jobs (SLURM scripts handle conda automatically):"
echo "  cd rivanna && sbatch slurm_train_enhancer.sh"
echo ""
echo "Note: SLURM scripts automatically load miniforge and activate the environment."
echo ""


