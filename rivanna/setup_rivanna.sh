#!/bin/bash
# Setup script for Rivanna HPC cluster

echo "========================================"
echo "Setting up Image Analyzer on Rivanna"
echo "========================================"
echo ""

# Load modules
module purge
module load miniforge3

# Create conda environment
echo "Creating conda environment..."
conda create -n image-analyzer python=3.11 -y

# Activate environment
source activate image-analyzer

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies..."
pip install opencv-python numpy pillow scikit-learn scikit-image scipy pandas tqdm dash dash-bootstrap-components

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate: source activate image-analyzer"
echo "To submit job: sbatch slurm_train_enhancer.sh"
echo ""


