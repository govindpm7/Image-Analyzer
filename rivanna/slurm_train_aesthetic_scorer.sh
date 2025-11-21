#!/bin/bash
#SBATCH --job-name=aesthetic_train
#SBATCH --output=logs/aesthetic_%j.out
#SBATCH --error=logs/aesthetic_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load modules
module purge
module load miniforge/24.3.0-py3.11

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment - use source activate for better SLURM compatibility
source activate image-analyzer

# Verify environment activation
if [[ "$CONDA_DEFAULT_ENV" != "image-analyzer" ]]; then
    echo "ERROR: Failed to activate image-analyzer environment!"
    echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
    echo "Trying alternative activation method..."
    conda activate image-analyzer
fi

# Verify environment and packages
echo "=========================================="
echo "Environment Verification:"
echo "=========================================="
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Python path: $(python -c 'import sys; print(sys.executable)')"
echo ""
echo "Testing critical packages..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || echo "✗ PyTorch import failed!"
python -c "import tqdm; print(f'✓ tqdm: {tqdm.__version__}')" || echo "✗ tqdm import failed!"
python -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')" || echo "✗ OpenCV import failed!"
python -c "import pandas; print(f'✓ Pandas: {pandas.__version__}')" || echo "✗ Pandas import failed!"
echo "=========================================="
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/..

# Verify we're in the right directory
echo "Current directory: $(pwd)"
echo "Project structure check:"
ls -d scripts/ models/ data/ 2>/dev/null || echo "WARNING: Project structure not found!"

# Set PYTHONPATH to include project root
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Create directories
mkdir -p logs weights

# Generate aesthetic labels from LOL dataset
echo "Generating aesthetic labels from LOL dataset..."
python scripts/generate_aesthetic_labels.py \
    --image_dirs data/LOLdataset/our485/low data/LOLdataset/our485/high \
    --high_quality_dirs data/LOLdataset/our485/high \
    --output_csv data/aesthetic_labels.csv

# Run training
echo "Starting aesthetic scorer training..."
python scripts/train_aesthetic_scorer.py \
    --data_dir data/LOLdataset/our485 \
    --val_dir data/LOLdataset/eval15 \
    --csv_path data/aesthetic_labels.csv \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir weights

echo "Training completed at: $(date)"

