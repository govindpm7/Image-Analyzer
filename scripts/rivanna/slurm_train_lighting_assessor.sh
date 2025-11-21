#!/bin/bash
#SBATCH --job-name=lighting_train
#SBATCH --output=logs/lighting_%j.out
#SBATCH --error=logs/lighting_%j.err
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

# Generate lighting labels
echo "Generating lighting labels from LOL our485 split..."

# Try new format first (--image_dirs), fall back to old format if needed
if python scripts/generate_lighting_labels.py --help 2>&1 | grep -q "image_dirs"; then
    echo "Using new script format (--image_dirs)..."
    python scripts/generate_lighting_labels.py \
        --image_dirs data/LOLdataset/our485/low data/LOLdataset/our485/high \
        --output_csv data/lighting_labels.csv \
        --reference_low_dir data/LOLdataset/our485/low \
        --reference_high_dir data/LOLdataset/our485/high
else
    echo "⚠ Using old script format (--image_dir). Please update scripts on Rivanna."
    echo "  Generating labels for low images..."
    python scripts/generate_lighting_labels.py \
        --image_dir data/LOLdataset/our485/low \
        --output_csv data/lighting_labels_low.csv \
        --reference_low_dir data/LOLdataset/our485/low \
        --reference_high_dir data/LOLdataset/our485/high || true
    echo "  Generating labels for high images..."
    python scripts/generate_lighting_labels.py \
        --image_dir data/LOLdataset/our485/high \
        --output_csv data/lighting_labels_high.csv \
        --reference_low_dir data/LOLdataset/our485/low \
        --reference_high_dir data/LOLdataset/our485/high || true
    # Combine CSVs if both exist
    if [ -f "data/lighting_labels_low.csv" ] && [ -f "data/lighting_labels_high.csv" ]; then
        echo "  Combining CSV files..."
        python -c "import pandas as pd; df1=pd.read_csv('data/lighting_labels_low.csv'); df2=pd.read_csv('data/lighting_labels_high.csv'); pd.concat([df1,df2]).to_csv('data/lighting_labels.csv', index=False)"
    fi
fi

# Run training
echo "Starting lighting assessor training..."

# Check if script supports --reference_lol_dir
if python scripts/train_lighting_assessor.py --help 2>&1 | grep -q "reference_lol_dir"; then
    echo "Using script with --reference_lol_dir support..."
    python scripts/train_lighting_assessor.py \
        --data_dir data/training/lighting \
        --val_dir data/validation/lighting \
        --csv_path data/lighting_labels.csv \
        --reference_lol_dir data/LOLdataset/our485 \
        --batch_size 32 \
        --epochs 50 \
        --lr 0.001 \
        --save_dir weights
else
    echo "⚠ Script doesn't support --reference_lol_dir. Using basic arguments..."
    python scripts/train_lighting_assessor.py \
        --data_dir data/training/lighting \
        --val_dir data/validation/lighting \
        --csv_path data/lighting_labels.csv \
        --batch_size 32 \
        --epochs 50 \
        --lr 0.001 \
        --save_dir weights
fi

echo "Training completed at: $(date)"

