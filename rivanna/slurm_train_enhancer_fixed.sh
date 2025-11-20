#!/bin/bash
#SBATCH --job-name=enhancer_train
#SBATCH --output=logs/enhancer_%j.out
#SBATCH --error=logs/enhancer_%j.err
#SBATCH --time=24:00:00
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

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate environment
conda activate image-analyzer

# Verify activation
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/..

# Create directories
mkdir -p logs weights

# Run training
echo "Starting training..."
python scripts/train_enhancer_lol.py \
    --batch_size 16 \
    --epochs 100 \
    --input_size 256 \
    --lr 0.0001 \
    --save_dir weights

# Complete
echo "Training completed at: $(date)"
