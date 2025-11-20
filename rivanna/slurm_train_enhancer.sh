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

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load modules (adjust based on Rivanna's available modules)
module purge
module load miniforge/24.3.0-py3.11

# Initialize conda for bash and activate environment
eval "$(conda shell.bash hook)"
conda activate image-analyzer

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory (assuming script is in rivanna/ subdirectory)
cd $SLURM_SUBMIT_DIR
cd ..

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p weights

# Run training
echo "Starting training..."
python scripts/train_enhancer_lol.py \
    --batch_size 16 \
    --epochs 100 \
    --input_size 256 \
    --lr 0.0001 \
    --save_dir weights

echo "Training completed at: $(date)"

