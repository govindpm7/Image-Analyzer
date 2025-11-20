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

# Generate/update lighting labels for LOL our485 low/high images
echo "Generating lighting labels from LOL our485 split..."
python scripts/generate_lighting_labels.py \
    --image_dirs data/LOLdataset/our485/low data/LOLdataset/our485/high \
    --output_csv data/lighting_labels.csv \
    --reference_low_dir data/LOLdataset/our485/low \
    --reference_high_dir data/LOLdataset/our485/high

# Run training
# Note: You need to provide --data_dir with images that include low-light examples
# The dataset should have a CSV with 'path' and 'brightness_score' columns
echo "Starting lighting assessor training..."
echo "IMPORTANT: Ensure your dataset includes low-light images for proper training!"
python scripts/train_lighting_assessor.py \
    --data_dir data/training/lighting \
    --val_dir data/validation/lighting \
    --csv_path data/lighting_labels.csv \
    --reference_lol_dir data/LOLdataset/our485 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir weights

echo "Training completed at: $(date)"

