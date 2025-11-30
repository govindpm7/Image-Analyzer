# Running Training on Rivanna HPC Cluster

## Prerequisites

- Rivanna account and access
- SSH access to Rivanna
- Your project code and data uploaded

## Step 1: Transfer Files to Rivanna

### Option A: Using SCP

```bash
# From your local machine
scp -r /path/to/Image-Analyzer username@rivanna.hpc.virginia.edu:/home/username/

# Or just the necessary files
scp -r scripts/ models/ data/ username@rivanna.hpc.virginia.edu:/home/username/Image-Analyzer/
```

### Option B: Using Git (Recommended)

```bash
# SSH into Rivanna
ssh username@rivanna.hpc.virginia.edu

# Clone your repository
cd ~
git clone <your-repo-url> Image-Analyzer
cd Image-Analyzer
```

### Option C: Using FileZilla or WinSCP

Use a GUI SFTP client to transfer files to:
- Host: `rivanna.hpc.virginia.edu`
- Username: Your UVA computing ID
- Port: 22

## Step 2: Setup Environment on Rivanna

### SSH into Rivanna

```bash
ssh username@rivanna.hpc.virginia.edu
```

### Run Setup Script

```bash
cd Image-Analyzer
cd rivanna
chmod +x setup_rivanna.sh
./setup_rivanna.sh
```

### Or Manual Setup

```bash
# Load modules
module purge
module load miniforge3
module load cuda/11.8.0
module load cudnn/8.6.0

# Create environment
conda create -n image-analyzer python=3.11 -y
source activate image-analyzer

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install opencv-python numpy pillow scikit-learn scikit-image scipy pandas tqdm dash dash-bootstrap-components
```

## Step 3: Verify Dataset Location

Make sure your datasets are accessible:

```bash
# Check LOL dataset structure (for enhancer training)
ls -la data/LOLdataset/our485/
# Should see: high/ and low/ directories

ls -la data/LOLdataset/eval15/
# Should see: high/ and low/ directories

# Check other datasets
# Blur detector: data/training/blur/ should have sharp/ and blurry/ subdirectories
# Lighting assessor: data/training/lighting/ with CSV or labeled images
# Aesthetic scorer: data/training/aesthetic/ with CSV or JSON metadata
```

## Step 4: Submit Training Job

### Low-Light Enhancement Training

```bash
cd rivanna
sbatch slurm_train_enhancer.sh
```

**Training time:** 24 hours, 100 epochs

### Lighting Assessor Training

```bash
cd rivanna
sbatch slurm_train_lighting_assessor.sh
```

**Training time:** 12 hours, 50 epochs

**Note:** Ensure your dataset includes low-light images for proper training. Use `scripts/generate_lighting_labels.py` to create labels.

### Blur Detector Training

```bash
cd rivanna
sbatch slurm_train_blur_detector.sh
```

**Training time:** 12 hours, 50 epochs

### Aesthetic Scorer Training

```bash
cd rivanna
sbatch slurm_train_aesthetic_scorer.sh
```

**Training time:** 12 hours, 50 epochs

### Custom Training

Edit the SLURM script or submit with custom parameters:

```bash
# Create a custom script
cat > my_training.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=my_train
#SBATCH --output=logs/my_%j.out
#SBATCH --error=logs/my_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=32G

module purge
module load miniforge3
module load cuda/11.8.0
module load cudnn/8.6.0

source activate image-analyzer
cd $SLURM_SUBMIT_DIR/..
mkdir -p logs weights

python scripts/train_enhancer_lol.py \
    --batch_size 16 \
    --epochs 50 \
    --input_size 256 \
    --save_dir weights
EOF

sbatch my_training.sh
```

## Step 5: Monitor Jobs

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Check job details
scontrol show job <job_id>

# View output in real-time
tail -f logs/enhancer_*.out
tail -f logs/lighting_*.out
tail -f logs/blur_*.out
tail -f logs/aesthetic_*.out
```

### Cancel Job

```bash
scancel <job_id>
```

### Check Job Output

```bash
# View output file
cat logs/enhancer_<job_id>.out

# View error file
cat logs/enhancer_<job_id>.err

# Follow output live
tail -f logs/enhancer_<job_id>.out
```

## Step 6: Download Results

After training completes, download the weights:

```bash
# From your local machine - download all weights
scp username@rivanna.hpc.virginia.edu:/home/username/Image-Analyzer/weights/*.pth ./

# Or download specific models
scp username@rivanna.hpc.virginia.edu:/home/username/Image-Analyzer/weights/enhancer_best.pth ./
scp username@rivanna.hpc.virginia.edu:/home/username/Image-Analyzer/weights/lighting_assessor_best.pth ./
scp username@rivanna.hpc.virginia.edu:/home/username/Image-Analyzer/weights/blur_detector_best.pth ./
scp username@rivanna.hpc.virginia.edu:/home/username/Image-Analyzer/weights/aesthetic_scorer_best.pth ./
```

Or use FileZilla/WinSCP to download the entire `weights/` directory.

## SLURM Script Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--job-name` | Name of the job | `enhancer_train` |
| `--time` | Maximum runtime | `24:00:00` (24 hours) |
| `--gres=gpu:1` | Request 1 GPU | Required for training |
| `--partition` | GPU partition | `gpu` |
| `--mem` | Memory request | `32G` |
| `--cpus-per-task` | CPU cores | `8` |
| `--output` | Output file | `logs/enhancer_%j.out` |
| `--error` | Error file | `logs/enhancer_%j.err` |

## Common Rivanna Commands

```bash
# Check available modules
module avail

# Check GPU availability
sinfo -p gpu

# Check your job history
sacct -u $USER

# Check disk quota
quota -s

# Check available storage
df -h
```

## Troubleshooting

### "Module not found"

Check available modules:
```bash
module avail miniforge
module avail cuda
```

### "GPU not available"

Check GPU partition:
```bash
sinfo -p gpu
```

Try different partition or wait for GPU availability.

### "Out of memory"

Reduce batch size in training script:
```bash
--batch_size 8  # Instead of 16
```

### "Dataset not found"

Check dataset path:
```bash
ls -la data/LOLdataset/
```

Make sure dataset is uploaded correctly.

### "Conda environment not found"

Recreate environment:
```bash
conda create -n image-analyzer python=3.11 -y
source activate image-analyzer
# Reinstall packages
```

## Best Practices

1. **Monitor first job** - Watch the output to catch errors early
2. **Save checkpoints** - Scripts save best models automatically
3. **Use appropriate time limits** - Don't request more time than needed
4. **Check GPU availability** - Use `sinfo -p gpu` before submitting
5. **Train models in order** - Start with lighting assessor and blur detector (faster), then enhancer and aesthetic scorer

## Example Workflow

```bash
# 1. SSH into Rivanna
ssh username@rivanna.hpc.virginia.edu

# 2. Navigate to project
cd Image-Analyzer

# 3. Setup environment (first time only)
cd rivanna
./setup_rivanna.sh

# 4. Submit training jobs
sbatch slurm_train_lighting_assessor.sh
sbatch slurm_train_blur_detector.sh
sbatch slurm_train_enhancer.sh
sbatch slurm_train_aesthetic_scorer.sh

# 5. Monitor
squeue -u $USER
tail -f logs/*.out

# 6. Download results when done
# (from local machine)
scp username@rivanna:/home/username/Image-Analyzer/weights/*.pth ./
```

## Additional Resources

- Rivanna User Guide: https://www.rc.virginia.edu/userinfo/rivanna/
- SLURM Documentation: https://slurm.schedmd.com/documentation.html
- UVA Research Computing: https://www.rc.virginia.edu/


