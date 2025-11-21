# Complete Rivanna Pipeline Guide

## Overview
This guide covers the complete pipeline from initial setup to running training jobs on Rivanna HPC.

---

## Step 1: Transfer Files to Rivanna

### Option A: Using Git (Recommended)
```bash
# SSH into Rivanna
ssh username@rivanna.hpc.virginia.edu

# Clone repository
cd ~
git clone <your-repo-url> Image-Analyzer
cd Image-Analyzer
```

### Option B: Using SCP
```bash
# From your local machine
scp -r /path/to/Image-Analyzer username@rivanna.hpc.virginia.edu:~/
```

### Option C: Using SFTP Client (FileZilla/WinSCP)
- Host: `rivanna.hpc.virginia.edu`
- Username: Your UVA computing ID
- Port: 22
- Transfer entire `Image-Analyzer` directory

---

## Step 2: Initial Setup (First Time Only)

### SSH into Rivanna
```bash
ssh username@rivanna.hpc.virginia.edu
```

### Navigate to Project
```bash
cd Image-Analyzer
```

### Run Setup Script
```bash
cd rivanna
chmod +x setup_rivanna.sh
./setup_rivanna.sh
```

This will:
- Load required modules
- Create conda environment `image-analyzer`
- Install PyTorch with CUDA support
- Install all dependencies

**Note:** Setup takes 5-10 minutes. You only need to do this once.

---

## Step 3: Verify Project Structure

After setup, verify your directory structure:

```bash
# From Image-Analyzer root directory
cd ~/Image-Analyzer

# Check project structure
ls -la
# Should see: scripts/, models/, data/, rivanna/, utils/

# Check SLURM scripts
ls -la rivanna/
# Should see: slurm_train_*.sh files

# Verify dataset locations (if already uploaded)
ls -la data/LOLdataset/our485/
# Should see: high/ and low/ directories

# Check that scripts exist
ls -la scripts/train_*.py
```

---

## Step 4: Prepare Datasets

### For Low-Light Enhancement (LOL Dataset)
```bash
# Verify LOL dataset structure
ls data/LOLdataset/our485/low/   # Training low-light images
ls data/LOLdataset/our485/high/  # Training well-lit images
ls data/LOLdataset/eval15/low/   # Validation low-light images
ls data/LOLdataset/eval15/high/  # Validation well-lit images
```

### For Lighting Assessor
The script automatically generates labels, but verify data exists:
```bash
# Training data
ls data/training/lighting/

# Validation data
ls data/validation/lighting/
```

### For Blur Detector
```bash
# Should have sharp/ and blurry/ subdirectories
ls data/training/blur/
ls data/validation/blur/
```

### For Aesthetic Scorer
```bash
# Should have CSV or JSON metadata files
ls data/training/aesthetic/
ls data/validation/aesthetic/
```

---

## Step 5: Submit Training Jobs

**IMPORTANT:** 
- Always submit jobs from the `rivanna/` directory or use absolute paths
- **All paths in SLURM scripts are Linux paths for Rivanna, NOT Windows paths**
- The scripts use relative paths (e.g., `data/LOLdataset/...`) which work once the project is on Rivanna
- Your local Windows paths (like `C:\Users\...`) will NOT work - the jobs run on Rivanna's Linux cluster

### Method 1: From rivanna/ directory (Recommended)
```bash
cd ~/Image-Analyzer/rivanna

# Submit individual jobs
sbatch slurm_train_lighting_assessor.sh
sbatch slurm_train_blur_detector.sh
sbatch slurm_train_enhancer.sh
sbatch slurm_train_aesthetic_scorer.sh
```

### Method 2: From project root with absolute path
```bash
cd ~/Image-Analyzer
sbatch rivanna/slurm_train_lighting_assessor.sh
```

### Path Explanation
The SLURM scripts contain paths like:
- `data/LOLdataset/our485/low` → This is a **relative path** on Rivanna
- When the script runs, it does `cd $SLURM_SUBMIT_DIR/..` to go to the project root
- So `data/...` resolves to `~/Image-Analyzer/data/...` on Rivanna
- **Do NOT edit these to use your local Windows paths** - they must stay as Linux paths

### Job Details

| Job | Time Limit | Epochs | Description |
|-----|------------|--------|-------------|
| `slurm_train_lighting_assessor.sh` | 12 hours | 50 | Generates labels + trains lighting assessor |
| `slurm_train_blur_detector.sh` | 12 hours | 50 | Trains blur detector |
| `slurm_train_enhancer.sh` | 24 hours | 100 | Trains low-light enhancer |
| `slurm_train_aesthetic_scorer.sh` | 12 hours | 50 | Trains aesthetic scorer |

---

## Step 6: Monitor Jobs

### Check Job Status
```bash
# List all your jobs
squeue -u $USER

# Check specific job details
scontrol show job <job_id>

# Check job history
sacct -u $USER
```

### View Output Logs
```bash
# View output in real-time
tail -f logs/lighting_*.out
tail -f logs/blur_*.out
tail -f logs/enhancer_*.out
tail -f logs/aesthetic_*.out

# View specific job output
tail -f logs/lighting_<job_id>.out

# View error logs
cat logs/lighting_<job_id>.err
```

### Cancel a Job
```bash
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

---

## Step 7: Verify Training Results

### Check for Saved Models
```bash
# List saved weights
ls -lh weights/

# Should see files like:
# - lighting_assessor_best.pth
# - blur_detector_best.pth
# - enhancer_best.pth
# - aesthetic_scorer_best.pth
```

### Check Training Metrics
```bash
# View training output
cat logs/lighting_<job_id>.out | grep -i "loss\|accuracy\|psnr\|mae"
```

---

## Step 8: Download Results

### Using SCP (from local machine)
```bash
# Download all weights
scp username@rivanna.hpc.virginia.edu:~/Image-Analyzer/weights/*.pth ./

# Download specific model
scp username@rivanna.hpc.virginia.edu:~/Image-Analyzer/weights/lighting_assessor_best.pth ./

# Download logs
scp username@rivanna.hpc.virginia.edu:~/Image-Analyzer/logs/*.out ./
```

### Using SFTP Client
- Connect to Rivanna
- Navigate to `~/Image-Analyzer/weights/`
- Download `.pth` files

---

## Troubleshooting

### Error: "Unable to open file"
**Problem:** Job can't find the SLURM script.

**Solution:**
```bash
# Make sure you're in the right directory
cd ~/Image-Analyzer/rivanna
pwd  # Should show: .../Image-Analyzer/rivanna

# Or use absolute path
sbatch ~/Image-Analyzer/rivanna/slurm_train_lighting_assessor.sh
```

### Error: "Module not found"
**Problem:** Module name or version incorrect.

**Solution:**
```bash
# Check available modules
module avail miniforge
module avail cuda

# Update SLURM script with correct module name
```

### Error: "Conda environment not found"
**Problem:** Environment wasn't created or activated.

**Solution:**
```bash
# Recreate environment
module load miniforge/24.3.0-py3.11
eval "$(conda shell.bash hook)"
conda create -n image-analyzer python=3.11 -y
conda activate image-analyzer
# Reinstall packages
```

### Error: "Dataset not found"
**Problem:** Dataset path incorrect or data not uploaded.

**Solution:**
```bash
# Verify dataset exists
ls -la data/LOLdataset/our485/

# Check paths in SLURM script match your structure
cat rivanna/slurm_train_lighting_assessor.sh | grep data_dir
```

### Error: "Out of memory"
**Problem:** Batch size too large for available GPU memory.

**Solution:**
- Edit SLURM script to reduce `--batch_size`
- Or request more memory: `#SBATCH --mem=64G`

### Job Stuck in PENDING
**Problem:** No GPU resources available.

**Solution:**
```bash
# Check GPU availability
sinfo -p gpu

# Wait for resources or try different partition
# Check queue time
squeue -u $USER
```

---

## Quick Reference Commands

```bash
# Setup (first time)
cd ~/Image-Analyzer/rivanna && ./setup_rivanna.sh

# Submit all jobs
cd ~/Image-Analyzer/rivanna
sbatch slurm_train_lighting_assessor.sh
sbatch slurm_train_blur_detector.sh
sbatch slurm_train_enhancer.sh
sbatch slurm_train_aesthetic_scorer.sh

# Monitor
squeue -u $USER
tail -f logs/*.out

# Cancel all
scancel -u $USER

# Check results
ls -lh weights/
```

---

## Complete Workflow Example

```bash
# 1. SSH into Rivanna
ssh username@rivanna.hpc.virginia.edu

# 2. Navigate to project
cd ~/Image-Analyzer

# 3. First-time setup (if not done)
cd rivanna
./setup_rivanna.sh

# 4. Verify structure
cd ..
ls -la scripts/ models/ data/ rivanna/

# 5. Submit jobs (from rivanna directory)
cd rivanna
sbatch slurm_train_lighting_assessor.sh
sbatch slurm_train_blur_detector.sh

# 6. Monitor
squeue -u $USER
tail -f logs/lighting_*.out

# 7. Once complete, download results (from local machine)
scp username@rivanna.hpc.virginia.edu:~/Image-Analyzer/weights/*.pth ./
```

---

## Additional Resources

- **Rivanna User Guide:** https://www.rc.virginia.edu/userinfo/rivanna/
- **SLURM Documentation:** https://slurm.schedmd.com/documentation.html
- **UVA Research Computing:** https://www.rc.virginia.edu/

---

## Notes

1. **Job Submission:** Always submit from `rivanna/` directory or use absolute paths
2. **Environment:** The conda environment persists between sessions
3. **Storage:** Check your quota with `quota -s`
4. **GPU Access:** Use `sinfo -p gpu` to check GPU availability
5. **Time Limits:** Don't request more time than needed (affects queue priority)

