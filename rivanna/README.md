# Rivanna HPC Setup

## Quick Start

1. **Transfer files to Rivanna:**
   ```bash
   scp -r Image-Analyzer username@rivanna.hpc.virginia.edu:~/
   ```

2. **SSH into Rivanna:**
   ```bash
   ssh username@rivanna.hpc.virginia.edu
   ```

3. **Setup environment:**
   ```bash
   cd Image-Analyzer/rivanna
   chmod +x setup_rivanna.sh
   ./setup_rivanna.sh
   ```

4. **Submit training job:**
   ```bash
   cd rivanna
   
   # Low-light enhancement training
   sbatch slurm_train_enhancer.sh
   
   # Lighting assessor training (includes low-light images)
   sbatch slurm_train_lighting_assessor.sh
   
   # Blur detector training
   sbatch slurm_train_blur_detector.sh
   
   # Aesthetic scorer training
   sbatch slurm_train_aesthetic_scorer.sh
   ```

5. **Monitor job:**
   ```bash
   squeue -u $USER
   tail -f logs/enhancer_*.out
   tail -f logs/lighting_*.out
   tail -f logs/blur_*.out
   tail -f logs/aesthetic_*.out
   ```

## Files

- `slurm_train_enhancer.sh` - Low-light enhancement training (24 hours, 100 epochs)
- `slurm_train_lighting_assessor.sh` - Lighting assessor training (12 hours, 50 epochs)
- `slurm_train_blur_detector.sh` - Blur detector training (12 hours, 50 epochs)
- `slurm_train_aesthetic_scorer.sh` - Aesthetic scorer training (12 hours, 50 epochs)
- `setup_rivanna.sh` - Environment setup script
- `RIVANNA_GUIDE.md` - Complete guide with all details

## Important Notes

**Lighting Assessor Training**: The lighting assessor must be trained on a dataset that includes low-light images so it can properly detect when images need enhancement. Use `scripts/generate_lighting_labels.py` to create labels for your dataset, or provide a CSV with `path` and `brightness_score` columns.

**Dataset Requirements**:
- **Blur Detector**: Dataset should have `sharp/` and `blurry/` subdirectories, or provide a CSV with `path` and `label` columns
- **Aesthetic Scorer**: Dataset should have a CSV with columns `path`, `overall`, `composition`, `color`, `contrast`, `focus`, or directory with JSON metadata files

See `RIVANNA_GUIDE.md` for detailed instructions.


