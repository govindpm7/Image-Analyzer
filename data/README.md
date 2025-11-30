# Data Folder

This folder contains datasets and data utilities for the Image Analyzer project.

## Contents

- **`dataset_utils.py`**: PyTorch Dataset classes for different training tasks
- **`LOLdataset/`**: Low-Light (LOL) dataset directory (see below)
- **`LOLdataset.zip`**: Compressed LOL dataset archive (not tracked in git due to size)

---

## LOL (Low-Light) Dataset

### What is the LOL Dataset?

The **LOL (Low-Light) Dataset** is a benchmark dataset designed for low-light image enhancement research. It contains 500 paired images (low-light and normal-light) that are used to train and evaluate image enhancement models.

**Key Characteristics:**
- **Total Image Pairs**: 500
  - **Training Set**: 485 pairs (`our485/`)
  - **Validation/Test Set**: 15 pairs (`eval15/`)
- **Image Resolution**: 400×600 pixels
- **Image Format**: PNG
- **Content**: Mostly indoor scenes
- **Noise**: Natural noise from photo-taking procedure (not synthetic)

### What is it Used For?

The LOL dataset is used in this project to:

1. **Train Low-Light Enhancement Models**
   - Train U-Net and Enhancement Curve Network architectures
   - Learn the mapping from low-light to well-lit images
   - Improve image brightness, contrast, and detail preservation

2. **Model Evaluation**
   - Validate enhancement quality using the 15-pair test set
   - Compare different enhancement architectures
   - Measure performance metrics (PSNR, SSIM, etc.)

3. **Research & Benchmarking**
   - Standard benchmark for low-light enhancement algorithms
   - Compare against state-of-the-art methods
   - Reproduce research results

### Dataset Structure

```
LOLdataset/
├── our485/              # Training set (485 image pairs)
│   ├── high/            # Well-lit reference images (ground truth)
│   │   └── *.png        # 485 images
│   └── low/             # Low-light images (input for training)
│       └── *.png        # 485 images (matching filenames)
└── eval15/              # Validation/test set (15 image pairs)
    ├── high/            # Well-lit reference images (ground truth)
    │   └── *.png        # 15 images
    └── low/             # Low-light images (input for testing)
        └── *.png        # 15 images (matching filenames)
```

**Important**: Each image in `low/` has a corresponding image in `high/` with the same filename. This pairing is essential for supervised learning.

---

## How to Access the LOL Dataset

### Method 1: Using the Local Zip File (Recommended)

1. **Extract the zip file**:
   ```bash
   # Navigate to the data directory
   cd data
   
   # Extract LOLdataset.zip
   # On Windows (PowerShell):
   Expand-Archive -Path LOLdataset.zip -DestinationPath .
   
   # On Linux/Mac:
   unzip LOLdataset.zip
   ```

2. **Verify the structure**:
   After extraction, you should have:
   - `data/LOLdataset/our485/high/` (485 images)
   - `data/LOLdataset/our485/low/` (485 images)
   - `data/LOLdataset/eval15/high/` (15 images)
   - `data/LOLdataset/eval15/low/` (15 images)

3. **Note**: The zip file (`LOLdataset.zip`) is **not tracked in git** due to its large size (331 MB exceeds GitHub's 100 MB limit). You need to obtain it separately or download it from the official source.

### Method 2: Using Activeloop Deep Lake (Alternative)

You can also access the LOL dataset programmatically using [Activeloop Deep Lake](https://datasets.activeloop.ai/docs/ml/datasets/lol-dataset/):

#### Installation

```bash
pip install deeplake
```

#### Load Training Set

```python
import deeplake

# Load training subset
ds_train = deeplake.load('hub://activeloop/lowlight-train')

# Access data
for sample in ds_train:
    highlight_image = sample.highlight_images.numpy()
    lowlight_image = sample.lowlight_images.numpy()
    # Process images...
```

#### Load Validation Set

```python
import deeplake

# Load validation subset
ds_val = deeplake.load('hub://activeloop/lowlight-val')

# Use with PyTorch DataLoader
dataloader = ds_val.pytorch(num_workers=0, batch_size=4, shuffle=False)
```

**Note**: This method streams data from the cloud and doesn't require local storage, but you'll need an internet connection during training.

### Method 3: Download from Official Source

1. **Homepage**: https://daooshee.github.io/BMVC2018website/
2. **Repository**: https://github.com/weichen582/RetinexNet
3. **Paper**: [Deep Retinex Decomposition for Low-Light Enhancement](https://arxiv.org/pdf/2005.02818.pdf)

---

## Using the Dataset in This Project

### Training Scripts

The LOL dataset is used by the following training scripts:

1. **`scripts/train_enhancer_lol.py`**
   - Specifically configured for the LOL dataset structure
   - Supports both U-Net and Enhancement Curve Network architectures
   - Usage:
     ```bash
     python scripts/train_enhancer_lol.py \
         --data_dir data/LOLdataset/our485 \
         --val_dir data/LOLdataset/eval15 \
         --architecture curve  # or 'unet'
     ```

2. **`data/dataset_utils.py`**
   - Contains `LowLightDataset` class that automatically handles the LOL dataset structure
   - Supports both directory-based and CSV-based loading
   - Handles image pairing and transformations

### Dataset Class Usage

```python
from data.dataset_utils import LowLightDataset, get_enhancer_transforms

# Create dataset
train_dataset = LowLightDataset(
    'data/LOLdataset/our485',
    transform=get_enhancer_transforms(train=True, input_size=256)
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

---

## Dataset Information

### Citation

If you use the LOL dataset in your research, please cite:

```bibtex
@article{wei2018deep,
  title={Deep retinex decomposition for low-light enhancement},
  author={Wei, Chen and Wang, Wenjing and Yang, Wenhan and Liu, Jiaying},
  journal={arXiv preprint arXiv:1808.04560},
  year={2018}
}
```

### Dataset Curators

Wei Xiong, Ding Liu, Xiaohui Shen, Chen Fang, and Jiebo Luo

### Licensing

The LOL dataset is publicly available for research purposes. Please refer to the original repository and paper for licensing details. Deep Lake users may have access to a variety of publicly available datasets, but it is your responsibility to determine whether you have permission to use the datasets under their license.

### Additional Resources

- **Original Homepage**: https://daooshee.github.io/BMVC2018website/
- **GitHub Repository**: https://github.com/weichen582/RetinexNet
- **Research Paper**: https://arxiv.org/pdf/2005.02818.pdf
- **Activeloop Deep Lake**: https://datasets.activeloop.ai/docs/ml/datasets/lol-dataset/

---

## Troubleshooting

### Dataset Not Found

If you get an error that the dataset is not found:

1. **Check if the zip file exists**: `data/LOLdataset.zip`
2. **Extract the zip file** if it exists but the directory structure is missing
3. **Verify the directory structure** matches the expected format above
4. **Check file permissions** - ensure read access to all image files

### Empty Dataset Error

If you see `ValueError: num_samples=0`:

1. Verify images exist in both `low/` and `high/` directories
2. Check that filenames match between `low/` and `high/` directories
3. Ensure images are in supported formats (PNG, JPG)
4. Check that the path in your training script is correct

### Large File Size

The `LOLdataset.zip` file is 331 MB, which exceeds GitHub's 100 MB file size limit. Therefore:
- The zip file is **not tracked in git**
- You must obtain it separately (download from official source or use Deep Lake)
- The unzipped directory is also not tracked (only structure files like `.gitkeep` are tracked)

---

## Related Files

- **`data/dataset_utils.py`**: Dataset loading utilities
- **`scripts/train_enhancer_lol.py`**: Training script for LOL dataset
- **`models/enhancer.py`**: Enhancement model definitions
- **`data/LOLdataset/README.md`**: Additional dataset-specific information

