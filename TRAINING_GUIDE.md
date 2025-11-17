# Training Guide for Image Analyzer Models

This guide explains how to train all ML models for the Image Analyzer application.

## Overview

The Image Analyzer uses 4 ML models:
1. **BlurDetector** - ResNet-18 for blur detection
2. **AestheticScorer** - EfficientNet-B0 for aesthetic quality assessment
3. **LowLightEnhancer** - U-Net for low-light image enhancement
4. **LightingAssessor** - MobileNet-V2 for brightness/lighting assessment

## Prerequisites

### Install Dependencies

```bash
pip install -r requirements.txt
pip install pandas tqdm  # Additional dependencies for training
```

### Prepare Datasets

Each model requires specific data formats. See below for details.

## 1. Training Blur Detector

### Dataset Structure

Organize your data in one of these formats:

**Option A: Directory Structure**
```
data/blur/
    sharp/
        image1.jpg
        image2.jpg
        ...
    blurry/
        image1.jpg
        image2.jpg
        ...
```

**Option B: CSV File**
```csv
path,label
data/blur/sharp/image1.jpg,1
data/blur/blurry/image1.jpg,0
```

### Training Command

```bash
python train_blur_detector.py \
    --data_dir data/blur \
    --val_dir data/blur_val \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir weights
```

### Parameters
- `--data_dir`: Path to training data
- `--val_dir`: Path to validation data (optional)
- `--csv_path`: Path to CSV file (if using CSV format)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--save_dir`: Directory to save weights (default: weights)
- `--resume`: Path to checkpoint to resume training

### Dataset Sources
- **CUHK Blur Dataset**: https://github.com/ngchc/deblurring
- **RealBlur Dataset**: https://github.com/csjcai/RealBlur
- Create your own by applying blur filters to sharp images

---

## 2. Training Aesthetic Scorer

### Dataset Structure

**Option A: CSV File (Recommended)**
```csv
path,overall,composition,color,contrast,focus
data/aesthetic/image1.jpg,8.5,9.0,8.0,8.5,8.0
data/aesthetic/image2.jpg,6.0,5.5,6.5,6.0,6.5
```

**Option B: Directory with JSON Metadata**
```
data/aesthetic/
    image1.jpg
    image1.json  # {"overall": 8.5, "composition": 9.0, "color": 8.0, "contrast": 8.5, "focus": 8.0}
    image2.jpg
    image2.json
```

### Training Command

```bash
python train_aesthetic_scorer.py \
    --data_dir data/aesthetic \
    --val_dir data/aesthetic_val \
    --csv_path data/aesthetic_scores.csv \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir weights
```

### Dataset Sources
- **AVA Dataset**: https://github.com/mtobeiyf/ava_downloader (250,000 images with aesthetic scores)
- **Photo.net Dataset**: Historical aesthetic dataset
- Create your own by manually scoring images

---

## 3. Training Low-Light Enhancer

### Dataset Structure

**Option A: Directory Structure**
```
data/lowlight/
    low/
        image1.jpg
        image2.jpg
        ...
    normal/
        image1.jpg  # Same filename as low version
        image2.jpg
        ...
```

**Option B: CSV File**
```csv
low_path,normal_path
data/lowlight/low/image1.jpg,data/lowlight/normal/image1.jpg
data/lowlight/low/image2.jpg,data/lowlight/normal/image2.jpg
```

### Training Command

```bash
python train_enhancer.py \
    --data_dir data/lowlight \
    --val_dir data/lowlight_val \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.0001 \
    --input_size 256 \
    --save_dir weights
```

### Parameters
- `--input_size`: Input image size (default: 256)
- Other parameters same as above

### Dataset Sources
- **LOL Dataset**: https://daooshee.github.io/BMVC2018website/ (Low-Light dataset)
- **SICE Dataset**: https://github.com/csjcai/SICE (Single Image Contrast Enhancement)
- Create pairs by darkening normal images or collecting real low-light/normal pairs

---

## 4. Training Lighting Assessor

### Dataset Structure

**Option A: CSV File**
```csv
path,brightness_score
data/lighting/image1.jpg,8.5
data/lighting/image2.jpg,3.2
```

**Option B: Directory with JSON Metadata**
```
data/lighting/
    image1.jpg
    image1.json  # {"brightness_score": 8.5}
    image2.jpg
    image2.json
```

### Training Command

```bash
python train_lighting_assessor.py \
    --data_dir data/lighting \
    --val_dir data/lighting_val \
    --csv_path data/lighting_scores.csv \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir weights
```

### Creating Training Data

You can automatically generate labels using the fallback method:

```python
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def generate_lighting_labels(image_dir, output_csv):
    """Generate brightness scores using HSV method"""
    images = []
    scores = []
    
    for img_path in Path(image_dir).glob('*.jpg'):
        img = cv2.imread(str(img_path))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv[:,:,2])
        brightness_score = (avg_brightness / 255.0) * 10
        
        images.append(str(img_path))
        scores.append(brightness_score)
    
    df = pd.DataFrame({'path': images, 'brightness_score': scores})
    df.to_csv(output_csv, index=False)

# Usage
generate_lighting_labels('data/lighting', 'data/lighting_scores.csv')
```

---

## Training Tips

### 1. Data Augmentation
All training scripts include data augmentation:
- Random horizontal flips
- Random rotations
- Color jitter (brightness, contrast, saturation)

### 2. Learning Rate Scheduling
- Learning rate is automatically reduced when validation metrics plateau
- Blur/Aesthetic/Lighting: Reduce on validation accuracy/MAE
- Enhancer: Reduce on validation PSNR

### 3. Monitoring Training
- Watch for overfitting (train loss decreases but val loss increases)
- Save best models based on validation metrics
- Use TensorBoard for visualization (can be added)

### 4. Resuming Training
```bash
python train_blur_detector.py \
    --data_dir data/blur \
    --resume weights/blur_detector_epoch_25.pth \
    --epochs 50
```

### 5. GPU vs CPU
- Training is much faster on GPU
- Models automatically use CUDA if available
- For CPU training, reduce batch size

---

## Model Weights

After training, model weights are saved to the `weights/` directory:
- `blur_detector_best.pth` - Best blur detection model
- `aesthetic_scorer_best.pth` - Best aesthetic scoring model
- `enhancer_best.pth` - Best low-light enhancement model
- `lighting_assessor_best.pth` - Best lighting assessment model

The application automatically loads these weights when available.

---

## Quick Start: Minimal Training

If you want to quickly test the training pipeline with minimal data:

1. **Create small test datasets** (10-20 images each)
2. **Train for a few epochs** to verify the pipeline works
3. **Scale up** with full datasets for production models

Example minimal training:
```bash
# Blur detector with 20 images
python train_blur_detector.py --data_dir data/blur_test --epochs 5 --batch_size 4

# Aesthetic scorer with 20 images
python train_aesthetic_scorer.py --data_dir data/aesthetic_test --epochs 5 --batch_size 4
```

---

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Reduce `--input_size` for enhancer
- Use gradient accumulation (not implemented, but can be added)

### Poor Performance
- Check data quality and labels
- Increase training epochs
- Adjust learning rate
- Add more data augmentation
- Use transfer learning (models start with pretrained weights)

### Model Not Loading
- Check that weights file exists in `weights/` directory
- Verify model architecture matches training script
- Check for CUDA/CPU device mismatches

---

## Next Steps

1. **Collect or download datasets** for your use case
2. **Train models** using the scripts above
3. **Evaluate** on test sets
4. **Deploy** trained weights to the application

For questions or issues, check the model files in `models/` directory for implementation details.

