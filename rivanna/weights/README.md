# Model Weights

This directory contains trained model weights (`.pth` files) for the deep learning models.

## Weights Files

Model weights are **not tracked in git** due to their large size (typically 50-100+ MB each).

### Expected Weights Files

1. **`aesthetic_scorer_best.pth`**
   - Model: EfficientNet-B0 based aesthetic scorer
   - Purpose: Aesthetic quality assessment (5 scores: overall, composition, color, contrast, focus)

2. **`aesthetic_scorer_final.pth`**
   - Final epoch weights for aesthetic scorer

3. **`lighting_assessor_best.pth`**
   - Model: MobileNet-V2 based lighting assessor
   - Purpose: Brightness assessment and enhancement recommendation

4. **`lighting_assessor_final.pth`**
   - Final epoch weights for lighting assessor

5. **`blur_detector_best.pth`**
   - Model: ResNet-18 based blur detector
   - Purpose: Blur detection (0-10 score)

6. **`enhancer_best.pth`** or **`enhancer_curve_best.pth`**
   - Model: U-Net or Enhancement Curve Network
   - Purpose: Low-light image enhancement
   - Architecture: Use `enhancer_curve_best.pth` for Enhancement Curve Network (more realistic)
   - Architecture: Use `enhancer_best.pth` for U-Net architecture

7. **`enhancer_final.pth`** or **`enhancer_curve_final.pth`**
   - Final epoch weights for enhancer

## Training

To generate these weights, run the training scripts:
- `scripts/train_aesthetic_scorer.py`
- `scripts/train_lighting_assessor.py`
- `scripts/train_blur_detector.py`
- `scripts/train_enhancer_lol.py --architecture curve` (for curve network)
- `scripts/train_enhancer_lol.py --architecture unet` (for U-Net)

## Usage

The application automatically loads weights from this directory if available:
- If weights are found: Uses ML-based inference
- If weights are not found: Falls back to classical CV methods

## File Size

- Typical weight file size: 50-100 MB
- GitHub file size limit: 100 MB (hard limit)
- GitHub recommended limit: 50 MB

These files should be stored locally or using Git LFS if needed for sharing.

