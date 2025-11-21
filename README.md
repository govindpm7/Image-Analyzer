# DS4002 - Image Analyzer

A comprehensive image analysis application with a Dash-based web interface for analyzing image quality, aesthetics, and enhancing low-light images using deep learning models.

---

## Application Capabilities

The Image Analyzer provides four main analysis capabilities:

### 1. **Blur Detection**
- **Model**: ResNet-18 based binary classifier
- **Output**: Blur score from 0-10 (higher = sharper)
- **Features**: 
  - ML-based detection when trained weights are available
  - Fallback to classical Laplacian variance method
  - Visual status indicators (Sharp/Moderate/Blurry)

### 2. **Aesthetic Quality Assessment**
- **Model**: EfficientNet-B0 based multi-task scorer
- **Output**: Five scores (0-10 scale):
  - **Overall**: Overall aesthetic quality
  - **Composition**: Rule of thirds and focal point analysis
  - **Color**: Color harmony and saturation analysis
  - **Contrast**: Overall contrast ratio
  - **Focus**: Edge density and sharpness
- **Features**: 
  - ML-based scoring when trained weights are available
  - Fallback to feature-based scoring using OpenCV

### 3. **Lighting/Brightness Assessment**
- **Model**: MobileNet-V2 based regression model
- **Output**: 
  - Brightness score (0-10 scale)
  - Enhancement recommendation (boolean)
- **Features**:
  - ML-based assessment when trained weights are available
  - Fallback to HSV brightness calculation
  - Automatic recommendation for low-light images

### 4. **Low-Light Image Enhancement**
- **Model**: U-Net architecture for image-to-image translation
- **Output**: Enhanced image with improved brightness and contrast
- **Features**:
  - ML-based enhancement when trained weights are available
  - Fallback to classical methods (CLAHE + gamma correction)
  - Adjustable brightness limit (180-255) to prevent over-enhancement
  - Color preservation mode to maintain natural colors

### Web Interface
- **Framework**: Dash with Bootstrap styling
- **Features**:
  - Drag-and-drop image upload
  - Real-time image preview
  - Interactive analysis results with progress bars
  - Side-by-side comparison of original and enhanced images
  - Model status indicators (ML vs fallback mode)

---

## Project Structure

```
Image-Analyzer/
├── data/                          # Dataset and data utilities
│   ├── __init__.py
│   ├── dataset_utils.py          # Dataset classes and data loaders
│   └── LOLdataset/               # Low-Light (LOL) dataset
│       ├── our485/               # Training set (485 image pairs)
│       │   ├── high/             # Well-lit reference images (485 images)
│       │   └── low/              # Low-light images (485 images)
│       └── eval15/               # Evaluation set (15 image pairs)
│           ├── high/             # Well-lit reference images (15 images)
│           └── low/              # Low-light images (15 images)
│
├── models/                        # Deep learning model definitions
│   ├── __init__.py
│   ├── aesthetic_scorer.py      # EfficientNet-B0 based aesthetic scorer
│   ├── blur_detector.py          # ResNet-18 based blur detector
│   ├── enhancer.py               # U-Net based low-light enhancer
│   ├── lighting_assessor.py      # MobileNet-V2 based lighting assessor
│   └── model_utils.py            # Model utility functions
│
├── outputs/                       # Training outputs and model weights
│   └── weights/                  # Trained model weights (.pth files)
│       ├── aesthetic_scorer_best.pth      # Best aesthetic scorer weights
│       ├── aesthetic_scorer_final.pth     # Final aesthetic scorer weights
│       ├── lighting_assessor_best.pth     # Best lighting assessor weights
│       └── lighting_assessor_final.pth    # Final lighting assessor weights
│
├── scripts/                       # Application scripts and training code
│   ├── __init__.py
│   ├── app.py                    # Main Dash web application entrypoint
│   ├── generate_aesthetic_labels.py    # Script to generate aesthetic labels
│   ├── generate_lighting_labels.py    # Script to generate lighting labels
│   ├── train_aesthetic_scorer.py       # Training script for aesthetic model
│   ├── train_blur_detector.py         # Training script for blur detector
│   ├── train_enhancer.py              # Training script for enhancer (generic)
│   ├── train_enhancer_lol.py          # Training script for enhancer (LOL dataset)
│   ├── train_lighting_assessor.py     # Training script for lighting assessor
│   ├── rivanna/                        # HPC cluster training scripts
│   │   ├── PIPELINE.md                 # Complete training pipeline guide
│   │   ├── README.md                   # Rivanna-specific documentation
│   │   ├── RIVANNA_GUIDE.md            # Detailed Rivanna setup guide
│   │   ├── setup_rivanna.sh            # Rivanna environment setup
│   │   ├── setup_github.sh             # GitHub setup script
│   │   ├── slurm_train_aesthetic_scorer.sh    # SLURM job for aesthetic training
│   │   ├── slurm_train_blur_detector.sh       # SLURM job for blur training
│   │   ├── slurm_train_enhancer.sh            # SLURM job for enhancer training
│   │   └── slurm_train_lighting_assessor.sh   # SLURM job for lighting training
│   └── utils/                          # Utility functions
│       ├── __init__.py
│       ├── image_processing.py         # OpenCV-based image processing utilities
│       ├── metrics.py                  # Evaluation metrics (PSNR, SSIM, etc.)
│       └── model_loader.py             # Model loading utilities
│
├── LICENSE                         # MIT License
└── README.md                        # This file
```

---

## Data Folder Structure and Metadata

### Data Directory: `data/`

The `data/` folder contains dataset utilities and the Low-Light (LOL) dataset used for training the image enhancement model.

#### Dataset Utilities (`data/dataset_utils.py`)

Provides PyTorch Dataset classes for different training tasks:

1. **BlurDataset**: For blur detection training
   - Supports directory structure: `sharp/` and `blurry/` subdirectories
   - Or CSV file with image paths and labels

2. **AestheticDataset**: For aesthetic scoring training
   - Supports CSV with columns: `path`, `overall`, `composition`, `color`, `contrast`, `focus`
   - Or JSON metadata files alongside images

3. **LowLightDataset**: For low-light enhancement training
   - Supports paired images: `low/` and `high/` (or `normal/`) directories
   - Or CSV with `low_path` and `normal_path` columns

4. **LightingDataset**: For lighting assessment training
   - Supports CSV with `path` and `brightness_score` columns
   - Or JSON metadata files with brightness scores

#### LOL Dataset (`data/LOLdataset/`)

The Low-Light (LOL) dataset is used for training the image enhancement model. It contains paired low-light and well-lit images.

**Training Set (`our485/`):**
- **Purpose**: Training the U-Net enhancement model
- **Structure**: 
  - `high/`: 485 well-lit reference images (ground truth)
  - `low/`: 485 corresponding low-light images (input)
- **Format**: PNG images
- **Usage**: Used during model training to learn the mapping from low-light to well-lit images

**Evaluation Set (`eval15/`):**
- **Purpose**: Validation and testing of the trained enhancement model
- **Structure**:
  - `high/`: 15 well-lit reference images (ground truth)
  - `low/`: 15 corresponding low-light images (input)
- **Format**: PNG images
- **Usage**: Used for model validation and performance evaluation

**Dataset Characteristics:**
- **Total Training Pairs**: 485
- **Total Evaluation Pairs**: 15
- **Image Format**: PNG
- **Pairing**: Each image in `low/` has a corresponding image in `high/` with the same filename
- **Use Case**: Supervised learning for low-light image enhancement

---

## Outputs Directory

### Outputs Directory: `outputs/`

The `outputs/` folder contains trained model weights and training artifacts.

#### Model Weights (`outputs/weights/`)

Contains PyTorch model checkpoint files (`.pth` format) saved during training:

1. **`aesthetic_scorer_best.pth`**
   - **Purpose**: Best performing weights for the aesthetic scoring model
   - **Model**: EfficientNet-B0 with custom classifier head
   - **Usage**: Loaded by `AestheticScorer` class for inference
   - **Output**: 5 scores (overall, composition, color, contrast, focus)

2. **`aesthetic_scorer_final.pth`**
   - **Purpose**: Final epoch weights for the aesthetic scoring model
   - **Model**: EfficientNet-B0 with custom classifier head
   - **Usage**: Alternative checkpoint, typically `_best.pth` is preferred

3. **`lighting_assessor_best.pth`**
   - **Purpose**: Best performing weights for the lighting assessment model
   - **Model**: MobileNet-V2 with regression head
   - **Usage**: Loaded by `LightingAssessor` class for inference
   - **Output**: Brightness score (0-10) and enhancement recommendation

4. **`lighting_assessor_final.pth`**
   - **Purpose**: Final epoch weights for the lighting assessment model
   - **Model**: MobileNet-V2 with regression head
   - **Usage**: Alternative checkpoint, typically `_best.pth` is preferred

**Note**: Additional model weights (e.g., `blur_detector_best.pth`, `enhancer_best.pth`) would be saved here when those models are trained.

**Weight Loading:**
- Models automatically check for weights in `outputs/weights/` (or `weights/` relative to the script)
- If weights are found, models use ML-based inference
- If weights are not found, models fall back to classical computer vision methods
- The web interface displays the current mode (ML vs fallback) to users

---

## Tech Stack

- **Backend**
  - Flask 3.0
  - Werkzeug 3.x
  - Dash 2.x
  - dash-bootstrap-components

- **ML / CV**
  - PyTorch (>= 2.6.0)
  - Torchvision (>= 0.21.0)
  - OpenCV (`opencv-python` >= 4.8.0)
  - scikit-learn (>= 1.3.2)
  - scikit-image (>= 0.22.0)
  - Pillow (>= 10.1.0)
  - NumPy (>= 1.26.0)

- **Image Metrics / Numerics**
  - SciPy (>= 1.11.4)

- **Configuration**
  - python-dotenv

- **Deployment**
  - Gunicorn (optional)

---

## Quick Start

### Running the Web Application

```bash
# Navigate to the scripts directory
cd scripts

# Run the Dash application
python app.py
```

The application will start on `http://localhost:8050`

### Training Models

See the `scripts/rivanna/` directory for training guides and SLURM scripts for HPC cluster training, or run training scripts directly:

```bash
# Example: Train the aesthetic scorer
python scripts/train_aesthetic_scorer.py --data_dir data/training/aesthetic --epochs 50
```

---

## Features

- **Hybrid ML/CV Approach**: Uses deep learning models when trained weights are available, falls back to classical computer vision methods otherwise
- **Modular Architecture**: Separate model definitions, training scripts, and utilities
- **Web Interface**: Interactive Dash-based UI for easy image analysis
- **Training Support**: Complete training pipeline with dataset utilities and HPC cluster support
- **Flexible Dataset Loading**: Supports multiple dataset formats (CSV, directory structure, JSON metadata)

---

## License

MIT License - see LICENSE file for details
