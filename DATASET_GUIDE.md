# Dataset Guide: When Do You Need Datasets?

## Quick Answer

**For using the application to analyze YOUR images: NO dataset needed!**

The application works immediately - just upload images and analyze them. It uses fallback methods that work without training.

**For training ML models: YES, you need datasets!**

If you want the models to use actual machine learning instead of fallback methods, you need to train them with datasets.

---

## Two Different Use Cases

### Use Case 1: Analyze Your Own Images (No Dataset Needed)

**What you need:**
- Just the application (`python app.py`)
- Your images to analyze

**What happens:**
- Application uses fallback methods (classical CV)
- Works immediately, no training required
- Good enough for basic analysis

**Example:**
```bash
python app.py
# Upload your images and analyze them
```

### Use Case 2: Train ML Models (Dataset Required)

**What you need:**
- Training datasets (images with labels)
- Training scripts
- Time to train models

**What happens:**
- Models learn from your datasets
- Better accuracy than fallback methods
- Models save weights for future use

**Example:**
```bash
# Prepare dataset
mkdir -p data/blur/{sharp,blurry}
# Add labeled images

# Train model
python train_blur_detector.py --data_dir data/blur --epochs 50
```

---

## What Datasets Do You Need?

### 1. Blur Detection Dataset

**Purpose:** Train model to detect blurry vs sharp images

**What you need:**
- Sharp images (label: 1)
- Blurry images (label: 0)

**Options:**
- **Download existing datasets:**
  - CUHK Blur Dataset
  - RealBlur Dataset
- **Create your own:**
  - Collect sharp images
  - Apply blur filters to create blurry versions
  - Or collect naturally blurry images

**Structure:**
```
data/blur/
    sharp/
        image1.jpg
        image2.jpg
    blurry/
        image1.jpg
        image2.jpg
```

### 2. Aesthetic Scoring Dataset

**Purpose:** Train model to score image aesthetics (composition, color, etc.)

**What you need:**
- Images
- Aesthetic scores (0-10) for: overall, composition, color, contrast, focus

**Options:**
- **Download existing datasets:**
  - AVA Dataset (250,000 images with scores) - **Best option**
  - Photo.net dataset
- **Create your own:**
  - Manually score images
  - Use crowdsourcing
  - Use existing photo rating websites

**Structure (CSV):**
```csv
path,overall,composition,color,contrast,focus
data/aesthetic/image1.jpg,8.5,9.0,8.0,8.5,8.0
data/aesthetic/image2.jpg,6.0,5.5,6.5,6.0,6.5
```

### 3. Low-Light Enhancement Dataset

**Purpose:** Train model to enhance dark images

**What you need:**
- Paired images: low-light version + normal-light version
- Same scene, different lighting

**Options:**
- **Download existing datasets:**
  - LOL Dataset (Low-Light dataset) - **Best option**
  - SICE Dataset
- **Create your own:**
  - Take photos in low light
  - Take same photos with flash/good lighting
  - Or darken normal images artificially

**Structure:**
```
data/lowlight/
    low/
        image1.jpg  (dark version)
        image2.jpg
    normal/
        image1.jpg  (bright version, same filename)
        image2.jpg
```

### 4. Lighting Assessment Dataset

**Purpose:** Train model to assess image brightness

**What you need:**
- Images
- Brightness scores (0-10)

**Options:**
- **Generate automatically** (easiest):
  ```bash
  python scripts/generate_lighting_labels.py \
      --image_dir data/lighting \
      --output_csv data/lighting_scores.csv
  ```
- **Create manually:**
  - Score images based on brightness
  - Use HSV brightness as starting point

**Structure (CSV):**
```csv
path,brightness_score
data/lighting/image1.jpg,8.5
data/lighting/image2.jpg,3.2
```

---

## Dataset Size Recommendations

### Minimum (For Testing)
- **10-20 images per class** - Just to test the pipeline works
- Good for: Verifying code, testing training scripts

### Small (For Quick Training)
- **100-500 images** - Can train basic models
- Good for: Personal use, learning, quick prototypes

### Medium (For Better Results)
- **1,000-5,000 images** - Good quality models
- Good for: Production use, better accuracy

### Large (For Best Results)
- **10,000+ images** - Professional quality
- Good for: Commercial use, research, best accuracy

---

## Where to Get Datasets

### Free/Open Source Datasets

1. **AVA Dataset** (Aesthetic)
   - 250,000 images with aesthetic scores
   - Download: https://github.com/mtobeiyf/ava_downloader
   - Best for aesthetic scoring

2. **LOL Dataset** (Low-Light)
   - 500 paired low/normal images
   - Download: https://daooshee.github.io/BMVC2018website/
   - Best for enhancement

3. **CUHK Blur Dataset** (Blur)
   - Blur detection dataset
   - Download: https://github.com/ngchc/deblurring

4. **RealBlur Dataset** (Blur)
   - Real-world blur images
   - Download: https://github.com/csjcai/RealBlur

### Create Your Own

**For Blur:**
- Take sharp photos
- Apply Gaussian blur in Photoshop/GIMP
- Or collect naturally blurry photos

**For Aesthetic:**
- Use photo rating websites
- Manually score your photo collection
- Use crowdsourcing (Amazon MTurk, etc.)

**For Low-Light:**
- Take photos with/without flash
- Use camera settings to create dark/bright pairs
- Or artificially darken images

**For Lighting:**
- Use the provided script to auto-generate labels
- Or manually assess brightness

---

## Quick Start: Minimal Dataset

If you just want to test training works:

```bash
# 1. Create tiny test dataset (5-10 images)
mkdir -p test_data/blur/{sharp,blurry}
# Add a few images to each folder

# 2. Train for 2 epochs (quick test)
python train_blur_detector.py \
    --data_dir test_data/blur \
    --epochs 2 \
    --batch_size 2

# 3. Check if weights created
ls weights/
# Should see: blur_detector_best.pth
```

---

## Do You Actually Need to Train?

### You DON'T need to train if:
- ✅ You just want to analyze your own images
- ✅ Fallback methods are good enough for you
- ✅ You don't have time/resources for training
- ✅ You want to use the app immediately

### You SHOULD train if:
- ✅ You want better accuracy than fallbacks
- ✅ You have specific use cases (e.g., specific types of images)
- ✅ You want ML-based analysis
- ✅ You have datasets available
- ✅ You want to customize models for your domain

---

## Recommendation

**Start without training:**
1. Run `python app.py`
2. Test with your images
3. See if fallback methods work for your needs

**Then consider training if:**
- You need better accuracy
- You have specific image types
- You have datasets available
- You want ML-based features

---

## Summary

| Question | Answer |
|----------|--------|
| Need dataset to use the app? | **NO** - App works immediately |
| Need dataset to train models? | **YES** - For ML-based analysis |
| Can I use the app without training? | **YES** - Uses fallback methods |
| Should I train models? | **Optional** - Only if you need better accuracy |
| Where to get datasets? | See "Where to Get Datasets" section above |
| Minimum dataset size? | 10-20 images for testing, 100+ for real use |

**Bottom line:** The application works without any datasets. Datasets are only needed if you want to train ML models for better accuracy.

