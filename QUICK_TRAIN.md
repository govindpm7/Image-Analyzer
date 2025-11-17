# Quick Training Commands (Minimal Time)

## Fastest Training Command (For Testing)

### Low-Light Enhancer with LOL Dataset

```bash
python train_enhancer_lol.py \
    --batch_size 4 \
    --epochs 5 \
    --input_size 128
```

**Expected time:** ~5-10 minutes (depending on GPU/CPU)

### All Models - Quick Test

```bash
# Blur Detector (if you have data)
python train_blur_detector.py \
    --data_dir data/blur \
    --batch_size 4 \
    --epochs 5

# Aesthetic Scorer (if you have data)
python train_aesthetic_scorer.py \
    --data_dir data/aesthetic \
    --batch_size 4 \
    --epochs 5

# Low-Light Enhancer (LOL dataset)
python train_enhancer_lol.py \
    --batch_size 4 \
    --epochs 5 \
    --input_size 128

# Lighting Assessor (if you have data)
python train_lighting_assessor.py \
    --data_dir data/lighting \
    --batch_size 4 \
    --epochs 5
```

## Recommended Minimal Training (Better Results, Still Fast)

### Low-Light Enhancer - Recommended Minimal

```bash
python train_enhancer_lol.py \
    --batch_size 8 \
    --epochs 10 \
    --input_size 256 \
    --lr 0.0001
```

**Expected time:** ~15-30 minutes

## Even Faster (CPU-Friendly)

If you're on CPU or want absolute minimum:

```bash
python train_enhancer_lol.py \
    --batch_size 2 \
    --epochs 3 \
    --input_size 128
```

**Expected time:** ~2-5 minutes (but results will be poor)

## Full Training (For Production)

```bash
python train_enhancer_lol.py \
    --batch_size 16 \
    --epochs 100 \
    --input_size 256 \
    --lr 0.0001
```

**Expected time:** Several hours (but best results)

---

## Quick Reference

| Setting | Fastest | Minimal | Recommended | Full |
|---------|---------|---------|-------------|------|
| Batch Size | 2-4 | 4-8 | 8-16 | 16-32 |
| Epochs | 3-5 | 5-10 | 10-20 | 50-100 |
| Input Size | 128 | 128-256 | 256 | 256-512 |
| Time | 2-5 min | 10-30 min | 30-60 min | Hours |

---

## One-Liner Commands

**Absolute minimum (test if it works):**
```bash
python train_enhancer_lol.py --batch_size 4 --epochs 5 --input_size 128
```

**Quick but useful:**
```bash
python train_enhancer_lol.py --batch_size 8 --epochs 10
```

**Balanced:**
```bash
python train_enhancer_lol.py --batch_size 8 --epochs 20 --input_size 256
```

