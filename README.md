# DS4002 - Image Analyzer

Minimal Flask/Dash-based image analysis backend scaffold using PyTorch and OpenCV for computer vision experiments.

---

## Features

- Flask web backend for serving image analysis endpoints. :contentReference[oaicite:0]{index=0}  
- Designed to integrate Dash layouts/components for interactive dashboards. :contentReference[oaicite:1]{index=1}  
- PyTorch + Torchvision model stack for deep-learning-based image analysis.
- OpenCV + scikit-image pipeline for classical image processing and feature extraction.
- NumPy/SciPy-based numeric and image metric utilities.
- Structured `models/` and `utils/` packages to keep model code and helper logic modular. :contentReference[oaicite:2]{index=2}  
- Configurable via environment variables loaded from a `.env` file (via `python-dotenv`).
- Ready for production deployment via Gunicorn.

> Note: This repository currently exposes a lean codebase (single `app.py` entrypoint plus `models/` and `utils/` packages). The sections below describe the intended usage and structure of this scaffold.

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

## Project Structure

Approximate high-level structure:

```text
Image-Analyzer/
├── app.py            # Flask/Dash application entrypoint
├── models/           # PyTorch models and/or weight loading utilities
├── utils/            # Image preprocessing, postprocessing, metrics, helpers
├── requirements.txt  # Python dependencies
├── .gitignore        # Git ignore rules
└── LICENSE           # MIT license
