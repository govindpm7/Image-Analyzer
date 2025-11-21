"""
Model loading utilities
"""
import torch
import os


def ensure_weights_directory():
    """Ensure weights directory exists"""
    os.makedirs('weights', exist_ok=True)


def check_model_weights():
    """Check which model weights are available"""
    weights_dir = 'weights'
    available = {
        'blur_detector': os.path.exists(os.path.join(weights_dir, 'blur_detector_best.pth')),
        'enhancer': os.path.exists(os.path.join(weights_dir, 'enhancer_best.pth')),
        'aesthetic_scorer': os.path.exists(os.path.join(weights_dir, 'aesthetic_scorer_best.pth'))
    }
    return available


