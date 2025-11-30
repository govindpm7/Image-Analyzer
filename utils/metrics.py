"""
Evaluation metrics for models
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    return peak_signal_noise_ratio(img1, img2)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return structural_similarity(img1, img2, multichannel=True, channel_axis=2)


