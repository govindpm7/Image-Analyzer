"""
Utility script to generate aesthetic labels for training data from LOL dataset
Uses image quality metrics to create initial aesthetic scores
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def _collect_image_files(root_dir):
    """Return list of image paths inside directory (jpg/png)."""
    image_dir = Path(root_dir)
    return list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))


def _compute_image_quality_metrics(img_path):
    """Compute various image quality metrics for aesthetic scoring"""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    metrics = {}
    
    # Brightness (V channel in HSV)
    metrics['brightness'] = float(np.mean(hsv[:, :, 2]))
    
    # Contrast (standard deviation of grayscale)
    metrics['contrast'] = float(np.std(gray))
    
    # Colorfulness (standard deviation in LAB color space)
    metrics['colorfulness'] = float(np.std(lab[:, :, 1]) + np.std(lab[:, :, 2]))
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metrics['sharpness'] = float(laplacian.var())
    
    # Saturation (S channel in HSV)
    metrics['saturation'] = float(np.mean(hsv[:, :, 1]))
    
    return metrics


def _score_from_metrics(metrics, is_high_quality=False):
    """
    Convert image quality metrics to aesthetic scores (0-10 scale)
    
    Args:
        metrics: Dictionary with brightness, contrast, colorfulness, sharpness, saturation
        is_high_quality: Whether this is from the 'high' directory (well-lit images)
    
    Returns:
        dict: Aesthetic scores for overall, composition, color, contrast, focus
    """
    # Base scores - high-quality images start higher
    base_overall = 6.0 if is_high_quality else 4.0
    base_composition = 6.0 if is_high_quality else 4.0
    
    # Normalize metrics to 0-1 range (using reasonable ranges)
    brightness_norm = np.clip(metrics['brightness'] / 255.0, 0, 1)
    contrast_norm = np.clip(metrics['contrast'] / 128.0, 0, 1)
    colorfulness_norm = np.clip(metrics['colorfulness'] / 100.0, 0, 1)
    sharpness_norm = np.clip(metrics['sharpness'] / 1000.0, 0, 1)
    saturation_norm = np.clip(metrics['saturation'] / 255.0, 0, 1)
    
    # Calculate scores
    overall = base_overall + (brightness_norm * 2.0) + (contrast_norm * 1.0) + (sharpness_norm * 1.0)
    overall = np.clip(overall, 0.0, 10.0)
    
    composition = base_composition + (contrast_norm * 1.5) + (sharpness_norm * 1.5)
    composition = np.clip(composition, 0.0, 10.0)
    
    color_score = 5.0 + (colorfulness_norm * 3.0) + (saturation_norm * 2.0)
    color_score = np.clip(color_score, 0.0, 10.0)
    
    contrast_score = 5.0 + (contrast_norm * 4.0)
    contrast_score = np.clip(contrast_score, 0.0, 10.0)
    
    focus_score = 5.0 + (sharpness_norm * 4.0)
    focus_score = np.clip(focus_score, 0.0, 10.0)
    
    return {
        'overall': round(overall, 2),
        'composition': round(composition, 2),
        'color': round(color_score, 2),
        'contrast': round(contrast_score, 2),
        'focus': round(focus_score, 2)
    }


def generate_aesthetic_labels(
    image_dirs,
    output_csv,
    high_quality_dirs=None
):
    """
    Generate aesthetic scores for images
    
    Args:
        image_dirs: List of directories containing images
        output_csv: Path to output CSV file
        high_quality_dirs: List of directories containing high-quality images (for scoring boost)
    """
    if isinstance(image_dirs, (str, Path)):
        image_dirs = [image_dirs]
    if high_quality_dirs is None:
        high_quality_dirs = []
    if isinstance(high_quality_dirs, (str, Path)):
        high_quality_dirs = [high_quality_dirs]
    
    # Convert to Path objects
    high_quality_paths = {Path(d).resolve() for d in high_quality_dirs}
    
    # Collect all images
    path_objects = [Path(d) for d in image_dirs]
    image_files = []
    seen = set()
    
    for directory in path_objects:
        if not directory.exists():
            print(f"Warning: directory {directory} was not found and will be skipped.")
            continue
        for img_path in _collect_image_files(directory):
            img_str = str(img_path.resolve())
            if img_str in seen:
                continue
            seen.add(img_str)
            image_files.append(img_path)
    
    if not image_files:
        raise ValueError("No images found in the provided directories.")
    
    print(f"Found {len(image_files)} images. Computing aesthetic scores...")
    
    # Get project root (assume script is in scripts/ directory)
    project_root = Path(__file__).parent.parent
    
    data_entries = []
    for img_path in image_files:
        try:
            metrics = _compute_image_quality_metrics(img_path)
            
            # Check if this image is in a high-quality directory
            is_high_quality = any(img_path.resolve().is_relative_to(hq_dir) for hq_dir in high_quality_paths)
            
            scores = _score_from_metrics(metrics, is_high_quality=is_high_quality)
            
            # Use relative path from project root for portability
            try:
                rel_path = img_path.relative_to(project_root)
            except ValueError:
                # If not relative, use absolute path
                rel_path = img_path
            
            data_entries.append({
                'path': str(rel_path),
                'overall': scores['overall'],
                'composition': scores['composition'],
                'color': scores['color'],
                'contrast': scores['contrast'],
                'focus': scores['focus']
            })
        except Exception as exc:
            print(f"Warning: Could not process {img_path}: {exc}")
            continue
    
    if not data_entries:
        raise ValueError("No valid images processed for aesthetic scoring.")
    
    # Create DataFrame and save
    df = pd.DataFrame(data_entries)
    df.to_csv(output_csv, index=False)
    
    print(f"Generated labels for {len(df)} images")
    print(f"Saved to {output_csv}")
    print(f"\nAesthetic score statistics:")
    print(f"  Overall - Mean: {df['overall'].mean():.2f}, Std: {df['overall'].std():.2f}")
    print(f"  Composition - Mean: {df['composition'].mean():.2f}, Std: {df['composition'].std():.2f}")
    print(f"  Color - Mean: {df['color'].mean():.2f}, Std: {df['color'].std():.2f}")
    print(f"  Contrast - Mean: {df['contrast'].mean():.2f}, Std: {df['contrast'].std():.2f}")
    print(f"  Focus - Mean: {df['focus'].mean():.2f}, Std: {df['focus'].std():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate aesthetic labels for training data from LOL dataset'
    )
    parser.add_argument(
        '--image_dirs',
        type=str,
        nargs='+',
        required=True,
        help='List of directories containing images (e.g., low and high directories)'
    )
    parser.add_argument(
        '--high_quality_dirs',
        type=str,
        nargs='+',
        default=None,
        help='Directories containing high-quality images (will receive higher scores)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    
    args = parser.parse_args()
    
    generate_aesthetic_labels(
        image_dirs=args.image_dirs,
        output_csv=args.output_csv,
        high_quality_dirs=args.high_quality_dirs
    )


if __name__ == '__main__':
    main()

