"""
Utility script to generate lighting/brightness labels for training data
Uses HSV-based method to create initial labels
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


def _compute_brightness(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def _collect_brightness_from_dir(directory):
    if not directory:
        return []
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    values = []
    for img_path in _collect_image_files(dir_path):
        try:
            values.append(_compute_brightness(img_path))
        except ValueError:
            continue
    return values


def generate_lighting_labels(
    image_dirs,
    output_csv,
    reference_low_dir=None,
    reference_high_dir=None,
    low_target_score=3.0,
    high_target_score=7.0
):
    """
    Generate brightness scores using HSV method
    
    Args:
        image_dirs: One or more directories containing images
        output_csv: Path to output CSV file
        reference_low_dir: Optional directory containing low-light reference images
        reference_high_dir: Optional directory containing well-lit reference images
        low_target_score: Desired score for reference low images
        high_target_score: Desired score for reference high images
    """
    if isinstance(image_dirs, (str, Path)):
        image_dirs = [image_dirs]

    path_objects = [Path(d) for d in image_dirs]
    image_files = []
    seen = set()
    for directory in path_objects:
        if not directory.exists():
            print(f"Warning: directory {directory} was not found and will be skipped.")
            continue
        for img_path in _collect_image_files(directory):
            img_str = str(img_path)
            if img_str in seen:
                continue
            seen.add(img_str)
            image_files.append(img_path)
    
    if not image_files:
        raise ValueError("No images found in the provided directories.")
    
    print(f"Found {len(image_files)} images. Calculating brightness...")
    
    data_entries = []
    brightness_values = []
    for img_path in image_files:
        try:
            brightness = _compute_brightness(img_path)
        except ValueError as exc:
            print(exc)
            continue
        
        brightness_values.append(brightness)
        data_entries.append((str(img_path), brightness))
    
    if not data_entries:
        raise ValueError("No valid images processed for brightness scoring.")
    
    min_b = min(brightness_values)
    max_b = max(brightness_values)
    denom = max_b - min_b if max_b > min_b else 1.0
    
    def brightness_to_raw_score(b):
        if max_b == min_b:
            return 5.0
        return ((b - min_b) / denom) * 10.0
    
    raw_scores = [brightness_to_raw_score(b) for _, b in data_entries]
    
    # Reference calibration using LOL low/high directories
    calibrated_scores = raw_scores
    ref_low_values = _collect_brightness_from_dir(reference_low_dir)
    ref_high_values = _collect_brightness_from_dir(reference_high_dir)
    
    if ref_low_values and ref_high_values:
        ref_low_mean = brightness_to_raw_score(np.mean(ref_low_values))
        ref_high_mean = brightness_to_raw_score(np.mean(ref_high_values))
        
        if abs(ref_high_mean - ref_low_mean) > 1e-6:
            scale = (high_target_score - low_target_score) / (ref_high_mean - ref_low_mean)
            shift = low_target_score - scale * ref_low_mean
            calibrated_scores = [
                float(np.clip(scale * score + shift, 0.0, 10.0))
                for score in raw_scores
            ]
            print(
                f"Applied LOL reference calibration: "
                f"low_mean→{low_target_score}, high_mean→{high_target_score}"
            )
        else:
            print("Reference calibration skipped (low/high brightness too similar).")
    elif reference_low_dir or reference_high_dir:
        print("Provided reference directories did not contain valid images. Skipping calibration.")
    
    # Create DataFrame and save
    df = pd.DataFrame({
        'path': [entry[0] for entry in data_entries],
        'brightness_score': [round(score, 2) for score in calibrated_scores]
    })
    
    df.to_csv(output_csv, index=False)
    print(f"Generated labels for {len(df)} images")
    print(f"Saved to {output_csv}")
    print(f"\nBrightness score statistics:")
    print(f"  Mean: {df['brightness_score'].mean():.2f}")
    print(f"  Std: {df['brightness_score'].std():.2f}")
    print(f"  Min: {df['brightness_score'].min():.2f}")
    print(f"  Max: {df['brightness_score'].max():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate lighting/brightness labels for training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Either --image_dir OR --image_dirs must be provided (not both required).'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Single directory containing images (alternative to --image_dirs)'
    )
    parser.add_argument(
        '--image_dirs',
        type=str,
        nargs='+',
        default=None,
        help='List of directories to combine before labeling (alternative to --image_dir)'
    )
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--reference_lol_dir', type=str, default=None,
                        help='Path to LOL dataset root. Uses low/high directories as calibration anchors.')
    parser.add_argument('--reference_low_dir', type=str, default=None,
                        help='Explicit path to directory containing low-light reference images.')
    parser.add_argument('--reference_high_dir', type=str, default=None,
                        help='Explicit path to directory containing well-lit reference images.')
    parser.add_argument('--low_target_score', type=float, default=3.0,
                        help='Desired score for reference low images (default: 3.0).')
    parser.add_argument('--high_target_score', type=float, default=7.0,
                        help='Desired score for reference high images (default: 7.0).')
    
    args = parser.parse_args()
    
    ref_low = args.reference_low_dir
    ref_high = args.reference_high_dir
    
    if args.reference_lol_dir:
        lol_root = Path(args.reference_lol_dir)
        candidate_low = lol_root / 'low'
        candidate_high = lol_root / 'high'
        if not candidate_high.exists():
            candidate_high = lol_root / 'normal'
        ref_low = ref_low or (str(candidate_low) if candidate_low.exists() else None)
        ref_high = ref_high or (str(candidate_high) if candidate_high.exists() else None)
    
    # Collect image sources - either --image_dir or --image_dirs (or both)
    image_sources = []
    if args.image_dir:
        image_sources.append(args.image_dir)
    if args.image_dirs:
        image_sources.extend(args.image_dirs)
    
    # Validate that at least one source was provided
    if not image_sources:
        parser.error(
            'ERROR: You must provide either --image_dir OR --image_dirs (or both).\n'
            '  Example: --image_dirs dir1 dir2 dir3\n'
            '  Example: --image_dir single_directory'
        )

    generate_lighting_labels(
        image_dirs=image_sources,
        output_csv=args.output_csv,
        reference_low_dir=ref_low,
        reference_high_dir=ref_high,
        low_target_score=args.low_target_score,
        high_target_score=args.high_target_score
    )


if __name__ == '__main__':
    main()

