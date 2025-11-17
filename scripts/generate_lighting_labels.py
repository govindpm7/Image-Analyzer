"""
Utility script to generate lighting/brightness labels for training data
Uses HSV-based method to create initial labels
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def generate_lighting_labels(image_dir, output_csv):
    """
    Generate brightness scores using HSV method
    
    Args:
        image_dir: Directory containing images
        output_csv: Path to output CSV file
    """
    images = []
    scores = []
    
    image_dir = Path(image_dir)
    
    # Find all image files
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    print(f"Found {len(image_files)} images")
    
    for img_path in image_files:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Calculate brightness using HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            avg_brightness = np.mean(hsv[:,:,2])
            brightness_score = (avg_brightness / 255.0) * 10
            
            images.append(str(img_path))
            scores.append(round(brightness_score, 2))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Create DataFrame and save
    df = pd.DataFrame({
        'path': images,
        'brightness_score': scores
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
        description='Generate lighting/brightness labels for training data'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    
    args = parser.parse_args()
    
    generate_lighting_labels(args.image_dir, args.output_csv)


if __name__ == '__main__':
    main()

