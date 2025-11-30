"""
Dataset loading utilities for training
"""
import os
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class BlurDataset(Dataset):
    """
    Dataset for blur detection training
    Expects directory structure:
        blur/
            sharp/
            blurry/
    Or a CSV file with image paths and labels
    """
    def __init__(self, data_dir, transform=None, is_csv=False, csv_path=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Only use CSV if it exists, otherwise fall back to directory loading
        if is_csv and csv_path and Path(csv_path).exists():
            self._load_from_csv(csv_path)
        else:
            if is_csv and csv_path:
                print(f"⚠ Warning: CSV file '{csv_path}' not found. Loading from directory instead.")
            self._load_from_directory(data_dir)
    
    def _load_from_directory(self, data_dir):
        """Load images from directory structure"""
        data_path = Path(data_dir)
        
        # Check if data directory exists
        if not data_path.exists():
            print(f"❌ ERROR: Data directory does not exist: {data_path}")
            print(f"   Please check the path and try again.")
            return
        
        if not data_path.is_dir():
            print(f"❌ ERROR: Path is not a directory: {data_path}")
            return
        
        # Check for blur/sharp structure
        sharp_dir = data_path / 'sharp'
        blurry_dir = data_path / 'blurry'
        
        if sharp_dir.exists() and blurry_dir.exists():
            # Load sharp images (label 1)
            sharp_images = list(sharp_dir.glob('*.jpg')) + list(sharp_dir.glob('*.png'))
            for img_path in sharp_images:
                self.images.append(str(img_path))
                self.labels.append(1)
            
            # Load blurry images (label 0)
            blurry_images = list(blurry_dir.glob('*.jpg')) + list(blurry_dir.glob('*.png'))
            for img_path in blurry_images:
                self.images.append(str(img_path))
                self.labels.append(0)
            
            print(f"✓ Loaded {len(sharp_images)} sharp images and {len(blurry_images)} blurry images from {data_path}")
        else:
            # Alternative: single directory with labels in filename
            # Format: image_0.jpg (blurry) or image_1.jpg (sharp)
            all_images = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))
            for img_path in all_images:
                label = 1 if 'sharp' in img_path.name.lower() or '1' in img_path.stem else 0
                self.images.append(str(img_path))
                self.labels.append(label)
            
            if all_images:
                print(f"✓ Loaded {len(all_images)} images from {data_path} (using filename-based labels)")
            else:
                print(f"⚠ WARNING: No images found in {data_path}")
                print(f"   Expected directory structure:")
                print(f"     {data_path}/")
                print(f"       sharp/")
                print(f"         *.jpg or *.png")
                print(f"       blurry/")
                print(f"         *.jpg or *.png")
                print(f"   Or provide images directly in {data_path} with labels in filenames")
    
    def _load_from_csv(self, csv_path):
        """Load images from CSV file (path, label)"""
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.images = df['path'].tolist()
        self.labels = df['label'].tolist()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label, dtype=torch.long)


class AestheticDataset(Dataset):
    """
    Dataset for aesthetic scoring
    Expects CSV file with columns: path, overall, composition, color, contrast, focus
    Or directory with JSON metadata files
    """
    def __init__(self, data_dir, transform=None, csv_path=None):
        self.transform = transform
        self.images = []
        self.scores = []
        
        # Only use CSV if it exists, otherwise fall back to directory loading
        if csv_path and Path(csv_path).exists():
            self._load_from_csv(csv_path)
        else:
            if csv_path:
                print(f"⚠ Warning: CSV file '{csv_path}' not found. Loading from directory instead.")
            self._load_from_directory(data_dir)
    
    def _load_from_csv(self, csv_path):
        """Load from CSV with scores"""
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Handle both absolute and relative paths
        project_root = Path.cwd()
        self.images = []
        for path_str in df['path'].tolist():
            path = Path(path_str)
            if path.is_absolute():
                self.images.append(str(path))
            else:
                # Try relative to current directory, then project root
                full_path = Path(path_str)
                if full_path.exists():
                    self.images.append(str(full_path.resolve()))
                else:
                    # Try relative to project root
                    project_path = project_root / path_str
                    if project_path.exists():
                        self.images.append(str(project_path.resolve()))
                    else:
                        # Use as-is and hope it works
                        self.images.append(path_str)
        
        self.scores = df[['overall', 'composition', 'color', 'contrast', 'focus']].values.tolist()
    
    def _load_from_directory(self, data_dir):
        """Load from directory with JSON metadata or use default scores"""
        import json
        data_path = Path(data_dir)
        
        images_found = []
        json_count = 0
        
        for img_path in list(data_path.glob('*.jpg')) + list(data_path.glob('*.png')):
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                # Load from JSON metadata
                try:
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                    self.images.append(str(img_path))
                    self.scores.append([
                        metadata.get('overall', 5.0),
                        metadata.get('composition', 5.0),
                        metadata.get('color', 5.0),
                        metadata.get('contrast', 5.0),
                        metadata.get('focus', 5.0)
                    ])
                    json_count += 1
                except Exception as e:
                    print(f"⚠ Warning: Could not load JSON for {img_path}: {e}")
                    # Fall through to use default scores
            else:
                images_found.append(img_path)
        
        # If no JSON files found, use default scores for all images
        if json_count == 0 and images_found:
            print(f"⚠ Warning: No JSON metadata files found in {data_dir}")
            print(f"  Using default scores (5.0 for all metrics) for {len(images_found)} images.")
            print(f"  For proper training, provide either:")
            print(f"    - A CSV file with columns: path, overall, composition, color, contrast, focus")
            print(f"    - JSON files alongside images with aesthetic scores")
            for img_path in images_found:
                self.images.append(str(img_path))
                self.scores.append([5.0, 5.0, 5.0, 5.0, 5.0])  # Default scores
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        scores = self.scores[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(scores, dtype=torch.float32)


class LowLightDataset(Dataset):
    """
    Dataset for low-light enhancement
    Expects paired images: low/ and normal/ directories
    Or a CSV with low_path and normal_path columns
    """
    def __init__(self, data_dir, transform=None, csv_path=None):
        self.transform = transform
        self.low_images = []
        self.normal_images = []
        
        if csv_path:
            self._load_from_csv(csv_path)
        else:
            self._load_from_directory(data_dir)
    
    def _load_from_csv(self, csv_path):
        """Load from CSV with paired paths"""
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.low_images = df['low_path'].tolist()
        self.normal_images = df['normal_path'].tolist()
    
    def _load_from_directory(self, data_dir):
        """Load from directory structure
        
        Supports multiple formats:
        - low/ and normal/ directories
        - low/ and high/ directories (LOL dataset format)
        """
        data_path = Path(data_dir)
        
        # Check if data directory exists
        if not data_path.exists():
            print(f"❌ ERROR: Data directory does not exist: {data_path}")
            return
        
        if not data_path.is_dir():
            print(f"❌ ERROR: Path is not a directory: {data_path}")
            return
        
        # Try different directory name combinations
        low_dir = data_path / 'low'
        normal_dir = data_path / 'normal'
        high_dir = data_path / 'high'
        
        # Determine which format is used
        if low_dir.exists() and normal_dir.exists():
            # Standard format: low/ and normal/
            low_files = {f.stem: f for f in list(low_dir.glob('*.jpg')) + list(low_dir.glob('*.png'))}
            normal_files = {f.stem: f for f in list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.png'))}
            
            # Find matching pairs
            common_stems = set(low_files.keys()) & set(normal_files.keys())
            for stem in common_stems:
                self.low_images.append(str(low_files[stem]))
                self.normal_images.append(str(normal_files[stem]))
            
            if common_stems:
                print(f"✓ Loaded {len(common_stems)} image pairs from {data_path} (low/ and normal/)")
            else:
                print(f"⚠ WARNING: No matching image pairs found in {data_path}")
                print(f"   Found {len(low_files)} low images and {len(normal_files)} normal images")
                print(f"   Images must have matching filenames (excluding extension) to be paired")
        
        elif low_dir.exists() and high_dir.exists():
            # LOL dataset format: low/ and high/
            low_files = {f.stem: f for f in list(low_dir.glob('*.jpg')) + list(low_dir.glob('*.png'))}
            high_files = {f.stem: f for f in list(high_dir.glob('*.jpg')) + list(high_dir.glob('*.png'))}
            
            # Find matching pairs
            common_stems = set(low_files.keys()) & set(high_files.keys())
            for stem in common_stems:
                self.low_images.append(str(low_files[stem]))
                self.normal_images.append(str(high_files[stem]))  # high = normal-light
            
            if common_stems:
                print(f"✓ Loaded {len(common_stems)} image pairs from {data_path} (low/ and high/)")
            else:
                print(f"⚠ WARNING: No matching image pairs found in {data_path}")
                print(f"   Found {len(low_files)} low images and {len(high_files)} high images")
                print(f"   Images must have matching filenames (excluding extension) to be paired")
        else:
            print(f"⚠ WARNING: Could not find expected directory structure in {data_path}")
            print(f"   Expected one of:")
            print(f"     - {low_dir} and {normal_dir}")
            print(f"     - {low_dir} and {high_dir} (LOL dataset format)")
            if not low_dir.exists():
                print(f"   Missing: {low_dir}")
            if not normal_dir.exists() and not high_dir.exists():
                print(f"   Missing: {normal_dir} or {high_dir}")
    
    def __len__(self):
        return len(self.low_images)
    
    def __getitem__(self, idx):
        low_path = self.low_images[idx]
        normal_path = self.normal_images[idx]
        
        # Load images
        low_img = cv2.imread(low_path)
        normal_img = cv2.imread(normal_path)
        
        if low_img is None or normal_img is None:
            raise ValueError(f"Could not load images: {low_path}, {normal_path}")
        
        # Convert BGR to RGB
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        normal_img = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            low_img = self.transform(low_img)
            normal_img = self.transform(normal_img)
        
        return low_img, normal_img


class LightingDataset(Dataset):
    """
    Dataset for lighting/brightness assessment
    Expects CSV with columns: path, brightness_score (0-10)
    Or directory with labels in filenames
    """
    def __init__(self, data_dir, transform=None, csv_path=None):
        self.transform = transform
        self.images = []
        self.scores = []
        
        if csv_path:
            self._load_from_csv(csv_path)
        else:
            self._load_from_directory(data_dir)
    
    def _load_from_csv(self, csv_path):
        """Load from CSV"""
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.images = df['path'].tolist()
        self.scores = df['brightness_score'].tolist()
    
    def _load_from_directory(self, data_dir):
        """Load from directory with labels in filenames or JSON"""
        import json
        data_path = Path(data_dir)
        
        for img_path in list(data_path.glob('*.jpg')) + list(data_path.glob('*.png')):
            # Try JSON first
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    score = metadata.get('brightness_score', 5.0)
            else:
                # Try to extract from filename (e.g., image_7.5.jpg)
                try:
                    parts = img_path.stem.split('_')
                    score = float(parts[-1]) if parts[-1].replace('.', '').isdigit() else 5.0
                except:
                    score = 5.0
            
            self.images.append(str(img_path))
            self.scores.append(score)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        score = self.scores[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(score, dtype=torch.float32)


def get_transforms(train=True, input_size=224):
    """Get data augmentation transforms"""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def get_enhancer_transforms(train=True, input_size=256):
    """Get transforms for enhancement (no normalization to [0,1])"""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),  # Already normalizes to [0, 1]
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

