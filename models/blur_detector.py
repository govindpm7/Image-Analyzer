"""
Blur Detection Model
Uses ResNet-18 for binary classification (sharp vs blurry)
"""
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from torchvision import transforms


class BlurDetector:
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _build_model(self):
        """Build ResNet-18 based blur detector"""
        try:
            # Try new weights API (torchvision >= 0.13)
            from torchvision.models import ResNet18_Weights
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # Fallback to old pretrained API
            model = models.resnet18(pretrained=True)
        # Replace final layer for binary classification
        model.fc = nn.Linear(512, 2)  # Sharp vs Blurry
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_weights(self, weights_path):
        """Load trained model weights"""
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"✓ Loaded blur detector weights from {weights_path}")
        except FileNotFoundError:
            print(f"⚠ Warning: Weights not found at {weights_path}. Using pretrained ResNet-18.")
        except Exception as e:
            print(f"⚠ Warning: Could not load weights: {e}. Using pretrained ResNet-18.")
    
    def predict(self, image):
        """
        Predict blur score for an image
        
        Args:
            image: numpy array (BGR format from OpenCV)
            
        Returns:
            float: Blur score from 0-10 (higher = sharper)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Fallback to classical CV method if model not available
        if not hasattr(self.model, 'fc') or self.model.fc.out_features != 2:
            return self._classical_blur_detection(image)
        
        try:
            # Preprocess image
            img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                sharp_prob = probabilities[0][1].item()  # Probability of being sharp
            
            # Convert probability to 0-10 score
            blur_score = sharp_prob * 10.0
            
            return blur_score
        except Exception as e:
            print(f"Error in blur prediction: {e}")
            return self._classical_blur_detection(image)
    
    def _classical_blur_detection(self, image):
        """Fallback: Classical CV blur detection using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-10 scale
        # Threshold: <100 = blurry, >100 = sharp
        # Map to 0-10 scale
        if laplacian_var < 50:
            score = (laplacian_var / 50.0) * 4.0  # 0-4 range
        elif laplacian_var < 200:
            score = 4.0 + ((laplacian_var - 50) / 150.0) * 4.0  # 4-8 range
        else:
            score = 8.0 + min((laplacian_var - 200) / 100.0, 1.0) * 2.0  # 8-10 range
        
        return min(max(score, 0.0), 10.0)


