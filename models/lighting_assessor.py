"""
Lighting/Brightness Assessment Model
Uses MobileNet-V2 for efficient brightness scoring
"""
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from torchvision import transforms


class LightingAssessor:
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
        """Build MobileNet-V2 based lighting assessor"""
        try:
            # Try new weights API (torchvision >= 0.13)
            from torchvision.models import MobileNet_V2_Weights
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # Fallback to old pretrained API
            model = models.mobilenet_v2(pretrained=True)
        
        # Replace classifier for regression (brightness score 0-10)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)  # Single output for brightness score
        )
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_weights(self, weights_path):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            # Handle both direct state_dict and checkpoint dict formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✓ Loaded lighting assessor weights from {weights_path}")
        except FileNotFoundError:
            print(f"⚠ Warning: Weights not found at {weights_path}. Using fallback method.")
        except Exception as e:
            print(f"⚠ Warning: Could not load weights: {e}. Using fallback method.")
    
    def assess(self, image):
        """
        Assess lighting/brightness of an image
        
        Args:
            image: numpy array (BGR format from OpenCV)
            
        Returns:
            tuple: (brightness_score (0-10), needs_enhancement (bool))
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Check if model has been properly trained
        if not hasattr(self.model, 'classifier') or len(list(self.model.classifier[-1].parameters())) == 0:
            return self._fallback_assessment(image)
        
        try:
            # Preprocess
            img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(img_tensor)
                brightness_score = output[0].item()
                
                # Ensure score is in 0-10 range
                brightness_score = max(0.0, min(10.0, brightness_score))
            
            # Determine if enhancement is needed (threshold at 4.0)
            needs_enhancement = brightness_score < 4.0
            
            return brightness_score, needs_enhancement
        except Exception as e:
            print(f"Error in lighting assessment: {e}")
            return self._fallback_assessment(image)
    
    def _fallback_assessment(self, image):
        """Fallback: Simple HSV brightness calculation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv[:,:,2])
        brightness_score = (avg_brightness / 255.0) * 10
        needs_enhancement = avg_brightness < 80
        return brightness_score, needs_enhancement

