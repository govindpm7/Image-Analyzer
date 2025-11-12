"""
Aesthetic Scoring Model
Uses EfficientNet-B0 for aesthetic quality assessment
"""
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from torchvision import transforms


class AestheticScorer:
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
        """Build EfficientNet-B0 based aesthetic scorer"""
        model = models.efficientnet_b0(pretrained=True)
        
        # Multi-task output
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 5)  # 5 scores: overall, composition, color, contrast, focus
        )
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_weights(self, weights_path):
        """Load trained model weights"""
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"✓ Loaded aesthetic scorer weights from {weights_path}")
        except FileNotFoundError:
            print(f"⚠ Warning: Weights not found at {weights_path}. Using feature-based scoring.")
        except Exception as e:
            print(f"⚠ Warning: Could not load weights: {e}. Using feature-based scoring.")
    
    def score(self, image):
        """
        Score image aesthetics
        
        Args:
            image: numpy array (BGR format from OpenCV)
            
        Returns:
            dict: Scores for overall, composition, color, contrast, focus (0-10 each)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Check if model has been properly trained
        if not hasattr(self.model, 'classifier') or len(list(self.model.classifier[-1].parameters())) == 0:
            return self._feature_based_scoring(image)
        
        try:
            # Preprocess
            img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(img_tensor)
                scores = outputs[0].cpu().numpy()
                
                # Normalize to 0-10 scale (assuming model outputs are in reasonable range)
                # If trained properly, scores should already be in 0-10 range
                # Otherwise, we'll use sigmoid to map to 0-10
                scores = 1 / (1 + np.exp(-scores)) * 10  # Sigmoid to 0-10
            
            return {
                'overall': float(np.clip(scores[0], 0, 10)),
                'composition': float(np.clip(scores[1], 0, 10)),
                'color': float(np.clip(scores[2], 0, 10)),
                'contrast': float(np.clip(scores[3], 0, 10)),
                'focus': float(np.clip(scores[4], 0, 10))
            }
        except Exception as e:
            print(f"Error in aesthetic scoring: {e}")
            return self._feature_based_scoring(image)
    
    def _feature_based_scoring(self, image):
        """Fallback: Feature-based aesthetic scoring using OpenCV"""
        scores = {}
        
        # Overall score (average of components)
        composition = self._check_rule_of_thirds(image)
        color = self._analyze_color_harmony(image)
        contrast = self._calculate_contrast(image)
        focus = self._calculate_edge_density(image)
        
        scores['composition'] = composition
        scores['color'] = color
        scores['contrast'] = contrast
        scores['focus'] = focus
        scores['overall'] = (composition + color + contrast + focus) / 4.0
        
        return scores
    
    def _check_rule_of_thirds(self, image):
        """Detect if focal points align with rule of thirds"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect interest points using ORB
        orb = cv2.ORB_create(nfeatures=50)
        keypoints = orb.detect(gray, None)
        
        if len(keypoints) == 0:
            return 5.0  # Neutral score
        
        # Rule of thirds grid intersections
        grid_h, grid_w = h // 3, w // 3
        intersections = [
            (grid_w, grid_h), (2*grid_w, grid_h),
            (grid_w, 2*grid_h), (2*grid_w, 2*grid_h)
        ]
        
        score = 0
        tolerance = min(w, h) * 0.15
        
        for kp in keypoints:
            for ix, iy in intersections:
                dist = np.sqrt((kp.pt[0]-ix)**2 + (kp.pt[1]-iy)**2)
                if dist < tolerance:
                    score += 1
        
        return min(score / 3.0, 10.0)
    
    def _analyze_color_harmony(self, image):
        """Analyze color harmony"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color distribution
        hue = hsv[:,:,0].flatten()
        saturation = hsv[:,:,1].flatten()
        
        # Good harmony if saturation is balanced (not too low, not too high)
        avg_saturation = np.mean(saturation)
        if 50 < avg_saturation < 200:
            score = 7.0
        elif avg_saturation < 30:
            score = 5.0  # Too desaturated
        else:
            score = 6.0  # Too saturated
        
        # Bonus for color variety
        unique_hues = len(np.unique(hue // 10))
        if 3 <= unique_hues <= 8:
            score += 1.0
        
        return min(score, 10.0)
    
    def _calculate_contrast(self, image):
        """Calculate overall contrast ratio"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        contrast = gray.std()
        # Normalize to 0-10 scale
        return min(contrast / 5.0, 10.0)
    
    def _calculate_edge_density(self, image):
        """Measure sharpness via edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.sum(edges > 0) / edges.size
        return edge_ratio * 100  # Scale to 0-10


